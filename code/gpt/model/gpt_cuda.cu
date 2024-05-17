#include "stdafx.h"
#define KERNEL_UNIT "cuda_gpt/"
#include "gpt_cuda.cuh"
#include <lib/cuda/cuda_util.cuh>
#include <lib/cuda/cuda_graph.cuh>
#include <lib/cuda/cuda_matmul.cuh>
#include <lib/cuda/cuda_mma.cuh>
#include <lib/cuda/vec_util.cuh>
#include "par_matrix_cuda.cuh"
#include <gpt/data/data.h>
#include <gpt/att/nodes_batch.h>
#include <lib/random/rand_utils.h>
#include <lib/math/matrix_utils.h>
#include <emmintrin.h>


using namespace NCuda;

constexpr float LOG2 = 0.693147f;

constexpr yint PREDICTION_ARR_SZ = 4096;

namespace NCUDA_GPT
{

///////////////////////////////////////////////////////////////////////////////////////////////////
// vec types

// element type of intermediate vectors (QK, QV, K, V)
// i8
typedef i8 TVecFloat;
typedef half TValueVecFloat;
typedef i8 TKVFloat; // much faster forward pass but precision is lost (can be replaced with half)
typedef T4SMemI8Tile T4SMemVecFloatTile;
typedef int TVecFloatMMAResult;
constexpr float VEC_SCALE = DISCR_SCALE;
//// half
//typedef half TVecFloat;
//typedef half TValueVecFloat;
//typedef half TKVFloat;
//typedef T4SMemHalfTile T4SMemVecFloatTile;
//typedef float TVecFloatMMAResult;
//constexpr float VEC_SCALE = 1;


// element type of state vector
typedef float TStateFloat;
//typedef half TStateFloat;


///////////////////////////////////////////////////////////////////////////////////////////////////
// utils

constexpr __host__ __device__ float CalcDotScale(int stateDim)
{
    return constexprSqrt(1.f / stateDim) / LOG2; // log(2) to use exp2f()
}



// gradient scale
__forceinline __device__ float GetGradScale(float gradMaxNorm)
{
    const float TARGET_MAX_NORM = 1;
    if (gradMaxNorm < TARGET_MAX_NORM / 4 || gradMaxNorm > TARGET_MAX_NORM * 4) {
        if (gradMaxNorm == 0) {
            return 1; // zero gradients can be multiplied by any number
        }
        return TARGET_MAX_NORM / gradMaxNorm;
    }
    return 1;
}


__forceinline __device__ float GetAttentionDecay(int dist, float alibiSlope, float alibiHyper)
{
    return -alibiSlope * dist + alibiHyper / dist;
}


inline __device__ void CvtToVecFloat(half *p, float x)
{
    *p = x;
}

inline __device__ void CvtToVecFloat(i8 *p, float x)
{
    *p = CvtToI8(x);
}



#include "gpt_i8.cuh"
#include "gpt_attention.cuh"
#include "gpt_combiner.cuh"


///////////////////////////////////////////////////////////////////////////////////////////////////
// AddProduct implementation

// array of state vectors should be multiple of this value
const int SAMPLE_ARR_CHUNK = MM_TILE_LARGE;


struct TCudaAttentionParams : public TThrRefBase
{
    TIntrusivePtr<TCudaModelMatrix> QK;
    TIntrusivePtr<TCudaModelMatrix> QV;
    TIntrusivePtr<TCudaModelMatrix> K;
    TIntrusivePtr<TCudaModelMatrix> V;
    TIntrusivePtr<TCudaModelMatrix> Combiner;
    TOpParameter<float> AlibiSlope;
    TOpParameter<float> AlibiHyper;
    bool WideLayer = true;

    TCudaAttentionParams(yint deviceId, TIntrusivePtr<TCudaModelMatrixScale> cudaMatrixScale, const TAttentionParams &att)
    {
        QK = new TCudaModelMatrix(deviceId, cudaMatrixScale, att.QK, MM_MEM_DEVICE);
        QV = new TCudaModelMatrix(deviceId, cudaMatrixScale, att.QV, MM_MEM_DEVICE);
        K = new TCudaModelMatrix(deviceId, cudaMatrixScale, att.K, MM_MEM_DEVICE);
        V = new TCudaModelMatrix(deviceId, cudaMatrixScale, att.V, MM_MEM_DEVICE);
        Combiner = new TCudaModelMatrix(deviceId, cudaMatrixScale, att.Combiner, MM_MEM_DEVICE);
        UpteLayerParams(att);
    }
    void UpteLayerParams(const TAttentionParams &att)
    {
        AlibiSlope.Set(att.AlibiSlope);
        AlibiHyper.Set(att.AlibiHyper);
        WideLayer = att.WideLayer;
    }
    void CopyToDevice(TIntrusivePtr<NCuda::TGraph> c)
    {
        QK->CopyToDevice(c);
        QV->CopyToDevice(c);
        K->CopyToDevice(c);
        V->CopyToDevice(c);
        Combiner->CopyToDevice(c);
    }
};


struct TAttentionComputeCtx
{
    yint StateDim = 0;
    yint TTDim = 0;
    TCuda2DArray<TVecFloat> QKState8;
    TCuda2DArray<TVecFloat> QVState8;
    TCuda2DArray<TVecFloat> KState8;
    TCuda2DArray<TValueVecFloat> VState8; // separate type to avoid expensive i8->half conversion in attention compute
    TCuda2DArray<TKVFloat> KVState8;
    TCuda2DArray<half> DQKState;
    TCuda2DArray<half> DQVState;
    TCuda2DArray<half> DKState;
    TCuda2DArray<half> DVState;
    TCuda2DArray<half> DKVState;
    TCuda2DArray<half> ValLookup8;
    TCuda2DArray<half> DValLookup;
    //
    TCuda2DArray<float> DeltaQK;
    TCuda2DArray<float> DeltaQV;
    TCuda2DArray<float> DeltaK;
    TCuda2DArray<float> DeltaV;
    TCuda2DArray<float> DeltaCombiner;
    //
    TCudaVector<float> SumWeightLog;
    TCudaVector<float> DScale;
    TCudaVector<float> QKStateScale; // scale to get correct QK state vector, should by multiplied by VEC_SCALE (from using normstate8 as source for QState compute)
    TCudaVector<float> QVStateScale; // scale to get correct QV state vector, should by multiplied by VEC_SCALE (from using normstate8 as source for QState compute)
    TCudaVector<float> KStateScale;
    TCudaVector<float> VStateScale;
    //
    TCuda2DArray<TStateFloat> NewState;

    void AllocateCuda(yint dim, yint qDim, yint ttDim, yint len)
    {
        if (QKState8.GetYSize() != len || StateDim != dim || TTDim != ttDim) {
            StateDim = dim;
            TTDim = ttDim;
            QKStateScale.AllocateCuda(len);
            QVStateScale.AllocateCuda(len);
            KStateScale.AllocateCuda(len);
            VStateScale.AllocateCuda(len);
            QKState8.AllocateCuda(qDim, len);
            QVState8.AllocateCuda(qDim, len);
            KState8.AllocateCuda(ttDim, len);
            VState8.AllocateCuda(ttDim, len);
            KVState8.AllocateCuda(GetCombinerWidth(ttDim), len);
            DQKState.AllocateCuda(qDim, len);
            DQVState.AllocateCuda(qDim, len);
            DKState.AllocateCuda(ttDim, len);
            DVState.AllocateCuda(ttDim, len);
            DKVState.AllocateCuda(GetCombinerWidth(ttDim), len);
            ValLookup8.AllocateCuda(ttDim, len);
            DValLookup.AllocateCuda(ttDim, len);
            //
            DeltaQK.AllocateCuda(dim, qDim);
            DeltaQV.AllocateCuda(dim, qDim);
            DeltaK.AllocateCuda(dim, ttDim);
            DeltaV.AllocateCuda(dim, ttDim);
            DeltaCombiner.AllocateCuda(GetCombinerWidth(ttDim), dim);
            //
            SumWeightLog.AllocateCuda(len);
            DScale.AllocateCuda(len);
            //
            NewState.AllocateCuda(dim, len);
        }
    }
};


struct TLayerGradComputeCtx
{
    yint StateDim = 0;
    TCuda2DArray<float> DNormState;

    void AllocateCuda(yint dim, yint qDim, yint ttDim, yint len)
    {
        (void)qDim;
        (void)ttDim;
        if (DNormState.GetYSize() < len || StateDim != dim) {
            StateDim = dim;
            DNormState.AllocateCuda(dim, len);
        }
    }
};


template <class T, int SIZE>
struct TComputeCtxSet
{
    T CtxArr[SIZE];
    yint CurCtx = 0;

    void AllocateCuda(yint dim, yint qDim, yint ttDim, yint len)
    {
        for (yint k = 0; k < SIZE; ++k) {
            CtxArr[k].AllocateCuda(dim, qDim, ttDim, len);
        }
    }
    T &GetCtx()
    {
        CurCtx = (CurCtx + 1) % SIZE;
        return CtxArr[CurCtx];
    }
};


struct TAttentionGroupData
{
    TCudaVector<TAttentionSpanGroup<ATT_GROUP>> AttSpans;
    TCudaVector<int> AttSpanPtr;
    TCudaVector<TAttentionSpanGroup<ATT_GROUP>> RevAttSpans;
    TCudaVector<int> RevAttSpanPtr;

    void Allocate(const TModelDim &modelDim, int maxLen, int groupPerBlock)
    {
        int spanGroups = DivCeil(maxLen, ATT_GROUP);
        AttSpans.Allocate(spanGroups * groupPerBlock);
        RevAttSpans.Allocate(spanGroups * groupPerBlock);
        AttSpanPtr.Allocate(spanGroups + 1);
        RevAttSpanPtr.Allocate(spanGroups + 1);
    }

    template <int N>
    void AssignAttentionGroups(TStream &stream, TAttentionInfoGrouped<N> *pGroups, yint attGroupCount, TCudaVector<TAttentionSpanGroup<N>> *pSpans, TCudaVector<int> *pSpanPtr)
    {
        while (pGroups->GetGroupCount() < attGroupCount) {
            pGroups->AddEmptySpanGroup();
        }
        pSpans->Put(stream, pGroups->SpanGroups);
        pSpanPtr->Put(stream, pGroups->SpanGroupPtr);
    }

    void Init(TStream &stream, yint lenBufferSize, TAttentionInfoGrouped<ATT_GROUP> *pAttGroups, TAttentionInfoGrouped<ATT_GROUP> *pRevAttGroups)
    {
        int attGroupCount = lenBufferSize / ATT_GROUP;
        AssignAttentionGroups(stream, pAttGroups, attGroupCount, &AttSpans, &AttSpanPtr);
        AssignAttentionGroups(stream, pRevAttGroups, attGroupCount, &RevAttSpans, &RevAttSpanPtr);
    }
};


struct TWindowParams
{
    int Offset;
    int Len;
    int LenMMTiles;
    int LenSampleBlocks;

    TWindowParams(int winOffset, int totalLen, int maxWinLen)
    {
        Offset = winOffset;
        Len = Min<int>(totalLen - Offset, maxWinLen);
        LenMMTiles = DivCeil(Len, MM_TILE);
        LenSampleBlocks = LenMMTiles * MM_TILE / SAMPLE_BLOCK;
    }
};


struct TComputeParams
{
    TOpParameter<int> InvLabelCount;

    int LenBufferSize = 0;
    TOpParameter<int> Len;
    TOpParameter<int> LenSampleBlocks;
    TOpParameter<int> LenTiles;
    TOpParameter<int> LenAttTiles;
    TOpParameter<int> LenMMTiles;
    TOpParameter<int> LenMMLargeTiles;
    TCudaVector<ui32> DropTable;
    TCudaVector<int> SampleIndex;
    TOpParameter<int> FinalOffset;
    TOpParameter<int> FinalLen;
    TOpParameter<int> FinalLenMMTiles;
    TOpParameter<int> FinalLenSampleBlocks;

    void Allocate(const TModelDim &modelDim, int maxLen)
    {
        DropTable.AllocateWC(CalcDropTableSize(modelDim));
        SampleIndex.AllocateWC(maxLen);
    }

    void Init(TStream &stream, yint len, const TVector<int> &sampleIndex, const TVector<ui32> &dropTable)
    {
        Y_ASSERT(MM_TILE_LARGE >= SAMPLE_BLOCK);
        Y_ASSERT(MM_TILE_LARGE >= TILE);
        Len.Set(len);
        yint nLargeTiles = DivCeil(len, MM_TILE_LARGE);
        LenBufferSize = nLargeTiles * MM_TILE_LARGE;
        LenSampleBlocks.Set(nLargeTiles * MM_TILE_LARGE / SAMPLE_BLOCK);
        LenTiles.Set(nLargeTiles * MM_TILE_LARGE / TILE);
        LenAttTiles.Set(nLargeTiles * MM_TILE_LARGE / ATT_GROUP);
        LenMMTiles.Set(nLargeTiles * MM_TILE_LARGE / MM_TILE);
        LenMMLargeTiles.Set(nLargeTiles);
        // dropout
        Y_VERIFY(YSize(dropTable) <= DropTable.GetSize());
        DropTable.Put(stream, dropTable);
        //
        SampleIndex.Put(stream, sampleIndex);
    }

    void SetInvLabelCount(yint invLabelCount)
    {
        InvLabelCount.Set(invLabelCount);
    }

    void SetFinalWindow(const TWindowParams &window)
    {
        FinalOffset.Set(window.Offset);
        FinalLen.Set(window.Len);
        FinalLenMMTiles.Set(window.LenMMTiles);
        FinalLenSampleBlocks.Set(window.LenSampleBlocks);
    }

    yint GetLenBufferSize() const { return LenBufferSize; }
};


struct TFragmentStates : public TThrRefBase
{
    TCuda2DArray<TVecFloat> NormState;
    TCudaVector<float> StateScale;

    void AllocateCuda(yint stateDim, yint len)
    {
        NormState.AllocateCuda(stateDim, len);
        StateScale.AllocateCuda(len);
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
// kernels
template <int STATE_DIM>
static __device__ void ApplyDropout(float *x, ui32 *dropMask)
{
    for (int k = 0; k < WCOUNT; ++k) {
        if ((dropMask[k] & (1 << threadIdx.x)) == 0) {
            x[k] = 0;
        }
    }
}


template <int STATE_DIM>
__global__ void NormalizeAndDropVecs(
    int len,
    TCuda2DPtr<TStateFloat> state, ui32 *dropTable,
    TCuda2DPtr<TVecFloat> normState8, float *normStateScale)
{
    int t = blockIdx.x * SAMPLE_BLOCK + threadIdx.y;

    // normalize
    if (t < len) {
        float v[WCOUNT];
        LoadVec<STATE_DIM>(v, state[t]);
        ApplyDropout<STATE_DIM>(v, dropTable);
        float sum2 = CalcSum2<STATE_DIM>(v);
        if (sum2 > 0) {
            float sko = sqrtf(sum2 / STATE_DIM);
            float discrScale = sko * VEC_SCALE;
            // discretize
            TVecFloat vNorm[WCOUNT];
            for (int k = 0; k < WCOUNT; ++k) {
                CvtToVecFloat(&vNorm[k], v[k] / discrScale);
            }
            StoreVec<STATE_DIM>(normState8[t], vNorm);
            if (threadIdx.x == 0) {
                normStateScale[t] = discrScale;
            }
            return;
        }
    }
    StoreZeroVec<STATE_DIM>(normState8[t]);
    if (threadIdx.x == 0) {
        normStateScale[t] = 1;
    }
}
KERNEL_BLOCK_SIZE(NormalizeAndDropVecs, WARP_SIZE, SAMPLE_BLOCK);


template <int STATE_DIM, class T>
struct TSumVecs
{
    TVector<TCuda2DArray<T> *> Args;
    TIntrusivePtr<TGraph> Graph;
    TComputeParams *Params = 0;

    TSumVecs(TIntrusivePtr<TGraph> c, TComputeParams *params) : Graph(c), Params(params) {}
    // have to call Add() first since it clears the buffer
    void Add(TCuda2DArray<T> *p)
    {
        if (!IsInSet(Args, p)) {
            Graph->ClearMem(*p, Params->Len);
            Args.push_back(p);
        }
    }
    void Sum(TCuda2DArray<T> *pNewState)
    {
        if (Args.empty()) {
            return;
        }
        while (YSize(Args) > 2) {
            CudaCall(Graph, SumVecsKernel2<STATE_DIM, T>).Grid(Params->LenSampleBlocks)(Params->Len, *Args[0], *Args[1]).Write(Args.back());
            Args.erase(Args.begin(), Args.begin() + 2);
        }
        if (YSize(Args) == 2) {
            CudaCall(Graph, SumVecsKernel2<STATE_DIM, T>).Grid(Params->LenSampleBlocks)(Params->Len, *Args[0], *Args[1]).Write(pNewState);
        } else {
            CudaCall(Graph, SumVecsKernel1<STATE_DIM, T>).Grid(Params->LenSampleBlocks)(Params->Len, *Args[0]).Write(pNewState);
        }
    }
};


// int8 version
template <int STATE_DIM, int DST_DIM, class TDst>
void MulForward(TIntrusivePtr<TGraph> c, TComputeParams *pParams, TCuda2DArray<i8> &normState8, TIntrusivePtr<TCudaModelMatrix> pTransform,
    TCuda2DArray<half> *pTempBuf, TCuda2DArray<TDst> *pRes, TCudaVector<float> *pResScale)
{
    constexpr int dstTiles = DST_DIM / MM_TILE;
    I8MatMulXYoZYeXZ<STATE_DIM>(c, normState8, pTransform->GetFast(), pTempBuf, pParams->LenMMTiles, dstTiles, TStoreI8Scaled(MATRIX_MULT_I8_SCALE));
    constexpr float tempScale = (1 / MATRIX_MULT_I8_SCALE);
    if (pResScale) {
        TCudaPOD<float> scale = pTransform->GetScale();
        CudaCall(c, ConvertHalfToVecFloat<DST_DIM, TDst>).Grid(pParams->LenSampleBlocks)(pParams->Len, *pTempBuf, tempScale, scale).Write(pRes, pResScale);
    } else {
        CudaCall(c, ConvertHalfToVecFloat<DST_DIM, TDst>).Grid(pParams->LenSampleBlocks)(pParams->Len, *pTempBuf, tempScale, nullptr).Write(pRes)(nullptr);
    }
}

// generic version
template <int STATE_DIM, int DST_DIM>
void MulForward(TIntrusivePtr<TGraph> c, TComputeParams *pParams, TCuda2DArray<half> &normState8, TIntrusivePtr<TCudaModelMatrix> pTransform,
    TCuda2DArray<half> *pTempBuf, TCuda2DArray<half> *pRes, TCudaVector<float> *pResScale)
{
    constexpr int stateTiles = STATE_DIM / MM_TILE;
    constexpr int dstTiles = DST_DIM / MM_TILE;
    (void)MATRIX_MULT_I8_SCALE;
    MatMulXYoZYeXZ(c, normState8, pTransform->GetFast(), pTempBuf, pParams->LenMMTiles, stateTiles, dstTiles, TStore());
    if (pResScale) {
        TCudaPOD<float> scale = pTransform->GetScale();
        CudaCall(c, ConvertHalfToVecFloat<DST_DIM, half>).Grid(pParams->LenSampleBlocks)(pParams->Len, *pTempBuf, 1.0f, scale).Write(pRes, pResScale);
    } else {
        CudaCall(c, ConvertHalfToVecFloat<DST_DIM, half>).Grid(pParams->LenSampleBlocks)(pParams->Len, *pTempBuf, 1.0f, nullptr).Write(pRes)(nullptr);
    }
}


// int8 version
template <int STATE_DIM, int TT_DIM, class TRes>
void Combine(TIntrusivePtr<TGraph> c, TComputeParams *pParams, TCudaAttentionParams &att, TAttentionComputeCtx &attCtx,
    TCuda2DArray<i8> &kvState8, TCuda2DArray<i8> &combiner,
    TCuda2DArray<TRes> *pTargetState)
{
    constexpr int stateLargeTiles = STATE_DIM / MM_TILE_LARGE;
    TCudaPOD<float> scaleCombiner = att.Combiner->GetScale();
    I8MatMulXYoZYeXZlarge<GetCombinerWidth(TT_DIM)>(c, attCtx.KVState8, att.Combiner->GetFast(), pTargetState, pParams->LenMMLargeTiles, stateLargeTiles, TStoreI8AddScaled(scaleCombiner, VEC_SCALE));
}

// generic version
template <int STATE_DIM, int TT_DIM, class T1, class T2, class TRes>
void Combine(TIntrusivePtr<TGraph> c, TComputeParams *pParams, TCudaAttentionParams &att, TAttentionComputeCtx &attCtx,
    TCuda2DArray<T1> &kvState8, TCuda2DArray<T2> &combiner,
    TCuda2DArray<TRes> *pTargetState)
{
    constexpr int kvTiles = GetCombinerWidth(TT_DIM) / MM_TILE;
    constexpr int stateTiles = STATE_DIM / MM_TILE;
    TCudaPOD<float> scaleCombiner = att.Combiner->GetScale();
    MatMulXYoZYeXZ(c, attCtx.KVState8, att.Combiner->GetFast(), pTargetState, pParams->LenMMTiles, kvTiles, stateTiles, TStoreAddScaled(scaleCombiner, VEC_SCALE));
}



// compute product of two attention lookups
template <int STATE_DIM, int Q_DIM, int TT_DIM, int ATT_BUFS>
static void AddLookupProduct(
    TIntrusivePtr<TGraph> c, bool copyModelToDevice, TComputeParams *pParams,
    TAttentionGroupData *pAttGD, TAttentionGroupData *pWideAttGD,
    TComputeCtxSet<TAttentionComputeCtx, ATT_BUFS> *pAttCtxSet,
    const TVector<TIntrusivePtr<TCudaAttentionParams>> &layerAtt,
    TCuda2DArray<TStateFloat> *pState, TFragmentStates *pKeepState)
{
    yint attCount = YSize(layerAtt);

    TCuda2DArray<TVecFloat> &normState8 = pKeepState->NormState;
    CudaCall(c, NormalizeAndDropVecs<STATE_DIM>).Grid(pParams->LenSampleBlocks)
        (pParams->Len, *pState, pParams->DropTable).Write(&normState8, &pKeepState->StateScale);

    TSumVecs<STATE_DIM, TStateFloat> sumState(c, pParams);

    for (yint at = 0; at < attCount; ++at) {
        TCudaAttentionParams &att = *layerAtt[at];
        TAttentionComputeCtx &attCtx = pAttCtxSet->GetCtx();
        TAttentionGroupData &attGD = att.WideLayer ? *pWideAttGD : *pAttGD;
        constexpr int TT_GROUPS = TT_DIM / TILE_GROUP_SIZE;

        if (copyModelToDevice) {
            att.CopyToDevice(c);
        }

        // mul forwardz
        MulForward<STATE_DIM, Q_DIM>(c, pParams, normState8, att.QK, &attCtx.DQKState, &attCtx.QKState8, &attCtx.QKStateScale);
        MulForward<STATE_DIM, Q_DIM>(c, pParams, normState8, att.QV, &attCtx.DQVState, &attCtx.QVState8, &attCtx.QVStateScale);
        MulForward<STATE_DIM, TT_DIM>(c, pParams, normState8, att.K, &attCtx.DKState, &attCtx.KState8, nullptr);
        MulForward<STATE_DIM, TT_DIM>(c, pParams, normState8, att.V, &attCtx.DVState, &attCtx.VState8, nullptr);

        // compute attention
        CudaCall(c, ComputeAttentionValLookup<Q_DIM, TT_DIM>).Block(WARP_SIZE, ATT_LOOKUP_BATCH + TT_GROUPS).Grid(pParams->LenAttTiles)
            (attCtx.QKState8, attCtx.QKStateScale, attCtx.QVState8, attCtx.QVStateScale, attCtx.VState8, attGD.AttSpans, attGD.AttSpanPtr, att.AlibiSlope, att.AlibiHyper)
            .Write(&attCtx.SumWeightLog, &attCtx.ValLookup8);

        CudaCall(c, KVProduct<TT_DIM>).Grid(pParams->LenSampleBlocks)
            (pParams->Len, attCtx.KState8, attCtx.ValLookup8)
            .Write(&attCtx.KVState8);

        TCuda2DArray<TStateFloat> *pTargetState = pState;
        if (attCount > 1) {
            pTargetState = &attCtx.NewState;
            sumState.Add(pTargetState);
        }

        Combine<STATE_DIM, TT_DIM>(c, pParams, att, attCtx, attCtx.KVState8, att.Combiner->GetFast(), pTargetState);
    }
    sumState.Sum(pState);
}



///////////////////////////////////////////////////////////////////////////////////////////////////
// backprop kernels and graph
template <int STATE_DIM>
__global__ void ComputeLayerStateGrad(
    int len, ui32 *dropTable,
    TCuda2DPtr<TVecFloat> normState, float *stateScale,
    TCuda2DPtr<float> dNormState,
    TCuda2DPtr<half> pStateGrad, float *nextStateGradMaxNorm
)
{
    int h = threadIdx.x;
    int warpId = threadIdx.y;
    int t = blockIdx.x * SAMPLE_BLOCK + warpId;
    if (t >= len) {
        StoreZeroVec<STATE_DIM>(pStateGrad[t]);
        return;
    }
    // states with and without target label
    float v[WCOUNT];
    LoadVec<STATE_DIM>(v, normState[t]);
    ScaleVec<STATE_DIM>(v, v, stateScale[t]);
    ApplyDropout<STATE_DIM>(v, dropTable);

    float vDNormState[WCOUNT];
    LoadVec<STATE_DIM>(vDNormState, dNormState[t]);

    float stateGrad[WCOUNT];
    StateNormalizeBackprop<STATE_DIM>(v, vDNormState, stateGrad);
    ApplyDropout<STATE_DIM>(stateGrad, dropTable);

    LoadVecAdd<STATE_DIM>(stateGrad, stateGrad, pStateGrad[t]);
    StoreVec<STATE_DIM>(pStateGrad[t], stateGrad);

    // compute grad max norm
    float gradMax = CalcLinf<STATE_DIM>(stateGrad);
    if (h == 0) {
        atomicMax((int *)nextStateGradMaxNorm, __float_as_int(gradMax));
    }
}
KERNEL_BLOCK_SIZE(ComputeLayerStateGrad, WARP_SIZE, SAMPLE_BLOCK);


template <int STATE_DIM>
__global__ void SumGrad(int len, TCuda2DPtr<half> src, TCuda2DPtr<half> dst)
{
    int t = blockIdx.x * SAMPLE_BLOCK + threadIdx.y;
    if (t < len) {
        float v[WCOUNT];
        LoadVec<STATE_DIM>(v, src[t]);
        LoadVecAdd<STATE_DIM>(v, v, dst[t]);
        StoreVec<STATE_DIM>(dst[t], v);
    }
}
KERNEL_BLOCK_SIZE(SumGrad, WARP_SIZE, SAMPLE_BLOCK);


template <int STATE_DIM>
__global__ void ScaleGrad(int len, float *gradMaxNorm, float *prevGradScale, TCuda2DPtr<half> grad1, float *gradScale)
{
    int t = blockIdx.x * SAMPLE_BLOCK + threadIdx.y;
    float gradScaleMult = GetGradScale(*gradMaxNorm);
    if (gradScaleMult != 1 && t < len) {
        float v[WCOUNT];
        LoadVec<STATE_DIM>(v, grad1[t]);
        ScaleVec<STATE_DIM>(v, v, gradScaleMult);
        StoreVec<STATE_DIM>(grad1[t], v);
    }
    if (t == 0 && threadIdx.x == 0) {
        CUDA_ASSERT(gradScaleMult != 0);
        *gradScale = *prevGradScale / gradScaleMult;
    }
}
KERNEL_BLOCK_SIZE(ScaleGrad, WARP_SIZE, SAMPLE_BLOCK);


// add gradient of product of two attention lookups
template <int STATE_DIM, int Q_DIM, int TT_DIM, int ATT_BUFS>
static void AddLookupProductBackprop(
    TIntrusivePtr<TGraph> c, int stepId, TComputeParams *pParams,
    TAttentionGroupData *pAttGD, TAttentionGroupData *pWideAttGD,
    TLayerGradComputeCtx *pGradCtx, TComputeCtxSet<TAttentionComputeCtx, ATT_BUFS> *pAttCtxSet,
    const TVector<TIntrusivePtr<TCudaAttentionParams>> &layerAtt,
    TCudaVector<int> &iterCounter,
    TFragmentStates &prevState,
    TCuda2DArray<half> *pStateGrad, TCudaVector<float> *pGradScaleArr, TCudaVector<float> *pStateGradMaxNorm
)
{
    TLayerGradComputeCtx &ctx = *pGradCtx;

    TCudaPOD<float> gradMaxNorm = pStateGradMaxNorm->GetElement(stepId);
    TCudaPOD<float> prevGradScale = pGradScaleArr->GetElement(stepId);
    TCudaPOD<float> gradScale = pGradScaleArr->GetElement(stepId + 1);
    CudaCall(c, ScaleGrad<STATE_DIM>).Grid(pParams->LenSampleBlocks)
        (pParams->Len, gradMaxNorm, prevGradScale).Write(pStateGrad, &gradScale);

    TCuda2DArray<TVecFloat> &normState8 = prevState.NormState;

    c->ClearMem(ctx.DNormState, pParams->Len);
    for (yint at = 0; at < YSize(layerAtt); ++at) {
        TCudaAttentionParams &att = *layerAtt[at];
        TAttentionComputeCtx &attCtx = pAttCtxSet->GetCtx();
        TAttentionGroupData &attGD = att.WideLayer ? *pWideAttGD : *pAttGD;
        constexpr int TT_GROUPS = TT_DIM / TILE_GROUP_SIZE;
        constexpr int Q_GROUPS = Q_DIM / TILE_GROUP_SIZE;
        constexpr int stateTiles = STATE_DIM / MM_TILE;
        constexpr int qTiles = Q_DIM / MM_TILE;
        constexpr int ttTiles = TT_DIM / MM_TILE;
        constexpr int kvTiles = GetCombinerWidth(TT_DIM) / MM_TILE;

        TCudaPOD<float> scaleCombiner = att.Combiner->GetScale();
        TCudaPOD<float> scaleQK = att.QK->GetScale();
        TCudaPOD<float> scaleQV = att.QV->GetScale();
        TCudaPOD<float> scaleK = att.K->GetScale();
        TCudaPOD<float> scaleV = att.V->GetScale();

        // mul forward
        MulForward<STATE_DIM, Q_DIM>(c, pParams, normState8, att.QK, &attCtx.DQKState, &attCtx.QKState8, &attCtx.QKStateScale);
        MulForward<STATE_DIM, Q_DIM>(c, pParams, normState8, att.QV, &attCtx.DQVState, &attCtx.QVState8, &attCtx.QVStateScale);
        MulForward<STATE_DIM, TT_DIM>(c, pParams, normState8, att.K, &attCtx.DKState, &attCtx.KState8, &attCtx.KStateScale);
        MulForward<STATE_DIM, TT_DIM>(c, pParams, normState8, att.V, &attCtx.DVState, &attCtx.VState8, &attCtx.VStateScale);

        // compute attention
        CudaCall(c, ComputeAttentionValLookup<Q_DIM, TT_DIM>).Block(WARP_SIZE, ATT_LOOKUP_BATCH + TT_GROUPS).Grid(pParams->LenAttTiles)
            (attCtx.QKState8, attCtx.QKStateScale, attCtx.QVState8, attCtx.QVStateScale, attCtx.VState8, attGD.AttSpans, attGD.AttSpanPtr, att.AlibiSlope, att.AlibiHyper)
            .Write(&attCtx.SumWeightLog, &attCtx.ValLookup8);

        CudaCall(c, KVProduct<TT_DIM>).Grid(pParams->LenSampleBlocks)
            (pParams->Len, attCtx.KState8, attCtx.ValLookup8)
            .Write(&attCtx.KVState8);

        // combiner derivatives
        // mul backward
        MatMulXYoYZeXZ(c, *pStateGrad, att.Combiner->GetFast(), &attCtx.DKVState, pParams->LenMMTiles, stateTiles, kvTiles, TStoreScaled(scaleCombiner));
        // former sum rank one
        MatMulXYoXZeYZ(c, *pStateGrad, attCtx.KVState8, &attCtx.DeltaCombiner, pParams->LenMMTiles, stateTiles, kvTiles, TStoreScaled(gradScale)); // need correct scale? VEC_SCALE?

        CudaCall(c, KVProductBackprop<TT_DIM>).Grid(pParams->LenSampleBlocks)
            (pParams->Len, attCtx.KState8, attCtx.ValLookup8, attCtx.DKVState)
            .Write(&attCtx.DKState, &attCtx.DValLookup, &attCtx.DScale);

        // attention derivative
        CudaCall(c, ComputeAttentionGradQK<Q_DIM, TT_DIM>).Block(WARP_SIZE, ATT_GRAD_BATCH + Q_GROUPS).Grid(pParams->LenAttTiles)
            (attCtx.QKState8, attCtx.QKStateScale, attCtx.QVState8, attCtx.QVStateScale, attCtx.VState8, attGD.AttSpans, attGD.AttSpanPtr, att.AlibiSlope, att.AlibiHyper)
            (attCtx.DValLookup, attCtx.DScale, attCtx.SumWeightLog)
            .Write(&attCtx.DQKState);

        CudaCall(c, ComputeAttentionGradQV<Q_DIM, TT_DIM>).Block(WARP_SIZE, ATT_GRAD_BATCH + Q_GROUPS + TT_GROUPS).Grid(pParams->LenAttTiles)
            (attCtx.QKState8, attCtx.QKStateScale, attCtx.QVState8, attCtx.QVStateScale, attCtx.VState8, attGD.RevAttSpans, attGD.RevAttSpanPtr, att.AlibiSlope, att.AlibiHyper)
            (attCtx.DValLookup, attCtx.DScale, attCtx.SumWeightLog)
            .Write(&attCtx.DQVState, &attCtx.DVState);

        CudaCall(c, BackpropNormalizeVecs8<Q_DIM, TVecFloat>).Grid(pParams->LenSampleBlocks)(pParams->Len, attCtx.QKState8, attCtx.QKStateScale, attCtx.DQKState).Write(&attCtx.DQKState);
        CudaCall(c, BackpropNormalizeVecs8<TT_DIM, TVecFloat>).Grid(pParams->LenSampleBlocks)(pParams->Len, attCtx.KState8, attCtx.KStateScale, attCtx.DKState).Write(&attCtx.DKState);
        CudaCall(c, BackpropNormalizeVecs8<TT_DIM, TValueVecFloat>).Grid(pParams->LenSampleBlocks)(pParams->Len, attCtx.VState8, attCtx.VStateScale, attCtx.DVState).Write(&attCtx.DVState);

        // mul backward
        MatMulXYoYZeXZ(c, attCtx.DQKState, att.QK->GetFast(), &ctx.DNormState, pParams->LenMMTiles, qTiles, stateTiles, TStoreAddScaled(scaleQK));
        MatMulXYoYZeXZ(c, attCtx.DQVState, att.QV->GetFast(), &ctx.DNormState, pParams->LenMMTiles, qTiles, stateTiles, TStoreAddScaled(scaleQV));
        MatMulXYoYZeXZ(c, attCtx.DKState, att.K->GetFast(), &ctx.DNormState, pParams->LenMMTiles, ttTiles, stateTiles, TStoreAddScaled(scaleK));
        MatMulXYoYZeXZ(c, attCtx.DVState, att.V->GetFast(), &ctx.DNormState, pParams->LenMMTiles, ttTiles, stateTiles, TStoreAddScaled(scaleV));

        // former sum rank one
        MatMulXYoXZeYZ(c, attCtx.DQKState, normState8, &attCtx.DeltaQK, pParams->LenMMTiles, qTiles, stateTiles, TStoreScaled(gradScale)); // need correct scale? VEC_SCALE?
        MatMulXYoXZeYZ(c, attCtx.DQVState, normState8, &attCtx.DeltaQV, pParams->LenMMTiles, qTiles, stateTiles, TStoreScaled(gradScale)); // need correct scale? VEC_SCALE?
        MatMulXYoXZeYZ(c, attCtx.DKState, normState8, &attCtx.DeltaK, pParams->LenMMTiles, ttTiles, stateTiles, TStoreScaled(gradScale)); // need correct scale? VEC_SCALE?
        MatMulXYoXZeYZ(c, attCtx.DVState, normState8, &attCtx.DeltaV, pParams->LenMMTiles, ttTiles, stateTiles, TStoreScaled(gradScale)); // need correct scale? VEC_SCALE?

        // accumulate delta on host
        att.Combiner->CopyDeltaToHostAndApply(c, attCtx.DeltaCombiner, iterCounter);
        att.QK->CopyDeltaToHostAndApply(c, attCtx.DeltaQK, iterCounter);
        att.QV->CopyDeltaToHostAndApply(c, attCtx.DeltaQV, iterCounter);
        att.K->CopyDeltaToHostAndApply(c, attCtx.DeltaK, iterCounter);
        att.V->CopyDeltaToHostAndApply(c, attCtx.DeltaV, iterCounter);
    }

    TCudaPOD<float> nextGradMaxNorm = pStateGradMaxNorm->GetElement(stepId + 1);
    CudaCall(c, ComputeLayerStateGrad<STATE_DIM>).Grid(pParams->LenSampleBlocks)
        (pParams->Len, pParams->DropTable, normState8, prevState.StateScale, ctx.DNormState)
        .Write(pStateGrad).AtomicWrite(&nextGradMaxNorm);
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// embedding & final layer
#include "gpt_embedding.cuh"
#include "gpt_final.cuh"


template <int STATE_DIM>
__global__ void NormalizeFinalVecs(int offset, int len, TCuda2DPtr<TStateFloat> src, TCuda2DPtr<half> dst)
{
    int t = blockIdx.x * SAMPLE_BLOCK + threadIdx.y;

    // normalize
    if (t < len) {
        float v[WCOUNT];
        float vNorm[WCOUNT];
        LoadVec<STATE_DIM>(v, src[t + offset]);
        NormalizeVec<STATE_DIM>(vNorm, v);
        StoreVec<STATE_DIM>(dst[t], vNorm);
    } else {
        StoreZeroVec<STATE_DIM>(dst[t]);
    }
}
KERNEL_BLOCK_SIZE(NormalizeFinalVecs, WARP_SIZE, SAMPLE_BLOCK);


template <int STATE_DIM>
__global__ void BackpropNormalizeFinalVecs(int len, TCuda2DPtr<TStateFloat> src, TCuda2DPtr<half> grad, TCuda2DPtr<half> dst)
{
    int t = blockIdx.x * SAMPLE_BLOCK + threadIdx.y;

    // normalize
    if (t < len) {
        float v[WCOUNT];
        LoadVec<STATE_DIM>(v, src[t]);
        float vGrad[WCOUNT];
        LoadVec<STATE_DIM>(vGrad, grad[t]);
        float stateGrad[WCOUNT];
        StateNormalizeBackprop<STATE_DIM>(v, vGrad, stateGrad);
        StoreVec<STATE_DIM>(dst[t], stateGrad);
    } else {
        StoreZeroVec<STATE_DIM>(dst[t]);
    }
}
KERNEL_BLOCK_SIZE(BackpropNormalizeFinalVecs, WARP_SIZE, SAMPLE_BLOCK);



template <int STATE_DIM>
__global__ void ComputeInitialGradNorm(int len, TCuda2DPtr<half> grad, float *gradMaxNorm, float *gradScale)
{
    int t = blockIdx.x * SAMPLE_BLOCK + threadIdx.y;
    if (t == 0 && threadIdx.x == 0) {
        *gradScale = 1;
    }
    if (t < len) {
        float v[WCOUNT];
        LoadVec<STATE_DIM>(v, grad[t]);
        float gradMax = CalcLinf<STATE_DIM>(v);
        if (threadIdx.x == 0) {
            atomicMax((int*)gradMaxNorm, __float_as_int(gradMax));
        }
    }
}
KERNEL_BLOCK_SIZE(ComputeInitialGradNorm, WARP_SIZE, SAMPLE_BLOCK);


__global__ void ComputeLossKernel(int offset, int len, TCuda2DPtr<float> predArr, float *predArrScale, int *targetArr, float *pRes)
{
    int h = threadIdx.x;
    float sum = 0;
    for (int base = 0; base < len; base += WARP_SIZE) {
        int t = base + h;
        if (t < len) {
            int target = targetArr[offset + t];
            if (target >= 0) {
                sum += logf(predArr[t][target] * predArrScale[t]);
            }
        }
    }
    sum = WarpSum(sum);
    if (h == 0) {
        *pRes += sum;
    }
}


class TSinlgeComputeContext : public TThrRefBase
{
    TStream Stream;
    TModelDim ModelDim;
    TIntrusivePtr<IModel> Model;
    TIntrusivePtr<TCudaModelMatrixScale> CudaMatrixScale;
    TIntrusivePtr<TCudaModelMatrix> LabelEmbed;
    TVector<TVector<TIntrusivePtr<TCudaAttentionParams>>> LayerArr;
    TIntrusivePtr<TCudaModelMatrix> FinalLayer;
    // model params
    TCudaVector<float> Bias;
    // gradients
    TCuda2DArray<float> DeltaFinalLayer;
    // compute params
    TCudaVector<TLabelIndex> LabelArr;
    TCudaVector<ui32> LabelPtr;
    TVector<TLabelIndex> KeepLabelArr;
    TVector<ui32> KeepLabelPtr;
    yint TargetNodeCount = 0;
    TCudaVector<TLabelIndex> InvLabelArr;
    TCudaVector<ui32> InvLabelPos;
    TCudaVector<ui32> InvLabelPosPtr;
    TVector<TIntrusivePtr<TFragmentStates>> AllStates;
    TCuda2DArray<TStateFloat> State; // after forward pass contains state after all layers applied
    TCuda2DArray<half> FinalStateNormalized;
    TCuda2DArray<half> StateGrad;
    TCudaVector<float> StateGradMaxNorm;
    TCudaVector<float> StateGradScale;
    TCuda2DArray<float> PredictionArr;
    TCudaVector<float> PredictionArrScale;
    TCuda2DArray<half> LogitBuf;
    TCudaVector<int> TargetArr;
    TCudaVector<int> IterCounter;
    TCudaVector<float> SumScore;

    TComputeParams ComputeParams;
    TLabelInverseIndex LabelInverseIndex;
    TAttentionGroupData AttGD;
    TAttentionGroupData WideAttGD;
    TComputeCtxSet<TLayerGradComputeCtx, 2> LayerCtxSet;
    TComputeCtxSet<TAttentionComputeCtx, 4> AttCtxSet;

    TIntrusivePtr<TGraph> ForwardComputer;
    TIntrusivePtr<TGraph> CopyModelForwardComputer;
    TIntrusivePtr<TGraph> ApplyFinalLayerComputer;
    TIntrusivePtr<TGraph> BackpropComputer;
    yint BackpropComputerLen = 0;
    TIntrusivePtr<TGraph> SumScoreComputer;
    yint SumScoreComputerLen = 0;

    bool ComputeInFly = false;
    bool NeedCopyToDevice = true;

private:
    // size dispatched functions
    struct ICreateGraph : public TThrRefBase
    {
        virtual TIntrusivePtr<TGraph> CreateForwardGraph(TSinlgeComputeContext *pThis, bool arg) = 0;
        virtual void CreateBackpropGraph(TSinlgeComputeContext *pThis, yint len) = 0;
        virtual TIntrusivePtr<TGraph> CreateApplyFinalLayerGraph(TSinlgeComputeContext *pThis) = 0;
        virtual void AddApplyFinalLayerGraph(TSinlgeComputeContext *pThis, TIntrusivePtr<TGraph> c, const TWindowParams &window) = 0;
        virtual void CreateSumScoreComputer(TSinlgeComputeContext *pThis, yint len) = 0;
    };
    TIntrusivePtr<ICreateGraph> DimDispatch;

    template <int STATE_DIM, int Q_DIM, int TT_DIM>
    struct TCreateGraph : public ICreateGraph
    {
        TIntrusivePtr<TGraph> CreateForwardGraph(TSinlgeComputeContext *pThis, bool arg) override
        {
            return pThis->CreateForwardGraph<STATE_DIM, Q_DIM, TT_DIM>(arg);
        }
        void CreateBackpropGraph(TSinlgeComputeContext *pThis, yint len) override
        {
            pThis->CreateBackpropGraph<STATE_DIM, Q_DIM, TT_DIM>(len);
        }
        TIntrusivePtr<TGraph> CreateApplyFinalLayerGraph(TSinlgeComputeContext *pThis) override
        {
            TIntrusivePtr<TGraph> c = new TGraph;
            TComputeParams *pParams = &pThis->ComputeParams;
            pThis->CreateApplyFinalLayerGraph<STATE_DIM>(c, pParams->FinalOffset, pParams->FinalLen, pParams->FinalLenSampleBlocks, pParams->FinalLenMMTiles);
            return c;
        }
        void AddApplyFinalLayerGraph(TSinlgeComputeContext *pThis, TIntrusivePtr<TGraph> c, const TWindowParams &window) override
        {
            pThis->CreateApplyFinalLayerGraph<STATE_DIM>(c, window);
        }
        void CreateSumScoreComputer(TSinlgeComputeContext *pThis, yint len) override
        {
            pThis->CreateSumScoreComputerImpl<STATE_DIM>(len);
        }
    };

private:
    yint GetVocabSizeRounded() const
    {
        return DivCeil(ModelDim.VocabSize, MM_TILE) * MM_TILE;
    }

    yint GetFinalLayerSizeRounded() const
    {
        return DivCeil(ModelDim.VocabSize, MM_TILE) * MM_TILE;
    }

    void CopyStaticModelParams()
    {
        TIntrusivePtr<TGraph> c = new TGraph;
        LabelEmbed->CopyToDevice(c.Get());
        FinalLayer->CopyToDevice(c.Get());
        if (ModelDim.HasFlag(MPF_TUNE_EMBED)) {
            // token embed delta only incrementally cleared in Backprop()
            c->ClearMem(LabelEmbed->GetDeltaHost());
        }
        c->Run(Stream);
        Bias.Put(Stream, Model->GetBias());
        Stream.Sync();
    }

public:
    void WaitCudaCompute()
    {
        if (ComputeInFly) {
            Stream.Sync();
            ComputeInFly = false;
        }
    }

public:
    TSinlgeComputeContext(yint deviceId, TIntrusivePtr<IModel> pModel, yint nodeCount) : Model(pModel)
    {
        ModelDim = Model->GetModelDim();
        // dimension restrictions
        Y_VERIFY((ModelDim.Dim % WARP_SIZE) == 0); // some kernels process states with warps
        Y_VERIFY((ModelDim.Dim % I8_TILE_GROUP_SIZE) == 0);
        Y_VERIFY((ModelDim.TTDim % MM_TILE) == 0);
        // tt restrictions
        Y_VERIFY((ModelDim.TTDim % TILE_GROUP_SIZE) == 0);
        // params
        yint maxLen = DivCeil(nodeCount, SAMPLE_ARR_CHUNK) * SAMPLE_ARR_CHUNK;
        yint vocabRoundSize = GetVocabSizeRounded();
        yint finalLayerRoundSize = GetFinalLayerSizeRounded();
        yint maxLabels = maxLen * 64; // upper cap
        yint finalMaxLen = Min<yint>(maxLen, PREDICTION_ARR_SZ);
        yint maxStepId = YSize(ModelDim.Layers) + 100; // upper cap
        //
        CudaMatrixScale = new TCudaModelMatrixScale(Model->GetMatrixScale(), Stream);
        yint depth = YSize(ModelDim.Layers);
        LayerArr.resize(depth);
        for (yint d = 0; d < depth; ++d) {
            for (yint k = 0; k < YSize(ModelDim.Layers[d]); ++k) {
                LayerArr[d].push_back(new TCudaAttentionParams(deviceId, CudaMatrixScale, Model->GetAttention(d, k)));
            }
        }
        LabelEmbed = new TCudaModelMatrix(deviceId, CudaMatrixScale, Model->GetLabelEmbed(), MM_MEM_HOST);
        FinalLayer = new TCudaModelMatrix(deviceId, CudaMatrixScale, Model->GetFinalLayer(), MM_MEM_DEVICE);
        //
        Bias.Allocate(ModelDim.VocabSize);
        // gradients
        if (ModelDim.HasFlag(MPF_TUNE_FINAL_LAYER)) {
            DeltaFinalLayer.AllocateCuda(ModelDim.Dim, finalLayerRoundSize);
        }
        //
        LabelArr.AllocateWC(maxLabels); // upper cap
        LabelPtr.AllocateWC(maxLen + 1);
        InvLabelArr.AllocateWC(ModelDim.LabelCount); // upper cap
        InvLabelPos.AllocateWC(maxLabels); // upper cap
        InvLabelPosPtr.AllocateWC(ModelDim.LabelCount + 1);
        AllStates.resize(YSize(ModelDim.Layers));
        for (yint k = 0; k < YSize(AllStates); ++k) {
            AllStates[k] = new TFragmentStates;
            AllStates[k]->AllocateCuda(ModelDim.Dim, maxLen);
        }
        State.Allocate(ModelDim.Dim, maxLen);
        FinalStateNormalized.AllocateCuda(ModelDim.Dim, finalMaxLen);
        StateGrad.AllocateCuda(ModelDim.Dim, maxLen);
        StateGradMaxNorm.AllocateCuda(maxStepId);
        StateGradScale.AllocateCuda(maxStepId);
        PredictionArr.Allocate(vocabRoundSize, finalMaxLen);
        PredictionArrScale.Allocate(finalMaxLen);
        LogitBuf.AllocateCuda(finalLayerRoundSize, finalMaxLen);
        TargetArr.Allocate(nodeCount);
        IterCounter.AllocateCuda(1);
        IterCounter.ClearDeviceMem(Stream);
        SumScore.Allocate(1);
        //
        Model->ResetIterCount();
        // compute params & contexts
        ComputeParams.Allocate(ModelDim, maxLen);
        AttGD.Allocate(ModelDim, maxLen, 4); // upper cap
        WideAttGD.Allocate(ModelDim, maxLen, 4); // upper cap
        LayerCtxSet.AllocateCuda(ModelDim.Dim, ModelDim.QDim, ModelDim.TTDim, maxLen);
        AttCtxSet.AllocateCuda(ModelDim.Dim, ModelDim.QDim, ModelDim.TTDim, maxLen);
        // create compute graphs
        if (ModelDim.Dim == 256 && ModelDim.QDim == 128 && ModelDim.TTDim == 64) {
            DimDispatch = new TCreateGraph<256, 128, 64>();
        } else if (ModelDim.Dim == 256 && ModelDim.QDim == 128 && ModelDim.TTDim == 128) {
            DimDispatch = new TCreateGraph<256, 128, 128>();
        } else if (ModelDim.Dim == 512 && ModelDim.QDim == 128 && ModelDim.TTDim == 64) {
            DimDispatch = new TCreateGraph<512, 128, 64>();
        } else if (ModelDim.Dim == 512 && ModelDim.QDim == 128 && ModelDim.TTDim == 128) {
            DimDispatch = new TCreateGraph<512, 128, 128>();
        } else if (ModelDim.Dim == 512 && ModelDim.QDim == 128 && ModelDim.TTDim == 256) {
            DimDispatch = new TCreateGraph<512, 128, 256>();
        } else if (ModelDim.Dim == 768 && ModelDim.QDim == 128 && ModelDim.TTDim == 128) {
            DimDispatch = new TCreateGraph<768, 128, 128>();
        } else if (ModelDim.Dim == 1024 && ModelDim.QDim == 128 && ModelDim.TTDim == 128) {
            DimDispatch = new TCreateGraph<1024, 128, 128>();
        } else if (ModelDim.Dim == 1024 && ModelDim.QDim == 128 && ModelDim.TTDim == 256) {
            DimDispatch = new TCreateGraph<1024, 128, 256>();
        } else if (ModelDim.Dim == 2048 && ModelDim.QDim == 128 && ModelDim.TTDim == 256) {
            DimDispatch = new TCreateGraph<2048, 128, 256>();
        } else {
            Y_VERIFY(0 && "no kernels for this model dimensions");
        }
        ForwardComputer = DimDispatch->CreateForwardGraph(this, false);
        CopyModelForwardComputer = DimDispatch->CreateForwardGraph(this, true);
        ApplyFinalLayerComputer = DimDispatch->CreateApplyFinalLayerGraph(this);
        // assign model params
        CopyStaticModelParams();
    }

    void CreateSumScoreComputer(yint len)
    {
        if (len == SumScoreComputerLen) {
            return;
        }
        DimDispatch->CreateSumScoreComputer(this, len);
    }

    template <int STATE_DIM>
    void CreateSumScoreComputerImpl(yint len)
    {
        TIntrusivePtr<TGraph> c = new TGraph;
        c->ClearMem(SumScore);
        for (yint winOffset = 0; winOffset < len; winOffset += PREDICTION_ARR_SZ) {
            TWindowParams window(winOffset, len, PREDICTION_ARR_SZ);
            // apply final layer on fragment
            CreateApplyFinalLayerGraph<STATE_DIM>(c, window);
            // compute score
            CudaCall(c, ComputeLossKernel)(window.Offset, window.Len, PredictionArr, PredictionArrScale, TargetArr)
                .Write(&SumScore);
        }
        SumScoreComputer = c;
        SumScoreComputerLen = len;
    }


    void CreateBackpropComputer(yint len)
    {
        if (len == BackpropComputerLen) {
            return;
        }
        DimDispatch->CreateBackpropGraph(this, len);
    }

    template <int STATE_DIM, int Q_DIM, int TT_DIM>
    TIntrusivePtr<TGraph> CreateForwardGraph(bool copyModelToDevice)
    {
        Y_VERIFY(ModelDim.Dim == STATE_DIM);
        Y_VERIFY(ModelDim.QDim == Q_DIM);
        Y_VERIFY(ModelDim.TTDim == TT_DIM);
        TComputeParams *pParams = &ComputeParams;

        TIntrusivePtr<TGraph> c = new TGraph;
        if (copyModelToDevice) {
            CudaMatrixScale->CopyToDevice(c);
            if (ModelDim.HasFlag(MPF_TUNE_EMBED)) {
                // have to copy only if we update label embed, otherwise copy once along with Bias
                LabelEmbed->CopyToDevice(c);
            }
        }

        TCudaPOD<float> scaleEmbed = LabelEmbed->GetScale();
        CudaCall(c, AddEmbeddings<STATE_DIM>).Grid(pParams->Len)
            (LabelArr, LabelPtr, LabelEmbed->GetFast(), scaleEmbed)
            .Write(&State);

        // apply layers
        Y_ASSERT(YSize(LayerArr) == YSize(ModelDim.Layers));
        for (yint d = 0; d < YSize(LayerArr); ++d) {
            AddLookupProduct<STATE_DIM, Q_DIM, TT_DIM>(c, copyModelToDevice, pParams, &AttGD, &WideAttGD, &AttCtxSet,
                LayerArr[d],
                &State, AllStates[d].Get());
        }

        if (copyModelToDevice && ModelDim.HasFlag(MPF_TUNE_FINAL_LAYER)) {
            FinalLayer->CopyToDevice(c); // have to copy once if we don't update final layer
        }
        return c;
    }


    template <int STATE_DIM, class T>
    void CreateApplyFinalLayerGraph(TIntrusivePtr<TGraph> c, T &&winOffset, T &&winLen, T &&winLenSampleBlocks, T &&winLenMMTiles)
    {
        Y_VERIFY(ModelDim.Dim == STATE_DIM);
        int finalTiles = GetFinalLayerSizeRounded() / MM_TILE;
        int stateTiles = STATE_DIM / MM_TILE;
        //TComputeParams *pParams = &ComputeParams;

        // somehow using i8 state vector quantization here breaks everything, use full precision state vectors
        CudaCall(c, NormalizeFinalVecs<STATE_DIM>).Grid(winLenSampleBlocks)
            (winOffset, winLen, State).Write(&FinalStateNormalized);

        MatMulXYoZYeXZ(c, FinalStateNormalized, FinalLayer->GetFast(), &LogitBuf, winLenMMTiles, stateTiles, finalTiles, TStore());

        TCudaPOD<float> scaleFinalLayer = FinalLayer->GetScale();
        CudaCall(c, Softmax).Grid(winLenSampleBlocks)
            (winLen, LogitBuf, scaleFinalLayer, CalcDotScale(STATE_DIM) * FINAL_LAYER_SOFTMAX_SCALE)
            (ModelDim.VocabSize, Bias)
            .Write(&PredictionArr, &PredictionArrScale);
    }

    template <int STATE_DIM>
    void CreateApplyFinalLayerGraph(TIntrusivePtr<TGraph> c, const TWindowParams &window)
    {
        CreateApplyFinalLayerGraph<STATE_DIM>(c, window.Offset, window.Len, window.LenSampleBlocks, window.LenMMTiles);
    }


    template <int STATE_DIM>
    void CreateBackpropFinalLayerGraph(TIntrusivePtr<TGraph> c, const TWindowParams &window)
    {
        Y_VERIFY(ModelDim.Dim == STATE_DIM);
        int finalTiles = GetFinalLayerSizeRounded() / MM_TILE;
        int stateTiles = STATE_DIM / MM_TILE;
        int vocabRoundSize = GetVocabSizeRounded();

        // compute gradient
        CudaCall(c, ComputeGradient).Grid(window.Len)(window.Offset, ModelDim.VocabSize, vocabRoundSize, PredictionArr, PredictionArrScale, TargetArr).Write(&LogitBuf);

        // mul backward
        TCudaPOD<float> scaleFinalLayer = FinalLayer->GetScale();
        auto stateGradFrag = StateGrad.MakeFragment(0, StateGrad.GetXSize(), window.Offset, window.Len);
        MatMulXYoYZeXZ(c, LogitBuf, FinalLayer->GetFast(), &stateGradFrag, window.LenMMTiles, finalTiles, stateTiles, TStoreScaled(scaleFinalLayer));

        if (ModelDim.HasFlag(MPF_TUNE_FINAL_LAYER)) {
            MatMulXYoXZeYZ(c, LogitBuf, FinalStateNormalized, &DeltaFinalLayer, window.LenMMTiles, finalTiles, stateTiles, TStoreAdd());
        }
    }


    template <int STATE_DIM, int Q_DIM, int TT_DIM>
    void CreateBackpropGraph(yint len)
    {
        Y_VERIFY(ModelDim.Dim == STATE_DIM);
        Y_VERIFY(ModelDim.QDim == Q_DIM);
        Y_VERIFY(ModelDim.TTDim == TT_DIM);
        TComputeParams *pParams = &ComputeParams;

        TIntrusivePtr<TGraph> c = new TGraph;

        // new flag value for deltas
        IncrementIterCounter(c, IterCounter);

        // backprop final layer
        if (ModelDim.HasFlag(MPF_TUNE_FINAL_LAYER)) {
            c->ClearMem(DeltaFinalLayer);
        }
        for (yint winOffset = 0; winOffset < len; winOffset += PREDICTION_ARR_SZ) {
            TWindowParams window(winOffset, len, PREDICTION_ARR_SZ);
            // apply final layer on fragment
            CreateApplyFinalLayerGraph<STATE_DIM>(c, window);
            // compute final layer gradient on fragment
            CreateBackpropFinalLayerGraph<STATE_DIM>(c, window);
        }
        if (ModelDim.HasFlag(MPF_TUNE_FINAL_LAYER)) {
            FinalLayer->CopyDeltaToHostAndApply(c, DeltaFinalLayer, IterCounter);
        }

        {
            // use State contents from forward pass
            CudaCall(c, BackpropNormalizeFinalVecs<STATE_DIM>).Grid(pParams->LenSampleBlocks)(pParams->Len, State, StateGrad).Write(&StateGrad);

            // can be merged with backprop normalize
            c->ClearMem(StateGradMaxNorm);
            TCudaPOD<float> gradMaxNorm = StateGradMaxNorm.GetElement(0);
            TCudaPOD<float> gradScale = StateGradScale.GetElement(0);
            CudaCall(c, ComputeInitialGradNorm<STATE_DIM>).Grid(pParams->LenSampleBlocks)(pParams->Len, StateGrad).AtomicWrite(&gradMaxNorm).Write(&gradScale);
        }

        // modify layers
        int stepId = 0;
        for (yint d = YSize(LayerArr) - 1; d >= 0; --d) {
            TLayerGradComputeCtx &layerGradCtx = LayerCtxSet.GetCtx();
            AddLookupProductBackprop<STATE_DIM, Q_DIM, TT_DIM>(c, stepId++, pParams, &AttGD, &WideAttGD, &layerGradCtx, &AttCtxSet,
                LayerArr[d],
                IterCounter,
                *AllStates[d],
                &StateGrad, &StateGradScale, &StateGradMaxNorm);
        }

        if (ModelDim.HasFlag(MPF_TUNE_EMBED)) {
            TCuda2DArray<float> &deltaLabel = LabelEmbed->GetDeltaHost();
            TCudaPOD<float> gradScale = StateGradScale.GetElement(stepId);
            CudaCall(c, BackpropEmbeddings<STATE_DIM>).Grid(pParams->InvLabelCount)
                (InvLabelArr, InvLabelPos, InvLabelPosPtr)
                (StateGrad, gradScale)
                .Write(&deltaLabel);
            LabelEmbed->ApplyHostDelta(c, IterCounter);
        }

        BackpropComputer = c;
        BackpropComputerLen = len;
    }


    TModelDim GetLocalModelDim()
    {
        return ModelDim;
    }

    void OnParamsUpdate()
    {
        for (yint d = 0; d < YSize(ModelDim.Layers); ++d) {
            for (yint k = 0; k < YSize(ModelDim.Layers[d]); ++k) {
                const TAttentionParams &att = Model->GetAttention(d, k);
                TCudaAttentionParams &dst = *LayerArr[d][k];
                dst.UpteLayerParams(att);
            }
        }
        CopyStaticModelParams();
        NeedCopyToDevice = true;
    }

    void SetTarget(yint len, const TVector<TNodeTarget> &target)
    {
        TVector<int> targetArr;
        targetArr.resize(len, -1);
        yint count = 0;
        for (const TNodeTarget &nt : target) {
            Y_ASSERT(nt.TargetId >= 0 && nt.TargetId < ModelDim.VocabSize);
            Y_ASSERT(targetArr[nt.Node] == -1); // current ComputeGradient() supports single target per node
            targetArr[nt.Node] = nt.TargetId;
            ++count;
        }
        TargetNodeCount = count;
        TargetArr.Put(Stream, targetArr);
    }

    void Init(const TNodesBatch &nodes, const TVector<ui32> &dropTable)
    {
        Y_ASSERT(ModelDim == Model->GetModelDim());
        yint len = nodes.GetNodeCount();
        Y_VERIFY(YSize(nodes.LabelPtr) <= LabelPtr.GetSize());
        Y_VERIFY(nodes.LabelPtr.back() <= LabelArr.GetSize());
        for (yint pos : nodes.LabelArr) {
            Y_ASSERT(pos < ModelDim.LabelCount);
        }
        TAttentionInfoGrouped<ATT_GROUP> attGroups;
        GroupAttention(nodes.Att, &attGroups);
        TAttentionInfoGrouped<ATT_GROUP> revAttGroups;
        GroupAttention(TransposeAttention(nodes.Att), &revAttGroups);
        TAttentionInfoGrouped<ATT_GROUP> wideAttGroups;
        GroupAttention(nodes.WideAtt, &wideAttGroups);
        TAttentionInfoGrouped<ATT_GROUP> wideRevAttGroups;
        GroupAttention(TransposeAttention(nodes.WideAtt), &wideRevAttGroups);
        KeepLabelArr = nodes.LabelArr;
        KeepLabelPtr = nodes.LabelPtr;

        WaitCudaCompute();
        LabelArr.Put(Stream, nodes.LabelArr);
        LabelPtr.Put(Stream, nodes.LabelPtr);
        SetTarget(len, nodes.Target);

        ComputeParams.Init(Stream, len, nodes.SampleIndex, dropTable);
        AttGD.Init(Stream, ComputeParams.GetLenBufferSize(), &attGroups, &revAttGroups);
        WideAttGD.Init(Stream, ComputeParams.GetLenBufferSize(), &wideAttGroups, &wideRevAttGroups);
    }

    void RunForward()
    {
        if (NeedCopyToDevice) {
            CopyModelForwardComputer->Run(Stream);
        } else {
            ForwardComputer->Run(Stream);
        }
        NeedCopyToDevice = false;
        ComputeInFly = true;
    }

    void ComputeFinalStateVectors(TVector<TVector<float>> *pStateVectors)
    {
        State.CopyToHost(Stream);
        Stream.Sync();
        GetAllData(State, pStateVectors);
        pStateVectors->resize(ComputeParams.Len.Get());
    }

    void ComputeFragmentPredictions(TVector<TVector<float>> *pPrediction)
    {
        yint len = ComputeParams.Len.Get();
        pPrediction->resize(len);
        for (yint winOffset = 0; winOffset < len; winOffset += PREDICTION_ARR_SZ) {
            TWindowParams window(winOffset, len, PREDICTION_ARR_SZ);
            ComputeParams.SetFinalWindow(window);
            ApplyFinalLayerComputer->Run(Stream);
            PredictionArr.CopyToHost(Stream, window.Len);
            PredictionArrScale.CopyToHost(Stream, window.Len);
            Stream.Sync();
            TVector<TVector<float>> winPred;
            TVector<float> winPredScale;
            PredictionArr.GetAllData(&winPred);
            PredictionArrScale.GetAllData(&winPredScale);
            // scale result
            for (yint t = 0; t < window.Len; ++t) {
                TVector<float> &dst = (*pPrediction)[window.Offset + t];
                TVector<float> &pred = winPred[t];
                Y_VERIFY(YSize(pred) >= ModelDim.VocabSize);
                yint width = ModelDim.VocabSize;
                dst.yresize(width);
                float scale = winPredScale[t];
                for (yint c = 0; c < width; ++c) {
                    dst[c] = pred[c] * scale;
                }
            }
        }
    }

    float ComputeScore()
    {
        yint len = ComputeParams.Len.Get();
        CreateSumScoreComputer(len);
        SumScoreComputer->Run(Stream);
        SumScore.CopyToHost(Stream);
        Stream.Sync();
        TVector<float> sumScore;
        SumScore.GetAllData(&sumScore);
        return sumScore[0] / TargetNodeCount;
    }

    void BackpropBuildInverseIndex()
    {
        LabelInverseIndex.BuildInverseIndex(KeepLabelArr, KeepLabelPtr);
    }

    void BackpropInitParams()
    {
        yint len = ComputeParams.Len.Get();
        yint invLabelCount = YSize(LabelInverseIndex.InvLabelArr);
        ComputeParams.SetInvLabelCount(invLabelCount);
        CreateBackpropComputer(len);
    }

    void RunBackprop()
    {
        InvLabelArr.Put(Stream, LabelInverseIndex.InvLabelArr);
        InvLabelPos.Put(Stream, LabelInverseIndex.InvLabelPos);
        InvLabelPosPtr.Put(Stream, LabelInverseIndex.InvLabelPosPtr);
        LabelEmbed->SetNonzeroRows(LabelInverseIndex.InvLabelArr);

        BackpropComputer->Run(Stream);
        ComputeInFly = true;
        NeedCopyToDevice = true;
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
// multi GPU support (for backprop only so far)
class TMultiComputeContext : public IComputeContext
{
    enum EJob
    {
        JOB_WAIT,
        JOB_ON_PARAMS_UPDATE,
        JOB_INIT,
        JOB_BAKCPROP_BUILD_INV_INDEX,
        JOB_BACKPROP_INIT_PARAMS,
        JOB_BACKPROP_RUN,
    };

    struct TDeviceControlThread : public TThrRefBase
    {
        TThread Worker;
        TSingleConsumerJobQueue<EJob> JobQueue;
        std::atomic<int> JobQueueSize;
        TNodesBatch Nodes;
        TVector<ui32> DropTable;

        TDeviceControlThread() : JobQueueSize(0)
        {
        }
        void Run(TMultiComputeContext *pThis)
        {
            Worker.Create(pThis);
        }
        void AddOp(EJob op)
        {
            JobQueueSize.fetch_add(1);
            JobQueue.Enqueue(op);
        }
        void WaitDevice()
        {
            while (JobQueueSize.load() > 0) {
                _mm_pause();
            }
        }
    };


private:
    TIntrusivePtr<IModel> Model;
    TVector<TIntrusivePtr<TSinlgeComputeContext>> CtxArr;
    bool ModelDeltaInFly = false;
    std::atomic<int> WorkerId;
    TVector<TIntrusivePtr<TDeviceControlThread>> ThrArr;
    volatile bool Exit = false;


private:
    void SetDevice(yint deviceId) const
    {
        if (YSize(CtxArr) > 1) {
            Y_VERIFY(cudaSetDevice(deviceId) == cudaSuccess);
        }
    }
public:
    void WorkerThread()
    {
        yint deviceId = WorkerId.fetch_add(1);
        SetDevice(deviceId);
        TDeviceControlThread *thr = ThrArr[deviceId].Get();
        while (!Exit) {
            EJob job;
            if (thr->JobQueue.DequeueFirst(&job)) {
                TSinlgeComputeContext *ctx = CtxArr[deviceId].Get();
                switch (job) {
                case JOB_WAIT:
                    ctx->WaitCudaCompute();
                    break;
                case JOB_ON_PARAMS_UPDATE:
                    ctx->OnParamsUpdate();
                    break;
                case JOB_INIT:
                    ctx->Init(thr->Nodes, thr->DropTable);
                    break;
                case JOB_BAKCPROP_BUILD_INV_INDEX:
                    ctx->BackpropBuildInverseIndex();
                    break;
                case JOB_BACKPROP_RUN:
                    ctx->BackpropInitParams();
                    ctx->RunForward();
                    ctx->RunBackprop();
                    break;
                }
                thr->JobQueueSize.fetch_add(-1);
            } else {
                _mm_pause();
            }
        }
    }
private:
    void ForeachDevice(EJob func)
    {
        for (yint deviceId = 0; deviceId < YSize(ThrArr); ++deviceId) {
            ThrArr[deviceId]->AddOp(func);
        }
    }

    void WaitDevices()
    {
        for (yint deviceId = 0; deviceId < YSize(ThrArr); ++deviceId) {
            ThrArr[deviceId]->WaitDevice();
        }
    }

    void WaitAllCompute()
    {
        // correct order is to wait gpu graph completion first, then wait cpu ops (gpu graphs launch cpu compute)
        ForeachDevice(JOB_WAIT);
        WaitDevices();
        Model->WaitCompute();
    }

    void WaitModelDelta()
    {
        if (ModelDeltaInFly) {
            WaitAllCompute();
            ModelDeltaInFly = false;
        }
    }

private:
    yint GetDeviceCount() override
    {
        return YSize(CtxArr);
    }

    TModelDim GetModelDim() override
    {
        for (auto &ctx : CtxArr) {
            Y_ASSERT(ctx->GetLocalModelDim() == Model->GetModelDim());
        }
        return Model->GetModelDim();
    }

    void GetParams(TModelParams *p) override
    {
        WaitModelDelta();
        Model->GetParamsImpl(p);
    }

    void SetParams(const TModelParams &p) override
    {
        WaitModelDelta();
        Model->SetParamsImpl(p);
        ForeachDevice(JOB_ON_PARAMS_UPDATE);
        WaitDevices();
    }

    void GetGradient(TModelParams *p) override
    {
        WaitModelDelta();
        Model->GetGradientImpl(p);
    }

    TNodesBatch &GetNodes(yint deviceId) override
    {
        ThrArr[deviceId]->WaitDevice();
        return ThrArr[deviceId]->Nodes;
    }

    TVector<ui32> &GetDropTable(yint deviceId)
    {
        ThrArr[deviceId]->WaitDevice();
        return ThrArr[deviceId]->DropTable;
    }

    void Init(yint deviceId) override
    {
        ThrArr[deviceId]->AddOp(JOB_INIT);
    }

    void ComputeFinalStateVectors(TVector<TVector<float>> *pStateVectors) override
    {
        WaitDevices();
        WaitModelDelta();
        // setDevice() not needed, device 0 is default
        CtxArr[0]->RunForward();
        CtxArr[0]->ComputeFinalStateVectors(pStateVectors);
    }

    void ComputeFragmentPredictions(TVector<TVector<float>> *pPrediction) override
    {
        WaitDevices();
        WaitModelDelta();
        // setDevice() not needed, device 0 is default
        CtxArr[0]->RunForward();
        CtxArr[0]->ComputeFragmentPredictions(pPrediction);
    }

    float ComputeScore() override
    {
        WaitDevices();
        WaitModelDelta();
        // setDevice() not needed, device 0 is default
        CtxArr[0]->RunForward();
        return CtxArr[0]->ComputeScore();
    }

    void Backprop(float step) override
    {
        ForeachDevice(JOB_BAKCPROP_BUILD_INV_INDEX);
        WaitAllCompute(); // modify cuda graphs when queue is empty
        Model->StartIteration(step); // no pending matrix ops at this point expected, can be called a bit later
        ForeachDevice(JOB_BACKPROP_RUN);
        ModelDeltaInFly = true;
    }

    ~TMultiComputeContext()
    {
        Exit = true;
    }

public:
    TMultiComputeContext(TIntrusivePtr<IModel> pModel, yint nodeCount) : Model(pModel), WorkerId(0)
    {
        yint deviceCount = Model->GetDeviceCount();
        CtxArr.resize(deviceCount);
        for (yint deviceId = 0; deviceId < deviceCount; ++deviceId) {
            SetDevice(deviceId);
            CtxArr[deviceId] = new TSinlgeComputeContext(deviceId, pModel, nodeCount);
        }
        SetDevice(0);
        ThrArr.resize(deviceCount);
        for (yint deviceId = 0; deviceId < deviceCount; ++deviceId) {
            ThrArr[deviceId] = new TDeviceControlThread();
        }
        for (yint deviceId = 0; deviceId < deviceCount; ++deviceId) {
            ThrArr[deviceId]->Run(this);
        }
    }
};

TIntrusivePtr<IComputeContext> CreateContext(TIntrusivePtr<IModel> pModel, yint nodeCount)
{
    return new TMultiComputeContext(pModel, nodeCount);
}
}
