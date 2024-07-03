#include "stdafx.h"
#include "cpu_infer.h"
#include <gpt/att/att.h>
#include <gpt/data/data.h>
#include <gpt/model_params/model_dim.h>
#include <gpt/model_params/model_params.h>
#include <gpt/compute/model.h>
#include <gpt/compute/gpt_cuda.cuh>
#include <gpt/att/sliding_window.h>
#include <lib/hp_timer/hp_timer.h>
#include <emmintrin.h>


// working version
//   toArr - not needed, keep only relevant vector in kvcache
// optimize
//   batching & multithreading
//   valLookup -> i8 (or i32?, need i16 exp precision)
//   sse
//   precompute att sink
//   DISCR_SCALE <- can use shift for certain discr_scale values
//   mmap-able model params

namespace NCPUInfer
{
struct TCPUModelParams
{
    struct TAttentionMatrices
    {
        TArray2D<i8> QK;
        TArray2D<i8> QV;
        TArray2D<i8> K;
        TArray2D<i8> V;
        TArray2D<i8> Combiner;
        float QVScale = 0;
        float VScale = 0;
        float CombinerScale = 0;
        int AttentionWidth = 0;
    };
    TModelDim ModelDim;
    TArray2D<i8> LabelEmbed;
    float LabelEmbedScale = 0;
    TVector<TVector<TAttentionMatrices>> LayerArr;
    TArray2D<i8> FinalLayer;
    float FinalLayerScale = 0;
    TVector<float> Bias;
};


static i8 ConvertToInt8(float x)
{
    int res = _mm_cvtss_si32(_mm_set_ss(x));
    return ClampVal<int>(res, -127, 127); // -128 is incompatible with signed * unsigned ops
}

static float ConvertMatrix(const TArray2D<float> &data, TArray2D<i8> *p)
{
    yint xSize = data.GetXSize();
    yint ySize = data.GetYSize();
    float sum2 = 0;
    for (yint y = 0; y < ySize; ++y) {
        for (yint x = 0; x < xSize; ++x) {
            sum2 += Sqr(data[y][x]);
        }
    }
    float sko = sqrt(sum2 / (xSize * ySize));
    float discrScale = sko * MODEL_DISCR_SCALE;
    float mult = (sko == 0) ? 0 : (1 / discrScale);
    p->SetSizes(xSize, ySize);
    for (yint y = 0; y < ySize; ++y) {
        for (yint x = 0; x < xSize; ++x) {
            (*p)[y][x] = ConvertToInt8(data[y][x] * mult);
        }
    }
    return discrScale;
}

static float ConvertMatrix(TModelMatrixRowDisp &data, TArray2D<i8> *p)
{
    return ConvertMatrix(data.GetMatrix(), p);
}

static void ConvertAtt(const TModelParams::TAttentionMatrices &att, TCPUModelParams::TAttentionMatrices *p)
{
    ConvertMatrix(att.QK, &p->QK);
    p->QVScale = ConvertMatrix(att.QV, &p->QV);
    ConvertMatrix(att.K, &p->K);
    p->VScale = ConvertMatrix(att.V, &p->V);
    p->CombinerScale = ConvertMatrix(att.Combiner, &p->Combiner);
}

void ConvertModel(TModelParams &params, TCPUModelParams *p)
{
    p->ModelDim = params.ModelDim;
    p->LabelEmbedScale = ConvertMatrix(params.LabelEmbed, &p->LabelEmbed);
    p->LayerArr.resize(YSize(params.LayerArr));
    for (yint layerId = 0; layerId < YSize(params.LayerArr); ++layerId) {
        yint cc = YSize(params.LayerArr[layerId]);
        p->LayerArr[layerId].resize(cc);
        for (yint k = 0; k < cc; ++k) {
            TCPUModelParams::TAttentionMatrices &resAtt = p->LayerArr[layerId][k];
            ConvertAtt(params.LayerArr[layerId][k], &resAtt);
            // pick up width
            const TModelDim::TAttentionPosParams &attPosParams = params.ModelDim.Layers[layerId][k];
            Y_VERIFY(attPosParams.AlibiHyper == 0 && attPosParams.AlibiSlope == 0); // need support if used
            Y_VERIFY((attPosParams.AttentionWidthId & (ATT_ID_CREATE_WIDE_FLAG | ATT_ID_USE_WIDE_FLAG)) == 0); // yoco not supported
            resAtt.AttentionWidth = params.ModelDim.AttentionWidthArr[attPosParams.AttentionWidthId];
        }
    }
    p->FinalLayerScale = ConvertMatrix(params.FinalLayer, &p->FinalLayer);
    p->Bias = params.Bias;
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// SSE utils
// 
inline int HorizontalSumInt(__m256i v)
{
    // Use SSE2 functions to extract the lower and higher 128 bits
    __m128i vlow = _mm256_castsi256_si128(v);
    __m128i vhigh = _mm256_extracti128_si256(v, 1);

    // Perform pairwise addition of 32-bit integers
    vlow = _mm_add_epi32(vlow, vhigh);

    // Shuffle and add until we get the sum across the vector
    __m128i shuf = _mm_shuffle_epi32(vlow, _MM_SHUFFLE(0, 3, 2, 1)); // Shuffle the elements
    vlow = _mm_add_epi32(vlow, shuf);
    shuf = _mm_shuffle_epi32(vlow, _MM_SHUFFLE(1, 0, 3, 2)); // Shuffle again
    vlow = _mm_add_epi32(vlow, shuf);

    // Extract the sum
    return _mm_extract_epi32(vlow, 0);
}


static inline __m256i dp64(const __m256i x1, const __m256i x2, const __m256i y1, const __m256i y2, const __m256i sum)
{
    // glorious Intel does not support VNNI in 12xxx - 14xxx cpus, use legacy instructions
    //sum = _mm256_dpbssd_epi32(aPtr[i], bPtr[i], sum);

    __m256i ax = _mm256_sign_epi8(x1, x1);
    __m256i sy = _mm256_sign_epi8(y1, x1);
    __m256i sum1 = _mm256_dpbusd_avx_epi32(sum, ax, sy);
    ax = _mm256_sign_epi8(x2, x2);
    sy = _mm256_sign_epi8(y2, x2);
    __m256i sum2 = _mm256_dpbusd_avx_epi32(sum1, ax, sy);
    return sum2;
}


static i32 DotInt8(const i8 *aData, const i8 *bData, yint sz)
{
    __m256i sum = _mm256_setzero_si256();
    const __m256i *aPtr = (const __m256i *)aData;
    const __m256i *bPtr = (const __m256i *)bData;
    for (yint i = 0; i < sz / 32; i += 2) {
        sum = dp64(aPtr[i], aPtr[i + 1], bPtr[i], bPtr[i + 1], sum);
        //_mm_prefetch((const char *)(aPtr + 4), _MM_HINT_NTA);
        //_mm_prefetch((const char *)(bPtr + 4), _MM_HINT_NTA);
    }
    return HorizontalSumInt(sum);
}

static i32 DotInt8(const TVector<i8> &a, const TVector<i8> &b)
{
    yint sz = YSize(a);
    Y_ASSERT(sz == YSize(b));
    return DotInt8(a.data(), b.data(), sz);
}


struct TSoftMaxBuf
{
    TVector<float> Buf;
    yint Ptr = 0;
    float MaxValue = 0;
    float Scale = 0;

    TSoftMaxBuf()
    {
        Buf.resize(8, -1e38f);
    }

    void Clear()
    {
        Ptr = 0;
        MaxValue = 0;
    }

    void Add(float x)
    {
        if (Ptr == YSize(Buf)) {
            Buf.resize(YSize(Buf) * 2, -1e38f);
        }
        Buf[Ptr++] = x;
        MaxValue = Max<float>(MaxValue, x);
    }

    void SoftMax()
    {
        yint sz = (Ptr + 7) / 8;
        float sumWeight = 0;
        __m256 *dataBuf = (__m256 *)Buf.data();
        __m256 sum = _mm256_setzero_ps();
        __m256 maxValue = _mm256_set1_ps(MaxValue);
        for (yint i = 0; i < sz; ++i) {
            // exp avx by Imperator@
            __m256 x = _mm256_sub_ps(dataBuf[i], maxValue);
            x = _mm256_max_ps(x, _mm256_set1_ps(-127));
            __m256 xf = _mm256_floor_ps(x);
            x = _mm256_sub_ps(x, xf);
            __m256 s = _mm256_sub_ps(x, xf);
            __m256i xfi = _mm256_cvtps_epi32(xf);

            __m256 x2 = _mm256_mul_ps(x, x);
            __m256 c0 = _mm256_set1_ps(-3.069678791803394491901405992213472390777e-1f);
            __m256 c1 = _mm256_set1_ps(-6.558811624324781017147952441210509604385e-2f);
            __m256 c2 = _mm256_set1_ps(-1.355574723481491770403079319055785445381e-2f);
            __m256 res = _mm256_fmadd_ps(_mm256_fmadd_ps(c2, x, c1), x, c0);

            __m256 one = _mm256_set1_ps(1);
            __m256 x_by_1_minus_x = _mm256_sub_ps(x, x2);
            res = _mm256_fmadd_ps(res, x_by_1_minus_x, x);
            res = _mm256_add_ps(res, one); //adding ymm_x and 1 separately in the end improves accuracy

            xfi = _mm256_slli_epi32(xfi, 23);
            res = _mm256_castsi256_ps(_mm256_add_epi32(xfi, _mm256_castps_si256(res)));
            dataBuf[i] = res;
            sum = _mm256_add_ps(sum, res);
        }
        Scale = 1 / HorizontalSum(sum);
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
// linear algebra
template <class T>
void PrintVec(const TVector<T> &vec)
{
    yint sz = Min<yint>(128, YSize(vec));
    for (yint i = 0; i < sz; ++i) {
        DebugPrintf("cpu vec[%g] = %g\n", i * 1., vec[i] * 1.);
    }
}

// resArr = kqv @ vecArr
static void MulForward(const TVector<i8> &vec, const TArray2D<i8> &kqv, TVector<i32> *resArr)
{
    yint dim = YSize(vec);
    yint rDim = kqv.GetYSize();
    Y_ASSERT(dim == kqv.GetXSize());
    Y_ASSERT(rDim == kqv.GetYSize());
    resArr->resize(rDim);
    for (yint k = 0; k < rDim; ++k) {
        (*resArr)[k] = DotInt8(vec.data(), &kqv[k][0], dim);
    }
}


static void AddScaled(TVector<float> *pRes, const TVector<i32> &delta, float scale)
{
    yint sz = YSize(*pRes);
    Y_ASSERT(sz == YSize(delta));
    for (yint k = 0; k < sz; ++k) {
        (*pRes)[k] += delta[k] * scale;
    }
}


static void KVProduct(const TVector<i8> &kState, const TVector<float> &valLookup,
    TVector<i8> *pKVState)
{
    yint ttDim = YSize(kState);
    Y_ASSERT(YSize(valLookup) == ttDim);
    pKVState->resize(GetCombinerWidth(ttDim));
    for (int blk = 0; blk < COMBINER_REP; ++blk) {
        yint base = blk * ttDim;
        for (yint k = 0; k < ttDim; ++k) {
            i8 keyShfl = kState[k ^ blk];
            float value = valLookup[k];
            (*pKVState)[base + k] = ConvertToInt8(keyShfl * value);
        }
    }
}


static void SoftMax(const TVector<float> &bias, const TVector<i32> &vec1, float vecScale1, const TVector<i32> &vec2, float vecScale2, TVector<float> *pPrediction)
{
    yint dim = YSize(bias);
    Y_ASSERT(YSize(vec1) == dim);
    Y_ASSERT(YSize(vec2) == dim);
    TSoftMaxBuf buf; // can be static
    for (yint k = 0; k < dim; ++k) {
        float w = vec1[k] * vecScale1 + vec2[k] * vecScale2 + bias[k]; // can be vectorized
        buf.Add(w);
    }
    buf.SoftMax();
    pPrediction->resize(dim);
    for (yint k = 0; k < dim; ++k) {
        (*pPrediction)[k] = buf.Buf[k] * buf.Scale;
    }
}


template <class TSrc>
static float NormalizeState(TVector<i8> *pRes, const TVector<TSrc> &state)
{
    yint dim = YSize(state);
    pRes->resize(dim);
    float sum2 = 0;
    for (yint x = 0; x < dim; ++x) {
        sum2 += Sqr((float)state[x]);
    }
    if (sum2 == 0) {
        for (yint x = 0; x < dim; ++x) {
            (*pRes)[x] = 0;
        }
        return 0;
    } else {
        float sko = sqrt(sum2 / dim);
        float discrScale = sko * MODEL_DISCR_SCALE;
        float mult = 1 / discrScale;
        for (yint x = 0; x < dim; ++x) {
            (*pRes)[x] = ConvertToInt8(state[x] * mult);
        }
        return discrScale;
    }
}


template <class TSrc>
static float NormalizeState2(TVector<i8> *pRes1, TVector<i8> *pRes2, const TVector<TSrc> &state)
{
    yint dim = YSize(state);
    pRes1->resize(dim);
    pRes2->resize(dim);
    float sum2 = 0;
    for (yint x = 0; x < dim; ++x) {
        sum2 += Sqr((float)state[x]);
    }
    if (sum2 == 0) {
        for (yint x = 0; x < dim; ++x) {
            (*pRes1)[x] = 0;
            (*pRes2)[x] = 0;
        }
        return 0;
    } else {
        float sko = sqrt(sum2 / dim);
        float discrScale = sko * MODEL_DISCR_SCALE;
        float mult = 1 / discrScale;
        for (yint x = 0; x < dim; ++x) {
            float val = state[x] * mult;
            i8 res1 = ConvertToInt8(val);
            i8 res2 = ConvertToInt8((val - res1) * 128);
            (*pRes1)[x] = res1;
            (*pRes2)[x] = res2;
        }
        return discrScale;
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// Attention

struct TAttentionVecHistory
{
    TVector<TVector<i8>> QVState;
    TVector<float> QVStateScale;
    TVector<TVector<i8>> VState;

    void AddVectors(const TVector<i8> &qv, float qvScale, const TVector<i8> &v)
    {
        QVState.push_back(qv);
        QVStateScale.push_back(qvScale);
        VState.push_back(v);
    }
    yint GetLength() const { return YSize(VState); }
};


void ComputeValLookup(yint width, yint qDim, yint ttDim,
    const TAttentionVecHistory &history,
    const TVector<i8> &qkState,
    TVector<float> *pValLookup)
{
    TVector<int> toArr;
    yint len = history.GetLength();
    if (len > width) {
        toArr.push_back(0);
        for (yint dt = 1; dt <= width; ++dt) {
            toArr.push_back(len - dt);
        }
    } else {
        for (yint t = 0; t < len; ++t) {
            toArr.push_back(t);
        }
    }

    yint toCount = YSize(toArr);
    TVector<float> weightArr;
    weightArr.resize(toCount);

    TSoftMaxBuf softMax;
    softMax.Add(0);
    float attDotScale = CalcDotScaleAttention(qDim);
    for (yint z = 0; z < toCount; ++z) {
        yint to = toArr[z];
        i32 qProduct = DotInt8(qkState, history.QVState[to]);
        softMax.Add(qProduct * history.QVStateScale[to] * attDotScale * MODEL_DISCR_SCALE);
    }
    softMax.SoftMax();

    TVector<float> &valLookup = *pValLookup;
    ClearPodArray(&valLookup, ttDim);
    for (yint z = 0; z < toCount; ++z) {
        yint to = toArr[z];
        float w = softMax.Buf[z + 1] * softMax.Scale * MODEL_DISCR_SCALE;
        for (yint x = 0; x < ttDim; ++x) {
            valLookup[x] += w * history.VState[to][x];
        }
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// add product

static void AddLookupProduct(
    const TModelDim &modelDim,
    const TVector<TCPUModelParams::TAttentionMatrices> &layerAtt,
    TVector<TAttentionVecHistory> *pKVCache,
    TVector<float> *pState)
{
    yint dim = modelDim.Dim;
    yint qDim = modelDim.QDim;
    yint ttDim = modelDim.TTDim;
    Y_ASSERT(dim == YSize(*pState));
    TVector<TAttentionVecHistory> &kvCache = *pKVCache;

    TVector<i8> normState;
    NormalizeState(&normState, *pState);

    yint attCount = YSize(layerAtt);
    Y_ASSERT(YSize(kvCache) == attCount);
    for (yint z = 0; z < attCount; ++z) {
        const TCPUModelParams::TAttentionMatrices &att = layerAtt[z];

        TVector<i32> qkSrc;
        MulForward(normState, att.QK, &qkSrc);
        TVector<i32> qvSrc;
        MulForward(normState, att.QV, &qvSrc);
        TVector<i32> kSrc;
        MulForward(normState, att.K, &kSrc);
        TVector<i32> vSrc;
        MulForward(normState, att.V, &vSrc);

        TVector<i8> qk;
        NormalizeState(&qk, qkSrc);
        TVector<i8> qv;
        float qvScale = NormalizeState(&qv, qvSrc) * att.QVScale * MODEL_DISCR_SCALE;
        TVector<i8> k;
        NormalizeState(&k, kSrc);
        TVector<i8> v;
        NormalizeState(&v, vSrc);
        //PrintVec(k);

        TVector<float> valLookup;
        ComputeValLookup(att.AttentionWidth, qDim, ttDim, kvCache[z], qk, &valLookup);

        kvCache[z].AddVectors(qv, qvScale, v);

        TVector<i8> kv;
        KVProduct(k, valLookup, &kv);
        TVector<i32> deltaState;
        MulForward(kv, att.Combiner, &deltaState);
        AddScaled(pState, deltaState, att.CombinerScale * MODEL_DISCR_SCALE);
    }
}



///////////////////////////////////////////////////////////////////////////////////////////////////
//

struct TCPUInferContext
{
    TVector<TVector<TAttentionVecHistory>> KVcacheArr;

public:
    void Init(const TCPUModelParams &params)
    {
        yint depth = YSize(params.LayerArr);
        KVcacheArr.resize(depth);
        for (yint d = 0; d < depth; ++d) {
            yint count = YSize(params.LayerArr[d]);
            KVcacheArr[d].resize(count);
        }
    }
};


void ComputePrediction(const TCPUModelParams &params, const TVector<TLabelIndex> &labels, TCPUInferContext *pCtx, TVector<float> *pResPrediction)
{
    TModelDim modelDim = params.ModelDim;
    yint dim = modelDim.Dim;

    // embedding
    TVector<float> state;
    ClearPodArray(&state, dim);
    for (TLabelIndex label : labels) {
        for (yint x = 0; x < dim; ++x) {
            state[x] += params.LabelEmbed[label][x] * params.LabelEmbedScale;
        }
    }

    // apply layers
    for (yint d = 0; d < YSize(params.LayerArr); ++d) {
        AddLookupProduct(modelDim, params.LayerArr[d], &pCtx->KVcacheArr[d], &state);
    }

    if (pResPrediction) {
        TVector<i8> finalState1;
        TVector<i8> finalState2;
        NormalizeState2(&finalState1, &finalState2, state);

        TVector<i32> prediction1;
        MulForward(finalState1, params.FinalLayer, &prediction1);
        TVector<i32> prediction2;
        MulForward(finalState2, params.FinalLayer, &prediction2);

        float finalScale1 = CalcDotScaleFinalLayer(dim) * params.FinalLayerScale * MODEL_DISCR_SCALE;
        float finalScale2 = finalScale1 / 128;
        SoftMax(params.Bias, prediction1, finalScale1, prediction2, finalScale2, pResPrediction);
    }
}


static int SampleFromDistr(TXRng &rng, const TVector<float> &distr, float temperature)
{
    // use gumbel max trick
    float best = -1e38f;
    yint res = 0;
    for (yint k = 0; k < YSize(distr); ++k) {
        //float score = distr[k] / -log(rng.GenRandReal3());
        float score = log(distr[k]) / temperature - log(-log(rng.GenRandReal3()));
        if (score > best) {
            best = score;
            res = k;
        }
    }
    return res;
}


void CpuInferenceProfile(const TCPUModelParams &cpuParams)
{
    TXRng rng(1313);
    DebugPrintf("start profiling\n");
    for (;;) {
        NHPTimer::STime tStart;
        NHPTimer::GetTime(&tStart);
        TLabelIndex prevLabel = 0;
        TCPUInferContext cpuCtx;
        cpuCtx.Init(cpuParams);
        for (yint t = 0; t < 100; ++t) {
            TVector<TLabelIndex> labelArr;
            labelArr.push_back(prevLabel);
            TVector<float> distr;
            ComputePrediction(cpuParams, labelArr, &cpuCtx, &distr);
            yint letter = rng.Uniform(cpuParams.ModelDim.VocabSize);//SampleFromDistr(rng, distr, 1);
            prevLabel = letter + 1 + 1;
        }
        DebugPrintf("%g secs\n", NHPTimer::GetTimePassed(&tStart));
    }
}


void Check()
{
    TXRng rng(1313);

    TModelParams params;
    Serialize(true, "D:/models/rus_big/eden_gpt_205k.bin", params);

    TCPUModelParams cpuParams;
    ConvertModel(params, &cpuParams);

    //CpuInferenceProfile(cpuParams);

    const yint CHECK_BATCH_SIZE = 1;
    yint nodeCount = 100;

    TIntrusivePtr<IModel> gpuModel = CreateModel(1, params);
    TIntrusivePtr<IComputeContext> gpuCtx = NCUDA_GPT::CreateContext(gpuModel, CHECK_BATCH_SIZE * nodeCount);

    TFragment frag;
    TLabelIndex prevLabel = 0;
    TCPUInferContext cpuCtx;
    cpuCtx.Init(cpuParams);
    float gpuLoss = 0;
    float cpuLoss = 0;
    for (yint t = 0; t < 30; ++t) {
        TVector<TFragment> xxFrag;
        xxFrag.push_back(frag);
        MakeTest(xxFrag, gpuCtx.Get(), MAIN_DEVICE);
        TVector<TVector<float>> gpuPredArr;
        gpuCtx->ComputeFragmentPredictions(&gpuPredArr);

        TVector<TLabelIndex> labelArr;
        labelArr.push_back(prevLabel);
        TVector<float> cpuDistr;
        ComputePrediction(cpuParams, labelArr, &cpuCtx, &cpuDistr);

        //for (yint k = 0; k < 5; ++k) {
        //    DebugPrintf("%g - %g\n", cpuDistr[k], gpuPredArr.back()[k]);
        //}
        //DebugPrintf("\n");

        //yint letter = rng.Uniform(params.GetModelDim().VocabSize);
        //yint letter = SampleFromDistr(rng, cpuDistr, 1);
        yint letter = SampleFromDistr(rng, gpuPredArr.back(), 1);
        prevLabel = letter + 1 + 1;
        frag.Text.push_back(letter);

        cpuLoss -= log(cpuDistr[letter]);
        gpuLoss -= log(gpuPredArr.back()[letter]);
    }
    DebugPrintf("cpu loss %g\ngpu loss %g\n", cpuLoss, gpuLoss);
}
}
