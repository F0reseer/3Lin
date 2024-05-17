#include "stdafx.h"
#include "gpt_cpu.h"
#include <gpt/data/data.h>
#include <lib/random/rand_utils.h>
#include <lib/math/matrix_utils.h>
#include <xmmintrin.h> // for SSE intrinsics


// simulate int8 truncation
//#define SIM_I8_MATRIX
//#define SIM_I8_VECS


namespace NCPU_GPT
{
class fp16
{
    //unsigned short Val;
    float Val;

    void FromFloat(float x)
    {
        //Val = __float2half(x);
        Val = x;
        *(int *)&Val &= 0xffff0000; // bfloat16
    }
    float ToFloat() const
    {
        //return __internal_half2float(Val);
        return Val;
    }
public:
    fp16() {}
    fp16(float x) { FromFloat(x); } 
    operator float() const { return Val; }
    // 
    fp16 &operator+=(fp16 x) { FromFloat(float(Val) + float(x.Val)); return *this; }
    fp16 &operator-=(fp16 x) { FromFloat(float(Val) - float(x.Val)); return *this; }
    fp16 &operator*=(fp16 x) { FromFloat(float(Val) * float(x.Val)); return *this; }
};

template <class T>
T Sqrt(T x)
{
    return sqrt(x);
}

template<>
fp16 Sqrt<fp16>(fp16 x)
{
    float val = x;
    return fp16(sqrtf(val));
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// Convert a float to an int
static int Float2I8(float arg)
{
    int x = _mm_cvt_ss2si(_mm_set_ss(arg));
    x = (x > 127) ? 127 : x;
    x = (x < -128) ? -128 : x;
    return x;
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// 
// most flops use this precision
//typedef fp16 TFloat;
typedef float TFloat;
//typedef double TFloat;

//typedef fp16 TFastFloat;
typedef float TFastFloat;
//typedef double TFastFloat;

typedef TFloat TAccumFloat;


#define LOG2 0.693147

inline TFloat CalcDotScale(yint dim)
{
    return Sqrt<TFloat>(1. / dim) / LOG2;
}

inline TFloat GetStateLength(yint dim)
{
    return Sqrt<TFloat>(1. * dim);
}

inline TFloat GetAttentionDecay(int dist, float alibiSlope, float alibiHyper)
{
    Y_ASSERT(dist > 0);
    return -alibiSlope * dist + alibiHyper / dist;
}


///////////////////////////////////////////////////////////////////////////////////////////////////
#ifdef SIM_I8_MATRIX
// reproduce int8
static TArray2D<float> GetData(TIntrusivePtr<TModelMatrix> p)
{
    TArray2D<float> res;
    p->GetDataFast(&res);
    return res;
}
#else
// fast exact
static const TArray2D<float> &GetData(TIntrusivePtr<TModelMatrix> p)
{
    return p->GetData();
}
#endif


template <class T>
static void PrintVec(int t, const TArray2D<T> &arr)
{
    for (int k = 0; k < arr.GetXSize(); ++k) {
        printf("cpu vec[%g] = %g\n", k * 1., arr[t][k]);
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// 
template <class T1, class T2>
static void CopyMatrix(TVector<TVector<T1>> *p, const TArray2D<T2> &src)
{
    p->resize(src.GetYSize());
    for (yint y = 0; y < src.GetYSize(); ++y) {
        (*p)[y].resize(src.GetXSize());
        for (yint x = 0; x < src.GetXSize(); ++x) {
            (*p)[y][x] = src[y][x];
        }
    }
}


template <class T1, class T2>
static void InitDeltaMatrix(TArray2D<T1> *p, const TArray2D<T2> &src)
{
    p->SetSizes(src.GetXSize(), src.GetYSize());
    p->FillZero();
}


template <class T1, class T2, class T3>
static void AddScaledMatrix(TArray2D<T1> *p, const TArray2D<T2> &src, T3 scale)
{
    yint xSize = src.GetXSize();
    yint ySize = src.GetYSize();
    Y_ASSERT(p->GetXSize() == xSize);
    Y_ASSERT(p->GetYSize() == ySize);
    for (yint y = 0; y < ySize; ++y) {
        for (yint x = 0; x < xSize; ++x) {
            (*p)[y][x] += src[y][x] * scale;
        }
    }
}


template <class T1, class T2>
static void ScaleMatrix(TArray2D<T1> *p, T2 scale)
{
    yint xSize = p->GetXSize();
    yint ySize = p->GetYSize();
    for (yint y = 0; y < ySize; ++y) {
        for (yint x = 0; x < xSize; ++x) {
            (*p)[y][x] *= scale;
        }
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// linear algebra

// resArr = kqv @ vecArr
static void MulForward(const TArray2D<TFastFloat> &vecArr, const TArray2D<TFastFloat> &kqv, TArray2D<TFastFloat> *resArr)
{
    yint len = vecArr.GetYSize();
    yint dim = vecArr.GetXSize();
    yint rDim = kqv.GetYSize();
    Y_ASSERT(dim == kqv.GetXSize());
    Y_ASSERT(rDim == kqv.GetYSize());
    resArr->SetSizes(rDim, len);
    for (yint t = 0; t < len; ++t) {
        for (yint k = 0; k < rDim; ++k) {
            TAccumFloat res = 0;
            for (yint x = 0; x < dim; ++x) {
                res += vecArr[t][x] * kqv[k][x];
            }
            (*resArr)[t][k] = res;
        }
    }
}


static void MulBackwardWithAccum(TArray2D<TFastFloat> *pVecArrGrad, const TArray2D<TFastFloat> &kqv, const TArray2D<TFastFloat> &resArrGrad)
{
    yint len = resArrGrad.GetYSize();
    yint dim = kqv.GetXSize();
    yint rDim = resArrGrad.GetXSize();
    Y_ASSERT(dim == kqv.GetXSize());
    Y_ASSERT(rDim == kqv.GetYSize());
    for (yint t = 0; t < len; ++t) {
        for (yint x = 0; x < dim; ++x) {
            TAccumFloat res = 0;
            for (yint k = 0; k < rDim; ++k) {
                res += resArrGrad[t][k] * kqv[k][x];
            }
            (*pVecArrGrad)[t][x] += res;
        }
    }
}


static void SumRankOne(const TArray2D<TFastFloat> &vecArr, TArray2D<TFloat> *pDelta, const TArray2D<TFastFloat> &resArrGrad)
{
    yint len = vecArr.GetYSize();
    yint dim = vecArr.GetXSize();
    yint rDim = resArrGrad.GetXSize();
    Y_ASSERT(len == resArrGrad.GetYSize());
    pDelta->SetSizes(dim, rDim);
    pDelta->FillZero();
    for (yint k = 0; k < rDim; ++k) {
        for (yint x = 0; x < dim; ++x) {
            TAccumFloat res = 0;
            for (yint t = 0; t < len; ++t) {
                res += resArrGrad[t][k] * vecArr[t][x];
            }
            (*pDelta)[k][x] += res;
        }
    }
}


static TFloat CalcSum2(const TArray2D<TFloat> &delta)
{
    TAccumFloat sum2 = 0;
    yint xSize = delta.GetXSize();
    yint ySize = delta.GetYSize();
    for (yint y = 0; y < ySize; ++y) {
        for (yint x = 0; x < xSize; ++x) {
            sum2 += Sqr(delta[y][x]);
        }
    }
    return sum2;
}


template <class TTargetFloat>
static void AddScaledMatrix(TArray2D<TTargetFloat> *pRes, const TArray2D<TFloat> &delta, TFloat scale)
{
    yint ySize = pRes->GetYSize();
    yint xSize = pRes->GetXSize();
    Y_ASSERT(xSize == delta.GetXSize());
    Y_ASSERT(ySize == delta.GetYSize());
    for (yint y = 0; y < ySize; ++y) {
        for (yint x = 0; x < xSize; ++x) {
            (*pRes)[y][x] += delta[y][x] * scale;
        }
    }
}


static void KVProduct(const TArray2D<TFastFloat> &kState, const TArray2D<TFastFloat> &valLookup,
    TArray2D<TFastFloat> *pKVState)
{
    yint ttDim = kState.GetXSize();
    yint len = kState.GetYSize();
    Y_ASSERT(valLookup.GetXSize() == ttDim);
    Y_ASSERT(valLookup.GetYSize() == len);
    pKVState->SetSizes(GetCombinerWidth(ttDim), len);
    for (yint t = 0; t < len; ++t) {
        TAccumFloat res = 0;
        for (int blk = 0; blk < COMBINER_REP; ++blk) {
            yint base = blk * ttDim;
            for (yint k = 0; k < ttDim; ++k) {
                TFastFloat keyShfl = kState[t][k ^ blk];
                TFastFloat value = valLookup[t][k];
                (*pKVState)[t][base + k] = keyShfl * value;
            }
        }
    }
}


static void KVProductBackprop(const TArray2D<TFastFloat> &kState, const TArray2D<TFastFloat> &valLookup, const TArray2D<TFastFloat> &dkv,
    TArray2D<TFastFloat> *pDKState, TArray2D<TFastFloat> *pDValLookup, TVector<TFloat> *pDScale)
{
    yint ttDim = kState.GetXSize();
    yint len = kState.GetYSize();
    Y_ASSERT(valLookup.GetXSize() == ttDim);
    Y_ASSERT(valLookup.GetYSize() == len);
    Y_ASSERT(dkv.GetXSize() == GetCombinerWidth(ttDim));
    Y_ASSERT(dkv.GetYSize() == len);
    ClearPodArray(pDScale, len);

    TArray2D<TAccumFloat> dKey;
    dKey.SetSizes(ttDim, len);
    dKey.FillZero();
    TArray2D<TAccumFloat> dValLookup;
    dValLookup.SetSizes(ttDim, len);
    dValLookup.FillZero();
    for (yint t = 0; t < len; ++t) {
        TAccumFloat dScale = 0;
        for (int blk = 0; blk < COMBINER_REP; ++blk) {
            yint base = blk * ttDim;
            for (yint k = 0; k < ttDim; ++k) {
                TFastFloat keyShfl = kState[t][k ^ blk];
                TAccumFloat dKeyShfl = dkv[t][base + (k ^ blk)] * valLookup[t][k ^ blk];
                TFastFloat value = valLookup[t][k];
                dKey[t][k] += dKeyShfl;
                dValLookup[t][k] += dkv[t][base + k] * keyShfl;
                dScale += dkv[t][base + k] * keyShfl * value;
            }
        }
        (*pDScale)[t] += dScale;
    }
    CopyMatrix(pDKState, dKey);
    CopyMatrix(pDValLookup, dValLookup);
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// 
static void SoftMax(const TArray2D<TFastFloat> &vecArr, TVector<TVector<float>> *pPrediction, const TVector<float> &bias)
{
    yint len = vecArr.GetYSize();
    yint dim = YSize(bias);
    Y_ASSERT(vecArr.GetXSize() == dim);
    pPrediction->resize(len);
    for (yint t = 0; t < len; ++t) {
        TVector<float> &dst = (*pPrediction)[t];
        dst.resize(dim);
        double sumWeight = 0;
        for (yint k = 0; k < dim; ++k) {
            float w = exp2(vecArr[t][k] + bias[k]);
            dst[k] = w;
            sumWeight += w;
        }
        float scale = 1 / sumWeight;
        for (yint k = 0; k < dim; ++k) {
            dst[k] *= scale;
        }
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
//
enum EDiscr
{
    DISCR_NONE,
    DISCR_BYPASS,
#ifdef SIM_I8_VECS
    DISCR_I8,
    DISCR_I8_KEEP_SCALE,
#else
    DISCR_I8 = DISCR_NONE,
    DISCR_I8_KEEP_SCALE = DISCR_BYPASS,
#endif
};

template <class TSrc>
static void NormalizeState(TArray2D<TFastFloat> *pRes, const TArray2D<TSrc> &state, EDiscr dd)
{
    if (dd == DISCR_BYPASS) {
        *pRes = state;
        return;
    }
    yint len = state.GetYSize();
    yint dim = state.GetXSize();
    TFloat stateScale = GetStateLength(dim);
    pRes->SetSizes(dim, len);
    for (yint t = 0; t < len; ++t) {
        TAccumFloat sum2 = 0;
        for (yint x = 0; x < dim; ++x) {
            sum2 += Sqr(state[t][x]);
        }
        if (sum2 == 0) {
            for (yint x = 0; x < dim; ++x) {
                (*pRes)[t][x] = 0;
            }
        } else {
            TFloat scale = stateScale / Sqrt<TFloat>(sum2);
            for (yint x = 0; x < dim; ++x) {
                TFloat val = state[t][x] * scale;
                if (dd != DISCR_NONE) {
                    int res = Float2I8(val * (1.0f / DISCR_SCALE));
                    if (dd == DISCR_I8) {
                        val = res * DISCR_SCALE;
                    } else if (dd == DISCR_I8_KEEP_SCALE) {
                        val = res * (DISCR_SCALE / scale);
                    }
                }
                (*pRes)[t][x] = val;
            }
        }
    }
}


template <class T1, class T2, class T3>
static void NormalizeStateBackward(const TArray2D<T1> &state, const TArray2D<T2> &dNormState, TArray2D<T3> *pGrad)
{
    yint len = state.GetYSize();
    yint dim = state.GetXSize();
    pGrad->SetSizes(dim, len);
    TFloat stateScale = GetStateLength(dim);
    for (yint t = 0; t < len; ++t) {
        TAccumFloat sum2 = 0;
        TAccumFloat dp = 0;
        for (yint x = 0; x < dim; ++x) {
            T1 src = state[t][x];
            T2 grad = dNormState[t][x];
            sum2 += Sqr(src);
            dp += src * grad;
        }
        if (sum2 == 0) {
            for (yint x = 0; x < dim; ++x) {
                (*pGrad)[t][x] = 0;
            }
        } else {
            TAccumFloat sigma = dp / sum2;
            TAccumFloat scale = stateScale / Sqrt<TFloat>(sum2);
            for (yint x = 0; x < dim; ++x) {
                T1 src = state[t][x];
                T2 grad = dNormState[t][x];
                (*pGrad)[t][x] = scale * (grad - src * sigma);
            }
        }
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// Attention

// attention related compute
struct TAttentionComputer
{
    TVector<TAccumFloat> SumWeight;
    TFloat DotScale = 0;
    float AlibiSlope = 0;
    float AlibiHyper = 0;

    TAttentionComputer(yint qDim, float alibiSlope, float alibiHyper) : AlibiSlope(alibiSlope), AlibiHyper(alibiHyper)
    {
        DotScale = CalcDotScale(qDim);
    }

    void ComputeValLookup(yint len, yint qDim, yint ttDim,
        const TArray2D<TFastFloat> &qkState, const TArray2D<TFastFloat> &qvState, const TArray2D<TFastFloat> &vState,
        const TAttentionInfo &attInfo,
        TArray2D<TFastFloat> *pValLookup)
    {
        TVector<TAccumFloat> valLookup;
        pValLookup->SetSizes(ttDim, len);
        SumWeight.resize(len);

        // compute weighted sum of val vectors
        for (yint from = 0; from < len; ++from) {
            TAccumFloat sumWeight = 1; // initialize with zero vector of weight 1
            ClearPodArray(&valLookup, ttDim);
            for (yint attIndex = attInfo.SpanPtr[from]; attIndex < attInfo.SpanPtr[from + 1]; ++attIndex) {
                const TAttentionSpan &span = attInfo.Spans[attIndex];
                for (yint to = span.Start; to <= span.Finish; ++to) {
                    TAccumFloat sum = 0;
                    for (yint x = 0; x < qDim; ++x) {
                        sum += qkState[from][x] * qvState[to][x];
                    }
                    TAccumFloat dp = sum * DotScale;
                    dp += GetAttentionDecay(from - to, AlibiSlope, AlibiHyper);
                    TAccumFloat w = exp2(dp);
                    Y_ASSERT(!isnan(w) && isfinite(w));
                    sumWeight += w;
                    for (yint x = 0; x < ttDim; ++x) {
                        valLookup[x] += w * vState[to][x];
                    }
                }
            }
            SumWeight[from] = sumWeight;
            TAccumFloat sumWeight1 = sumWeight == 0 ? 0 : 1 / sumWeight;
            for (yint x = 0; x < ttDim; ++x) {
                (*pValLookup)[from][x] = valLookup[x] * sumWeight1;
            }
        }
    }

    void AddGradQK(yint len, yint qDim, yint ttDim,
        const TArray2D<TFastFloat> &qkState, const TArray2D<TFastFloat> &qvState, const TArray2D<TFastFloat> &vState,
        const TAttentionInfo &attInfo,
        const TArray2D<TFastFloat> &dValLookupArr, const TVector<TFloat> &dScaleArr,
        TArray2D<TFastFloat> *pDQKState)
    {
        TArray2D<TAccumFloat> dqkState;
        dqkState.SetSizes(qDim, len);
        dqkState.FillZero();
        for (yint from = 0; from < len; ++from) {
            for (yint attIndex = attInfo.SpanPtr[from]; attIndex < attInfo.SpanPtr[from + 1]; ++attIndex) {
                const TAttentionSpan &span = attInfo.Spans[attIndex];
                for (yint to = span.Start; to <= span.Finish; ++to) {
                    TAccumFloat sumWeight = SumWeight[from];
                    Y_ASSERT(sumWeight > 0);
                    TAccumFloat sum = 0;
                    for (yint x = 0; x < qDim; ++x) {
                        sum += qkState[from][x] * qvState[to][x];
                    }
                    TAccumFloat attDecay = GetAttentionDecay(from - to, AlibiSlope, AlibiHyper);
                    TAccumFloat w = exp2(sum * DotScale + attDecay) / sumWeight;
                    Y_ASSERT(!isnan(w) && isfinite(w));

                    TAccumFloat dW = 0;
                    for (yint x = 0; x < ttDim; ++x) {
                        TFastFloat dValLookup = dValLookupArr[from][x];
                        TFastFloat val = vState[to][x]; // val2
                        dW += dValLookup * val;
                    }

                    TFloat dScale = dScaleArr[from];
                    TFloat dDot = w * (dW - dScale) * DotScale * LOG2;
                    for (yint x = 0; x < qDim; ++x) {
                        dqkState[from][x] += dDot * qvState[to][x];
                    }
                }
            }
        }
        AddScaledMatrix(pDQKState, dqkState, 1);
    }

    void AddGradQV(yint len, yint qDim, yint ttDim,
        const TArray2D<TFastFloat> &qkState, const TArray2D<TFastFloat> &qvState, const TArray2D<TFastFloat> &vState,
        const TAttentionInfo &revAttInfo,
        const TArray2D<TFastFloat> &dValLookupArr, const TVector<TFloat> &dScaleArr,
        TArray2D<TFastFloat> *pDQVState, TArray2D<TFastFloat> *pDVState)
    {
        for (yint to = 0; to < len; ++to) {
            for (yint attIndex = revAttInfo.SpanPtr[to]; attIndex < revAttInfo.SpanPtr[to + 1]; ++attIndex) {
                const TAttentionSpan &span = revAttInfo.Spans[attIndex];
                for (yint from = span.Start; from <= span.Finish; ++from) {
                    TAccumFloat sumWeight = SumWeight[from];
                    Y_ASSERT(sumWeight > 0);
                    TAccumFloat sum = 0;
                    for (yint x = 0; x < qDim; ++x) {
                        sum += qkState[from][x] * qvState[to][x];
                    }
                    TAccumFloat attDecay = GetAttentionDecay(from - to, AlibiSlope, AlibiHyper);
                    TAccumFloat w = exp2(sum * DotScale + attDecay) / sumWeight;
                    Y_ASSERT(!isnan(w) && isfinite(w));

                    TAccumFloat dW = 0;
                    for (yint x = 0; x < ttDim; ++x) {
                        TFastFloat dValLookup = dValLookupArr[from][x];
                        TFastFloat val = vState[to][x]; // val2
                        dW += dValLookup * val;
                        (*pDVState)[to][x] += dValLookup * w;
                    }

                    TFloat dScale = dScaleArr[from];
                    TFloat dDot = w * (dW - dScale) * DotScale * LOG2;
                    for (yint x = 0; x < qDim; ++x) {
                        (*pDQVState)[to][x] += dDot * qkState[from][x];
                    }
                }
            }
        }
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
// add product

struct TFragmentStates
{
    TArray2D<TFloat> State;

    void SetLength(yint len, yint dim)
    {
        State.SetSizes(dim, len);
        State.FillZero();
    }
};


struct TAttentionFB
{
    TAttentionInfo Att;
    TAttentionInfo RevAtt;

    void Assign(const TAttentionInfo &att)
    {
        Att = att;
        RevAtt = TransposeAttention(att);
    }
};


// compute product of two attention lookups
static void AddLookupProduct(
    const TModelDim &modelDim,
    const TVector<const TAttentionParams *> &layerAtt,
    const TAttentionFB &attFB, const TAttentionFB &wideAttFB,
    const TFragmentStates &prevState, TFragmentStates *pState)
{
    int dim = modelDim.Dim;
    yint qDim = modelDim.QDim;
    int ttDim = modelDim.TTDim;
    yint len = prevState.State.GetYSize();
    Y_ASSERT(dim == prevState.State.GetXSize());

    TArray2D<TFastFloat> normState;
    NormalizeState(&normState, prevState.State, DISCR_I8);

    pState->State = prevState.State;
    for (const TAttentionParams *pAtt: layerAtt) {
        TAttentionComputer attComp(qDim, pAtt->AlibiSlope, pAtt->AlibiHyper);
        const TAttentionInfo &attInfo = pAtt->WideLayer ? wideAttFB.Att : attFB.Att;

        TArray2D<TFastFloat> qkSrc;
        MulForward(normState, GetData(pAtt->QK), &qkSrc);
        TArray2D<TFastFloat> qvSrc;
        MulForward(normState, GetData(pAtt->QV), &qvSrc);
        TArray2D<TFastFloat> kSrc;
        MulForward(normState, GetData(pAtt->K), &kSrc);
        TArray2D<TFastFloat> vSrc;
        MulForward(normState, GetData(pAtt->V), &vSrc);

        TArray2D<TFastFloat> qk;
        NormalizeState(&qk, qkSrc, DISCR_I8);
        TArray2D<TFastFloat> qv;
        NormalizeState(&qv, qvSrc, DISCR_I8_KEEP_SCALE);
        TArray2D<TFastFloat> k;
        NormalizeState(&k, kSrc, DISCR_I8);
        TArray2D<TFastFloat> v;
        NormalizeState(&v, vSrc, DISCR_I8);

        TArray2D<TFastFloat> valLookup;
        attComp.ComputeValLookup(len, qDim, ttDim, qk, qv, v, attInfo, &valLookup);

        TArray2D<TFastFloat> kv;
        KVProduct(k, valLookup, &kv);
        TArray2D<TFastFloat> deltaState;
        MulForward(kv, GetData(pAtt->Combiner), &deltaState);
        AddScaledMatrix(&pState->State, deltaState, 1);
    }
}


// add gradient of product of two attention lookups
static void AddLookupProductBackprop(
    const TModelDim &modelDim,
    const TVector<const TAttentionParams *> &layerAtt,
    const TAttentionFB &attFB, const TAttentionFB &wideAttFB,
    const TFragmentStates &prevState,
    TFragmentStates *pGrad,
    float step)
{
    yint len = prevState.State.GetYSize();
    yint dim = modelDim.Dim;
    yint qDim = modelDim.QDim;
    yint ttDim = modelDim.TTDim;

    TArray2D<TFastFloat> normState;
    NormalizeState(&normState, prevState.State, DISCR_I8);

    TArray2D<TFastFloat> dNormState;
    InitDeltaMatrix(&dNormState, normState);

    for (const TAttentionParams *pAtt : layerAtt) {
        TAttentionComputer attComp(qDim, pAtt->AlibiSlope, pAtt->AlibiHyper);
        const TAttentionInfo &attInfo = pAtt->WideLayer ? wideAttFB.Att : attFB.Att;
        const TAttentionInfo &revAttInfo = pAtt->WideLayer ? wideAttFB.RevAtt : attFB.RevAtt;

        // recompute forward pass (could keep them)
        TArray2D<TFastFloat> qkSrc;
        MulForward(normState, GetData(pAtt->QK), &qkSrc);
        TArray2D<TFastFloat> qvSrc;
        MulForward(normState, GetData(pAtt->QV), &qvSrc);
        TArray2D<TFastFloat> kSrc;
        MulForward(normState, GetData(pAtt->K), &kSrc);
        TArray2D<TFastFloat> vSrc;
        MulForward(normState, GetData(pAtt->V), &vSrc);

        TArray2D<TFastFloat> qk;
        NormalizeState(&qk, qkSrc, DISCR_I8);
        TArray2D<TFastFloat> qv;
        NormalizeState(&qv, qvSrc, DISCR_I8_KEEP_SCALE);
        TArray2D<TFastFloat> k;
        NormalizeState(&k, kSrc, DISCR_I8);
        TArray2D<TFastFloat> v;
        NormalizeState(&v, vSrc, DISCR_I8);

        TArray2D<TFastFloat> valLookup;
        attComp.ComputeValLookup(len, qDim, ttDim, qk, qv, v, attInfo, &valLookup);

        //PrintVec(10, valLookup);

        TArray2D<TFastFloat> kv;
        KVProduct(k, valLookup, &kv);
        TArray2D<TFastFloat> dkv;
        InitDeltaMatrix(&dkv, kv);
        MulBackwardWithAccum(&dkv, GetData(pAtt->Combiner), pGrad->State);
        TArray2D<TFloat> deltaCombiner;
        SumRankOne(kv, &deltaCombiner, pGrad->State);

        TArray2D<TFastFloat> dK;
        TArray2D<TFastFloat> dValLookup;
        TVector<TFloat> dScale;
        KVProductBackprop(k, valLookup, dkv, &dK, &dValLookup, &dScale);

        TArray2D<TFastFloat> dQK;
        InitDeltaMatrix(&dQK, qk);
        attComp.AddGradQK(len, qDim, ttDim, qk, qv, v, attInfo, dValLookup, dScale, &dQK);
        TArray2D<TFastFloat> dQV;
        InitDeltaMatrix(&dQV, qv);
        TArray2D<TFastFloat> dV;
        InitDeltaMatrix(&dV, v);
        attComp.AddGradQV(len, qDim, ttDim, qk, qv, v, revAttInfo, dValLookup, dScale, &dQV, &dV);

        NormalizeStateBackward(qkSrc, dQK, &dQK);
        NormalizeStateBackward(kSrc, dK, &dK);
        NormalizeStateBackward(vSrc, dV, &dV);

        MulBackwardWithAccum(&dNormState, GetData(pAtt->QK), dQK);
        MulBackwardWithAccum(&dNormState, GetData(pAtt->QV), dQV);
        MulBackwardWithAccum(&dNormState, GetData(pAtt->K), dK);
        MulBackwardWithAccum(&dNormState, GetData(pAtt->V), dV);

        TArray2D<float> deltaQK;
        SumRankOne(normState, &deltaQK, dQK);
        TArray2D<float> deltaQV;
        SumRankOne(normState, &deltaQV, dQV);
        TArray2D<float> deltaK;
        SumRankOne(normState, &deltaK, dK);
        TArray2D<float> deltaV;
        SumRankOne(normState, &deltaV, dV);

        pAtt->Combiner->ApplyDelta(deltaCombiner);
        pAtt->QK->ApplyDelta(deltaQK);
        pAtt->QV->ApplyDelta(deltaQV);
        pAtt->K->ApplyDelta(deltaK);
        pAtt->V->ApplyDelta(deltaV);
    }
    TArray2D<TFloat> stateGrad;
    NormalizeStateBackward(prevState.State, dNormState, &stateGrad);
    AddScaledMatrix(&pGrad->State, stateGrad, 1);

    // can normalize pGrad, all deltas are normalized anyway
}


///////////////////////////////////////////////////////////////////////////////////////////////////
//
class TComputeContext : public IComputeContext
{
    TIntrusivePtr<IModel> Model;
    TVector<TVector<const TAttentionParams *>> LayerArr;
    TVector<TFragmentStates> AllStates;
    TVector<TLabelIndex> LabelArr;
    TVector<ui32> LabelPtr;
    TVector<TNodeTarget> KeepTarget;
    TAttentionFB Att;
    TAttentionFB WideAtt;
    yint MaxNodeCount = 0;
    TArray2D<TFastFloat> FinalNormState;
    bool HasAsyncOps = false;
    TNodesBatch Nodes;
    TVector<ui32> DropTable;
public:
    TComputeContext(TIntrusivePtr<IModel> model, yint nodeCount) : Model(model), MaxNodeCount(nodeCount)
    {
        TModelDim modelDim = Model->GetModelDim();
        LayerArr.resize(YSize(modelDim.Layers));
        for (yint d = 0; d < YSize(modelDim.Layers); ++d) {
            for (yint k = 0; k < YSize(modelDim.Layers[d]); ++k) {
                LayerArr[d].push_back(&Model->GetAttention(d, k));
            }
        }
    }

    yint GetDeviceCount() override
    {
        return 1;
    }

    TModelDim GetModelDim() override
    {
        return Model->GetModelDim();
    }

    void GetParams(TModelParams *p) override
    {
        Model->WaitCompute();
        Model->GetParamsImpl(p);
    }

    void SetParams(const TModelParams &p) override
    {
        Model->WaitCompute();
        Model->SetParamsImpl(p);
    }

    void GetGradient(TModelParams *p) override
    {
        Model->WaitCompute();
        Model->GetGradientImpl(p);
    }

    TNodesBatch &GetNodes(yint deviceId) override
    {
        Y_ASSERT(deviceId == 0);
        return Nodes;
    }

    TVector<ui32> &GetDropTable(yint deviceId) override
    {
        Y_ASSERT(deviceId == 0);
        return DropTable;
    }

    void Init(yint deviceId) override
    {
        Y_ASSERT(deviceId == 0);
        Y_ASSERT(DropTable[0] == 0xffffffff); // TODO support dropout in cpu implementation
        TModelDim modelDim = Model->GetModelDim();
        yint len = Nodes.GetNodeCount();
        Y_ASSERT(len <= MaxNodeCount);
        yint depth = YSize(modelDim.Layers);
        AllStates.resize(depth + 1);
        for (yint k = 0; k < YSize(AllStates); ++k) {
            AllStates[k].SetLength(len, modelDim.Dim);
        }
        LabelArr = Nodes.LabelArr;
        LabelPtr = Nodes.LabelPtr;
        KeepTarget = Nodes.Target;
        Att.Assign(Nodes.Att);
        WideAtt.Assign(Nodes.WideAtt);
    }

    void ComputeEmbedding(const TArray2D<float> &labelEmbed)
    {
        TModelDim modelDim = Model->GetModelDim();
        int dim = modelDim.Dim;
        yint len = YSize(LabelPtr) - 1;

        AllStates[0].State.FillZero();
        for (yint t = 0; t < len; ++t) {
            for (yint k = LabelPtr[t], kFinish = LabelPtr[t + 1]; k < kFinish; ++k) {
                yint label = LabelArr[k];
                for (yint x = 0; x < dim; ++x) {
                    AllStates[0].State[t][x] += labelEmbed[label][x];
                }
            }
        }
    }

    void ComputeForward(TVector<TVector<float>> *pPrediction, TVector<TVector<float>> *pStateVectors)
    {
        TModelDim modelDim = Model->GetModelDim();
        int dim = modelDim.Dim;

        if (HasAsyncOps) {
            Model->WaitCompute();
            HasAsyncOps = false;
        }

        // embedding
        ComputeEmbedding(GetData(Model->GetLabelEmbed()));

        // apply layers
        for (yint d = 0; d < YSize(LayerArr); ++d) {
            AllStates[d + 1] = AllStates[d];
            AddLookupProduct(modelDim, LayerArr[d], Att, WideAtt, AllStates[d], &AllStates[d + 1]);
        }

        NormalizeState(&FinalNormState, AllStates.back().State, DISCR_NONE);

        if (pStateVectors) {
            CopyMatrix(pStateVectors, AllStates.back().State);
        }

        if (pPrediction) {
            TArray2D<TFastFloat> predictionArr;
            MulForward(FinalNormState, GetData(Model->GetFinalLayer()), &predictionArr);

            ScaleMatrix(&predictionArr, CalcDotScale(dim) * FINAL_LAYER_SOFTMAX_SCALE);
            SoftMax(predictionArr, pPrediction, Model->GetBias());
        }
    }

    void ComputeFinalStateVectors(TVector<TVector<float>> *pStateVectors) override
    {
        ComputeForward(0, pStateVectors);
    }

    void ComputeFragmentPredictions(TVector<TVector<float>> *pPrediction) override
    {
        ComputeForward(pPrediction, 0);
    }

    float ComputeScore() override
    {
        TVector<TVector<float>> prediction;
        ComputeForward(&prediction, 0);
        float sum = 0;
        yint count = 0;
        for (const TNodeTarget &nt : KeepTarget) {
            sum += log(prediction[nt.Node][nt.TargetId]);
            count += 1;
        }
        return sum / count;
    }

    void Backprop(float step) override
    {
        TModelDim modelDim = Model->GetModelDim();
        yint len = YSize(LabelPtr) - 1;
        int dim = modelDim.Dim;

        TVector<TVector<float>> predArr;
        ComputeForward(&predArr, 0);
        Y_ASSERT(YSize(predArr) == len);

        Y_ASSERT(!HasAsyncOps);
        Model->StartIteration(step);

        TFragmentStates grad;
        grad.SetLength(len, modelDim.Dim);
        {
            // final soft max gradient
            TArray2D<TFastFloat> gradArr;
            gradArr.SetSizes(modelDim.VocabSize, len);
            gradArr.FillZero();
            for (const TNodeTarget &nt : KeepTarget) {
                for (yint q = 0; q < modelDim.VocabSize; ++q) {
                    gradArr[nt.Node][q] += -predArr[nt.Node][q];
                }
                gradArr[nt.Node][nt.TargetId] += 1;
            }

            ScaleMatrix(&gradArr, CalcDotScale(dim) * LOG2);

            TArray2D<TFastFloat> normStateGrad;
            InitDeltaMatrix(&normStateGrad, FinalNormState);
            MulBackwardWithAccum(&normStateGrad, GetData(Model->GetFinalLayer()), gradArr);

            // modify final layer
            if (modelDim.HasFlag(MPF_TUNE_FINAL_LAYER)) {
                TArray2D<float> deltaFinalLayer;
                SumRankOne(FinalNormState, &deltaFinalLayer, gradArr);
                Model->GetFinalLayer()->ApplyDelta(deltaFinalLayer);
            }

            NormalizeStateBackward(AllStates.back().State, normStateGrad, &grad.State);
        }

        // modify layers
        for (yint d = YSize(LayerArr) - 1; d >= 0; --d) {
            AddLookupProductBackprop(modelDim, LayerArr[d], Att, WideAtt, AllStates[d], &grad, step);
        }

        // modify embedding
        if (modelDim.HasFlag(MPF_TUNE_EMBED)) {
            TArray2D<float> deltaLabel;
            deltaLabel.SetSizes(dim, modelDim.LabelCount);
            deltaLabel.FillZero();
            for (yint t = 0; t < len; ++t) {
                for (yint k = LabelPtr[t], kFinish = LabelPtr[t + 1]; k < kFinish; ++k) {
                    yint label = LabelArr[k];
                    for (yint x = 0; x < modelDim.Dim; ++x) {
                        deltaLabel[label][x] += grad.State[t][x];
                    }
                }
            }
            Model->GetLabelEmbed()->ApplyDelta(deltaLabel);
        }

        HasAsyncOps = true;
    }
};

TIntrusivePtr<IComputeContext> CreateContext(TIntrusivePtr<IModel> pModel, yint nodeCount)
{
    return new TComputeContext(pModel, nodeCount);
}
}
