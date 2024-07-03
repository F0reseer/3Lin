#pragma once
#include <gpt/rng/xrng.h>

constexpr yint COMBINER_REP = 16;

constexpr yint GetCombinerWidth(yint ttDim)
{
    return ttDim * COMBINER_REP;
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// what scaling is optimal and why we need it all is unclear
constexpr float LOG2 = 0.693147f;
constexpr float FINAL_LAYER_SOFTMAX_SCALE = 4;
constexpr float ATT_DOTPRODUCT_SCALE = 0.33f;

inline float CalcDotScaleFinalLayer(yint dim)
{
    return sqrt(1. / dim) / LOG2 * FINAL_LAYER_SOFTMAX_SCALE; // div by log(2) to use exp2f()
}

// support calling from device code without including cuda everywhere
#define CalcDotScaleAttention(dim) (sqrt(1. / dim) / LOG2 * ATT_DOTPRODUCT_SCALE)


///////////////////////////////////////////////////////////////////////////////////////////////////
// model params flags
const ui64 MPF_NOFLAGS = 0;
const ui64 MPF_HASHED_EMBED = 0x1;
const ui64 MPF_PPM = 0x2;
const ui64 MPF_USE_DOC_START_TOKEN = 0x4;
const ui64 MPF_TUNE_FINAL_LAYER = 0x8;
const ui64 MPF_TUNE_EMBED = 0x10;
const ui64 MPF_TAIL_LOSS = 0x20;
const ui64 MPF_SIM_QUANT_2BIT = 0x40;
const ui64 MPF_SIM_QUANT_4BIT = 0x80;
const ui64 MPF_GROK_BINARY_OP = 0x100;
const ui64 MPF_COMBINE_LAYERS = 0x200;
const ui64 MPF_MLM_BERT = 0x400;


///////////////////////////////////////////////////////////////////////////////////////////////////
constexpr yint ATT_ID_CREATE_WIDE_FLAG = 0x20000;
constexpr yint ATT_ID_USE_WIDE_FLAG = 0x10000;
constexpr yint ATT_ID_LAYER_MASK = 0xffff;


///////////////////////////////////////////////////////////////////////////////////////////////////
struct TModelDim
{
    struct TAttentionPosParams
    {
        float AlibiSlope = 0;
        float AlibiHyper = 0;
        yint AttentionWidthId = 0;

        TAttentionPosParams() {}
        TAttentionPosParams(float slope, float aHyper, yint id) : AlibiSlope(slope), AlibiHyper(aHyper), AttentionWidthId(id) {}
    };
    yint Dim = 0;
    yint QDim = 0;
    yint TTDim = 0;
    yint LabelCount = 0;
    yint VocabSize = 0;
    TVector<yint> AttentionWidthArr;
    TVector<TVector<TAttentionPosParams>> Layers;
    ui64 Flags = 0;
    ui64 DocStartToken = 0;
    yint FragLen = 0;
    SAVELOAD(Dim, QDim, TTDim, LabelCount, VocabSize, AttentionWidthArr, Layers, Flags, DocStartToken, FragLen);

    yint GetAttentionWidthCount() const
    {
        return YSize(AttentionWidthArr);
    }
    yint GetAttentionCount() const
    {
        yint res = 0;
        for (const TVector<TAttentionPosParams> &x : Layers) {
            res += YSize(x);
        }
        return res;
    }
    yint GetWideLimitWindow() const
    {
        yint res = 1;
        for (yint x : AttentionWidthArr) {
            res = Max<yint>(x, res);
        }
        return res;
    }
    bool HasFlag(ui64 f) const
    {
        return (Flags & f) != 0;
    }
    void SetDocStartToken(ui64 token)
    {
        Flags |= MPF_USE_DOC_START_TOKEN;
        DocStartToken = token;
    }
};

inline bool operator==(const TModelDim::TAttentionPosParams &a, const TModelDim::TAttentionPosParams &b)
{
    return a.AlibiSlope == b.AlibiSlope && a.AlibiHyper == b.AlibiHyper && a.AttentionWidthId == b.AttentionWidthId;
}

inline bool operator==(const TModelDim &a, const TModelDim &b)
{
    return
        a.Dim == b.Dim && a.QDim == b.QDim && a.TTDim == b.TTDim &&
        a.LabelCount == b.LabelCount && a.VocabSize == b.VocabSize &&
        a.AttentionWidthArr == b.AttentionWidthArr && a.Layers == b.Layers && a.Flags == b.Flags &&
        a.DocStartToken == b.DocStartToken &&
        a.FragLen == b.FragLen;
}


enum EAlibi
{
    ALIBI_NONE,
    ALIBI_V1,
    ALIBI_V2,
    ALIBI_V2_YOCO,
    ALIBI_V3,
};

enum ECombinerInit
{
    COMBINER_INIT_RANDOM,
    COMBINER_INIT_ZERO,
};


void InitModelDim(TModelDim *pRes, const TString &modelDimStr, EAlibi alibi, yint vocabSize, yint labelCount, ui64 flags);
TString GetModelDimsString(const TModelDim &modelDim);


///////////////////////////////////////////////////////////////////////////////////////////////////
yint CalcDropTableSize(const TModelDim &modelDim);
void MakeDropTable(TXRng &rng, const TModelDim &modelDim, TVector<ui32> *pDropTable, float channelDrop);
