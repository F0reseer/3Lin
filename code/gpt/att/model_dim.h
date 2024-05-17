#pragma once
#include <gpt/data/xrng.h>

constexpr yint COMBINER_REP = 16;

constexpr yint GetCombinerWidth(yint ttDim)
{
    return ttDim * COMBINER_REP;
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// what scaling is optimal and why we need it all is unclear
const float FINAL_LAYER_SOFTMAX_SCALE = 4;


// model params flags
const ui64 MPF_NOFLAGS = 0;
const ui64 MPF_HASHED_EMBED = 0x1;
const ui64 MPF_PPM = 0x2;
const ui64 MPF_USE_DOC_START_TOKEN = 0x4;
const ui64 MPF_TUNE_FINAL_LAYER = 0x8;
const ui64 MPF_TUNE_EMBED = 0x10;
const ui64 MPF_TAIL_LOSS = 0x20;
const ui64 MPF_SIM_QUANT_2BIT = 0x40;
const ui64 MPF_GROK_BINARY_OP = 0x80;


struct TWindowSizeLimit
{
    yint Limit = 0;
    yint LimitWide = 0;

    TWindowSizeLimit() {}
    TWindowSizeLimit(yint limit, yint limitWide) : Limit(limit), LimitWide(limitWide) {}
};

inline bool operator==(const TWindowSizeLimit &a, const TWindowSizeLimit &b)
{
    return a.Limit == b.Limit && a.LimitWide == b.LimitWide;
}


struct TModelDim
{
    struct TAttentionPosParams
    {
        float AlibiSlope = 0;
        float AlibiHyper = 0;
        bool WideLayer = true;
    };
    yint Dim = 0;
    yint QDim = 0;
    yint TTDim = 0;
    yint LabelCount = 0;
    yint VocabSize = 0;
    TVector<TVector<TAttentionPosParams>> Layers;
    ui64 Flags = 0;
    ui64 DocStartToken = 0;
    TWindowSizeLimit Window;
    yint FragLen = 0;
    SAVELOAD(Dim, QDim, TTDim, LabelCount, VocabSize, Layers, Flags, DocStartToken, Window, FragLen);

    void CreateLayers(const TVector<yint> &attPerLayer)
    {
        yint depth = YSize(attPerLayer);
        Layers.resize(depth);
        for (yint d = 0; d < depth; ++d) {
            Layers[d].resize(attPerLayer[d]);
        }
    }
    yint GetAttentionCount() const
    {
        yint res = 0;
        for (const TVector<TAttentionPosParams> &x : Layers) {
            res += YSize(x);
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
    return a.AlibiSlope == b.AlibiSlope && a.AlibiHyper == b.AlibiHyper && a.WideLayer == b.WideLayer;
}

inline bool operator==(const TModelDim &a, const TModelDim &b)
{
    return
        a.Dim == b.Dim && a.QDim == b.QDim && a.TTDim == b.TTDim &&
        a.LabelCount == b.LabelCount && a.VocabSize == b.VocabSize &&
        a.Layers == b.Layers && a.Flags == b.Flags &&
        a.DocStartToken == b.DocStartToken && a.Window == b.Window && 
        a.FragLen == b.FragLen;
}

enum EAlibi
{
    ALIBI_NONE,
    ALIBI_V1,
};

enum ECombinerInit
{
    COMBINER_INIT_RANDOM,
    COMBINER_INIT_ZERO,
};

void InitAlibi(TModelDim *p, EAlibi alibi);

///////////////////////////////////////////////////////////////////////////////////////////////////
yint CalcDropTableSize(const TModelDim &modelDim);
void MakeDropTable(TXRng &rng, const TModelDim &modelDim, TVector<ui32> *pDropTable, float channelDrop);


///////////////////////////////////////////////////////////////////////////////////////////////////
// attention per layer configurations
inline void AttSquare(TVector<yint> *p, yint depth, yint attPerLayer)
{
    for (yint d = 0; d < depth; ++d) {
        p->push_back(attPerLayer);
    }
}

inline void AttPyramidArithmetic(TVector<yint> *p, yint depth)
{
    for (yint d = 1; d <= depth; ++d) {
        p->push_back(d);
    }
}

inline void AttPyramidGeometric(TVector<yint> *p, yint depth)
{
    for (yint d = 0; d < depth; ++d) {
        p->push_back(1ll << d);
    }
}

inline void AttRepeat(TVector<yint> *p, const TVector<yint> &vec, yint rep)
{
    for (yint k : vec) {
        for (yint i = 0; i < rep; ++i) {
            p->push_back(k);
        }
    }
}
