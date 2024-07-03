#pragma once
#include "model_dim.h"
#include "model_matrix.h"
#include <gpt/rng/xrng.h>


// model matrices discretization step
//constexpr float MODEL_DISCR_SCALE = 1.f / 16;
//constexpr float MODEL_DISCR_SCALE = 1.f / 24;
constexpr float MODEL_DISCR_SCALE = 1.f / 32;


struct TModelParams
{
    struct TAttentionMatrices
    {
        TArray2D<float> QK; // query matrix low rank decomposition
        TArray2D<float> QV; //
        TArray2D<float> K; // 1 x Key (not a key from paper, this make argument to combiner from current vector)
        TArray2D<float> V; // 1 x Value
        TArray2D<float> Combiner; // block diagonal combiner, (ttDim * COMBINER_TILE) x (dim)
        SAVELOAD(QK, QV, K, V, Combiner);
    };
    TModelDim ModelDim;
    TModelMatrixRowDisp LabelEmbed;
    TVector<TVector<TAttentionMatrices>> LayerArr;
    TModelMatrixRowDisp FinalLayer;
    TVector<float> Bias;
    SAVELOAD(ModelDim, LabelEmbed, LayerArr, FinalLayer, Bias);

    TModelDim GetModelDim() const
    {
        Y_ASSERT(ModelDim.Dim == LabelEmbed.GetXSize());
        Y_ASSERT(ModelDim.LabelCount == LabelEmbed.GetYSize());
        Y_ASSERT(ModelDim.VocabSize == FinalLayer.GetYSize());
        Y_ASSERT(YSize(ModelDim.Layers) == YSize(LayerArr));
        for (yint d = 0; d < YSize(ModelDim.Layers); ++d) {
            Y_ASSERT(YSize(ModelDim.Layers[d]) == YSize(LayerArr[d]));
        }
        return ModelDim;
    }
    bool IsEmpty() const { return Bias.empty(); }
};


struct TModelParamsHolder : public TThrRefBase
{
    TModelParams Params;
};


///////////////////////////////////////////////////////////////////////////////////////////////////
void InitModel(TModelParams *pParams, TXRng &rng, const TModelDim &modelDim, ECombinerInit combinerInit, const TVector<float> &biasArr);

void AddScaled(TModelParams *dst, const TModelParams &arg, float scale, float rowDispScale);
void Scale(TModelParams *pParams, float scale, float rowDispScale);
yint CountModelSize(const TModelParams &params);
void Randomize(TXRng &rng, TModelParams *pParams);
double CalcDot(const TModelParams &params1, const TModelParams &params2);
double CalcSum2(const TModelParams &params);


///////////////////////////////////////////////////////////////////////////////////////////////////
void PackModelParams(TBufferedStream &f, TModelParams &params);
void UnpackModelParams(TModelParams *pParams, TBufferedStream &f);
void AddPackedModelParamsScaled(TModelParams *pParams, TBufferedStream &f, float scale, float rowDispScale);
