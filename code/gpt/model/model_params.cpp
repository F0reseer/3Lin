#include "stdafx.h"
#include "model_params.h"
#include <lib/random/rand_utils.h>
#include <lib/config/config.h>


///////////////////////////////////////////////////////////////////////////////////////////////////
// Model Params Initialize
template <class TMatrix>
void InitMatrixNormal(TMatrix *pRes, TXRng &rng, yint xSize, yint ySize, float sko)
{
    pRes->SetSizes(xSize, ySize);
    // match pytorch Linear initialization (I hope)
    float bound = sqrt(1. / xSize) * sko;
    for (yint y = 0; y < ySize; ++y) {
        for (yint x = 0; x < xSize; ++x) {
            //(*pRes)[y][x] = (rng.GenRandReal3() * 2 - 1) * bound;
            (*pRes)[y][x] = GenNormal(rng) * bound;
        }
    }
}
//template <class TMatrix>
//void InitLinearMatrix(TMatrix *pRes, TRng &rng, yint xSize, yint ySize)
//{
//    float bound = sqrt(1. / xSize); // uniform [-bound;bound] in original?
//    InitMatrixNormal(pRes, rng, xSize, ySize, bound);
//}

template <class TMatrix>
void InitEmbedMatrix(TMatrix *pRes, TXRng &rng, yint xSize, yint ySize)
{
    pRes->SetSizes(xSize, ySize);
    for (yint y = 0; y < ySize; ++y) {
        for (yint x = 0; x < xSize; ++x) {
            (*pRes)[y][x] = GenNormal(rng);
            //(*pRes)[y][x] = (rng.Uniform(2)) == 0 ? 1. : -1.; // marginally better?
        }
    }
}

template <class TMatrix>
void InitIdentity(TMatrix *pRes, TXRng &rng, yint xSize, yint ySize)
{
    pRes->SetSizes(xSize, ySize);
    for (yint y = 0; y < ySize; ++y) {
        for (yint x = 0; x < xSize; ++x) {
            (*pRes)[y][x] = (x == y) ? 1 : 0;
        }
    }
}


static void InitAttention(TModelParams::TAttentionMatrices *p, TXRng &rng, yint dim, yint qDim, yint ttDim, const TModelDim::TAttentionPosParams &layerParams, ECombinerInit combinerInit)
{
    InitMatrixNormal(&p->QK, rng, dim, qDim, 1);
    InitMatrixNormal(&p->QV, rng, dim, qDim, 1);
    //InitIdentity(&p->QK, rng, dim, qDim);
    //InitIdentity(&p->QV, rng, dim, qDim);
    InitMatrixNormal(&p->K, rng, dim, ttDim, 1);
    InitMatrixNormal(&p->V, rng, dim, ttDim, 1);
    if (combinerInit == COMBINER_INIT_RANDOM) {
        InitMatrixNormal(&p->Combiner, rng, GetCombinerWidth(ttDim), dim, sqrt(1. * ttDim)); // required for binary classification to converge to interesting solutions?
    } else if (combinerInit == COMBINER_INIT_ZERO) {
        p->Combiner.SetSizes(GetCombinerWidth(ttDim), dim);
        p->Combiner.FillZero();
    } else {
        Y_ASSERT(0 && "unsupported combiner init");
    }
}

void InitModel(TModelParams *pParams, TXRng &rng,
    const TString &modelDims, yint vocabSize, yint labelCount,
    EAlibi alibi, ECombinerInit combinerInit,
    const TVector<float> &biasArr, const TVector<yint> &attPerLayer, ui64 flags)
{
    yint dim = 256;
    yint qDim = 128;
    yint ttDim = 64;
    TStringParams sp(modelDims);
    for (TStringParams::TParam &param : sp.Params) {
        if (param.Name == "e") {
            dim = param.Value;
        } else if (param.Name == "q") {
            qDim = param.Value;
        } else if (param.Name == "tt") {
            ttDim = param.Value;
        }
    }

    TModelDim &modelDim = pParams->ModelDim;
    modelDim.Dim = dim;
    modelDim.QDim = qDim;
    modelDim.TTDim = ttDim;
    modelDim.LabelCount = labelCount;
    modelDim.VocabSize = vocabSize;
    modelDim.CreateLayers(attPerLayer);
    modelDim.Flags = flags;
    InitAlibi(&modelDim, alibi);
    // init params
    InitEmbedMatrix(&pParams->LabelEmbed, rng, dim, modelDim.LabelCount);
    yint depth = YSize(modelDim.Layers);
    pParams->LayerArr.resize(depth);
    for (yint d = 0; d < depth; ++d) {
        yint attCount = YSize(modelDim.Layers[d]);
        pParams->LayerArr[d].resize(attCount);
        for (yint at = 0; at < attCount; ++at) {
            InitAttention(&pParams->LayerArr[d][at], rng, dim, qDim, ttDim, modelDim.Layers[d][at], combinerInit);
        }
    }
    //InitMatrixNormal(&pParams->FinalLayer, rng, dim, vocabSize, 1);
    InitMatrixNormal(&pParams->FinalLayer, rng, dim, vocabSize, 4); // init for fixed final layer
    Y_VERIFY(YSize(biasArr) == vocabSize);
    pParams->Bias = biasArr;
}


///////////////////////////////////////////////////////////////////////////////////////////////////
void ChopModel(TModelParams *pParams, yint depth)
{
    Y_ASSERT(depth >= 0 && depth <= YSize(pParams->LayerArr));
    TModelDim &modelDim = pParams->ModelDim;
    Y_ASSERT(YSize(modelDim.Layers) >= depth);
    modelDim.Layers.resize(depth);
    pParams->LayerArr.resize(depth);
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// model ops
static void AddScaled(TArray2D<float> *dst, const TArray2D<float> &arg, float scale)
{
    yint xSize = dst->GetXSize();
    yint ySize = dst->GetYSize();
    Y_ASSERT(arg.GetXSize() == xSize);
    Y_ASSERT(arg.GetYSize() == ySize);
    for (yint y = 0; y < ySize; ++y) {
        for (yint x = 0; x < xSize; ++x) {
            (*dst)[y][x] += arg[y][x] * scale;
        }
    }
}


// enum all model matrices
// ignores bias
template <class TMP, class TMatr>
static void GetParamMatrices(TMP *pParams, TVector<TMatr *> *pRes)
{
    yint depth = YSize(pParams->LayerArr);
    pRes->push_back(&pParams->LabelEmbed.Matr);
    for (yint d = 0; d < depth; ++d) {
        for (auto &ll : pParams->LayerArr[d]) {
            pRes->push_back(&ll.QK);
            pRes->push_back(&ll.QV);
            pRes->push_back(&ll.K);
            pRes->push_back(&ll.V);
            pRes->push_back(&ll.Combiner);
        }
    }
    pRes->push_back(&pParams->FinalLayer.Matr);
}


void AddScaled(TModelParams *dst, const TModelParams &arg, float scale)
{
    TVector<TArray2D<float> *> dstMatrices;
    GetParamMatrices(dst, &dstMatrices);

    TVector<const TArray2D<float> *> argMatrices;
    GetParamMatrices(&arg, &argMatrices);

    Y_ASSERT(YSize(dstMatrices) == YSize(argMatrices));
    for (yint k = 0; k < YSize(dstMatrices); ++k) {
        AddScaled(dstMatrices[k], *argMatrices[k], scale);
    }
    dst->LabelEmbed.ResetDisp();
    dst->FinalLayer.ResetDisp();
}


yint CountModelSize(const TModelParams &params)
{
    TVector<const TArray2D<float> *> allMatrices;
    GetParamMatrices(&params, &allMatrices);
    yint res = YSize(params.Bias);
    for (auto *mp : allMatrices) {
        res += mp->GetXSize() * mp->GetYSize();
    }
    return res;
}


void Randomize(TXRng &rng, TModelParams *pParams)
{
    TVector<TArray2D<float> *> allMatrices;
    GetParamMatrices(pParams, &allMatrices);
    for (auto *mp : allMatrices) {
        for (yint y = 0; y < mp->GetYSize(); ++y) {
            for (yint x = 0; x < mp->GetXSize(); ++x) {
                (*mp)[y][x] *= -log(rng.GenRandReal3());
            }
        }
    }
    pParams->LabelEmbed.ResetDisp();
    pParams->FinalLayer.ResetDisp();
}


double CalcDot(const TModelParams &params1, const TModelParams &params2)
{
    TVector<const TArray2D<float> *> am1;
    GetParamMatrices(&params1, &am1);
    TVector<const TArray2D<float> *> am2;
    GetParamMatrices(&params2, &am2);
    Y_ASSERT(YSize(am1) == YSize(am2));
    double res = 0;
    for (yint k = 0; k < YSize(am1); ++k) {
        auto *mp1 = am1[k];
        auto *mp2 = am2[k];
        for (yint y = 0; y < mp1->GetYSize(); ++y) {
            for (yint x = 0; x < mp1->GetXSize(); ++x) {
                res += (*mp1)[y][x] * (*mp2)[y][x];
            }
        }
    }
    return res;
}


double CalcSum2(const TModelParams &params)
{
    return CalcDot(params, params);
}

void Scale(TModelParams *pParams, double scale)
{
    TVector<TArray2D<float> *> allMatrices;
    GetParamMatrices(pParams, &allMatrices);
    for (auto *mp : allMatrices) {
        for (yint y = 0; y < mp->GetYSize(); ++y) {
            for (yint x = 0; x < mp->GetXSize(); ++x) {
                (*mp)[y][x] *= scale;
            }
        }
    }
}
