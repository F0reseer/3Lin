#include "stdafx.h"
#include "model_params.h"
#include "sse_utils.h"
#include <lib/random/rand_utils.h>


///////////////////////////////////////////////////////////////////////////////////////////////////
// Model Params Initialize
template <class TMatrix>
void InitMatrixNormal(TMatrix *pRes, TXRng &rng, yint xSize, yint ySize, float sko)
{
    pRes->SetSizes(xSize, ySize);
    float bound = sko * 0.5f;
    for (yint y = 0; y < ySize; ++y) {
        for (yint x = 0; x < xSize; ++x) {
            //(*pRes)[y][x] = (rng.GenRandReal3() * 2 - 1) * bound;
            (*pRes)[y][x] = GenNormal(rng) * bound;
        }
    }
}

template <class TMatrix>
void InitEmbedMatrix(TMatrix *pRes, TXRng &rng, yint xSize, yint ySize)
{
    TArray2D<float> embed;
    embed.SetSizes(xSize, ySize);
    for (yint y = 0; y < ySize; ++y) {
        for (yint x = 0; x < xSize; ++x) {
            embed[y][x] = GenNormal(rng);
            //embed[y][x] = (rng.Uniform(2)) == 0 ? 1. : -1.; // marginally better?
        }
    }
    pRes->SetMatrix(embed);
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


static void InitAttention(TModelParams::TAttentionMatrices *p, TXRng &rng, yint dim, yint qDim, yint ttDim, ECombinerInit combinerInit)
{
    InitMatrixNormal(&p->QK, rng, dim, qDim, 1);
    InitMatrixNormal(&p->QV, rng, dim, qDim, 1);
    //InitIdentity(&p->QK, rng, dim, qDim);
    //InitIdentity(&p->QV, rng, dim, qDim);
    InitMatrixNormal(&p->K, rng, dim, ttDim, 1);
    InitMatrixNormal(&p->V, rng, dim, ttDim, 1);
    if (combinerInit == COMBINER_INIT_RANDOM) {
        InitMatrixNormal(&p->Combiner, rng, GetCombinerWidth(ttDim), dim, 1); // required for binary classification to converge to interesting solutions?
    } else if (combinerInit == COMBINER_INIT_ZERO) {
        p->Combiner.SetSizes(GetCombinerWidth(ttDim), dim);
        p->Combiner.FillZero();
    } else {
        Y_ASSERT(0 && "unsupported combiner init");
    }
}

void InitModel(TModelParams *pParams, TXRng &rng, const TModelDim &modelDim, ECombinerInit combinerInit, const TVector<float> &biasArr)
{
    TModelDim &dims = pParams->ModelDim;
    dims = modelDim;
    // init params
    InitEmbedMatrix(&pParams->LabelEmbed, rng, dims.Dim, dims.LabelCount);
    yint depth = YSize(dims.Layers);
    pParams->LayerArr.resize(depth);
    for (yint d = 0; d < depth; ++d) {
        yint attCount = YSize(dims.Layers[d]);
        pParams->LayerArr[d].resize(attCount);
        for (yint at = 0; at < attCount; ++at) {
            InitAttention(&pParams->LayerArr[d][at], rng, dims.Dim, dims.QDim, dims.TTDim, combinerInit);
        }
    }
    TArray2D<float> finalLayer;
    //InitMatrixNormal(&finalLayer, rng, dims.Dim, vocabSize, 1);
    //InitMatrixNormal(&finalLayer, rng, dims.Dim, dims.VocabSize, 4); // init for fixed final layer
    InitMatrixNormal(&finalLayer, rng, dims.Dim, dims.VocabSize, 1); // init for fixed final layer
    pParams->FinalLayer.SetMatrix(finalLayer);
    Y_VERIFY(YSize(biasArr) == dims.VocabSize);
    pParams->Bias = biasArr;
}


static void AllocateAttention(TModelParams::TAttentionMatrices *p, yint dim, yint qDim, yint ttDim)
{
    p->QK.SetSizes(dim, qDim);
    p->QV.SetSizes(dim, qDim);
    p->K.SetSizes(dim, ttDim);
    p->V.SetSizes(dim, ttDim);
    p->Combiner.SetSizes(GetCombinerWidth(ttDim), dim);
}

static void AllocateModel(TModelParams *pParams, const TModelDim &dims)
{
    pParams->ModelDim = dims;
    TArray2D<float> matr;
    matr.SetSizes(dims.Dim, dims.LabelCount);
    pParams->LabelEmbed.SetMatrix(matr);
    yint depth = YSize(dims.Layers);
    pParams->LayerArr.resize(depth);
    for (yint d = 0; d < depth; ++d) {
        yint attCount = YSize(dims.Layers[d]);
        pParams->LayerArr[d].resize(attCount);
        for (yint at = 0; at < attCount; ++at) {
            AllocateAttention(&pParams->LayerArr[d][at], dims.Dim, dims.QDim, dims.TTDim);
        }
    }
    matr.SetSizes(dims.Dim, dims.VocabSize);
    pParams->FinalLayer.SetMatrix(matr);
    pParams->Bias.resize(dims.VocabSize);
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
    pRes->push_back(&pParams->LabelEmbed.GetMatrix());
    for (yint d = 0; d < depth; ++d) {
        for (auto &ll : pParams->LayerArr[d]) {
            pRes->push_back(&ll.QK);
            pRes->push_back(&ll.QV);
            pRes->push_back(&ll.K);
            pRes->push_back(&ll.V);
            pRes->push_back(&ll.Combiner);
        }
    }
    pRes->push_back(&pParams->FinalLayer.GetMatrix());
}

template <class TMP, class TMatr>
static void GetRowDispMatrices(TMP *pParams, TVector<TMatr *> *pRes)
{
    pRes->push_back(&pParams->LabelEmbed);
    pRes->push_back(&pParams->FinalLayer);
}


void AddScaled(TModelParams *dst, const TModelParams &arg, float scale, float rowDispScale)
{
    Y_VERIFY(dst->ModelDim == arg.ModelDim);

    TVector<TArray2D<float> *> dstMatrices;
    GetParamMatrices(dst, &dstMatrices);
    TVector<const TArray2D<float> *> argMatrices;
    GetParamMatrices(&arg, &argMatrices);
    for (yint k = 0; k < YSize(dstMatrices); ++k) {
        AddScaled(dstMatrices[k], *argMatrices[k], scale);
    }

    TVector<TModelMatrixRowDisp *> rdDstMatrices;
    GetRowDispMatrices(dst, &rdDstMatrices);
    TVector<const TModelMatrixRowDisp *> rdArgMatrices;
    GetRowDispMatrices(&arg, &rdArgMatrices);
    for (yint k = 0; k < YSize(rdDstMatrices); ++k) {
        const TModelMatrixRowDisp *src = rdArgMatrices[k];
        rdDstMatrices[k]->AddRowDisp(src->GetRowDisp(), src->GetSumWeight(), rowDispScale);
    }
}


void Scale(TModelParams *pParams, float scale, float rowDispScale)
{
    TVector<TArray2D<float> *> allMatrices;
    GetParamMatrices(pParams, &allMatrices);
    if (scale != 1) {
        for (auto *mp : allMatrices) {
            for (yint y = 0; y < mp->GetYSize(); ++y) {
                for (yint x = 0; x < mp->GetXSize(); ++x) {
                    (*mp)[y][x] *= scale;
                }
            }
        }
    }
    TVector<TModelMatrixRowDisp *> rdMatrices;
    GetRowDispMatrices(pParams, &rdMatrices);
    for (auto *mp : rdMatrices) {
        mp->ScaleRowDisp(rowDispScale);
    }
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


///////////////////////////////////////////////////////////////////////////////////////////////////
// pack/unpack model params
void PackModelParams(TBufferedStream &f, TModelParams &params)
{
    // model dim
    WriteStruct(f, params.ModelDim);
    // bias
    WriteStruct(f, params.Bias);
    // model matrices
    TVector<ui16> row;
    TVector<const TArray2D<float> *> allMatrices;
    GetParamMatrices(&params, &allMatrices);
    for (const TArray2D<float> *p : allMatrices) {
        yint xSize = p->GetXSize();
        yint ySize = p->GetYSize();
        row.resize(xSize);
        for (yint y = 0; y < ySize; ++y) {
            float sum2 = HorizontalSum(CalcRowSum2(p->GetRow(y), xSize));
            float sko = sqrt(sum2 / xSize);
            f.Write(&sko, sizeof(sko));
            if (sko != 0) {
                __m256 mult = _mm256_set1_ps(1 / sko);
                ConvertToFp16(row.data(), p->GetRow(y), xSize, mult);
                f.Write(row.data(), xSize * sizeof(row[0]));
            }
        }
    }
    // row disp matrices
    TVector<const TModelMatrixRowDisp *> rdMatrices;
    GetRowDispMatrices(&params, &rdMatrices);
    for (const TModelMatrixRowDisp *p : rdMatrices) {
        const TVector<float> &rowDisp = p->GetRowDisp();
        f.Write(rowDisp.data(), YSize(rowDisp) * sizeof(rowDisp[0]));
        float sumWeight = p->GetSumWeight();
        f.Write(&sumWeight, sizeof(sumWeight));
    }
}


void UnpackModelParams(TModelParams *pParams, TBufferedStream &f)
{
    // model dim
    TModelDim modelDim;
    ReadStruct(f, modelDim);
    AllocateModel(pParams, modelDim);
    // bias
    ReadStruct(f, pParams->Bias);
    // model matrices
    TVector<ui16> row;
    TVector<TArray2D<float> *> allMatrices;
    GetParamMatrices(pParams, &allMatrices);
    for (TArray2D<float> *p : allMatrices) {
        yint xSize = p->GetXSize();
        yint ySize = p->GetYSize();
        row.resize(xSize);
        for (yint y = 0; y < ySize; ++y) {
            float discrScale = 0;
            f.Read(&discrScale, sizeof(discrScale));
            if (discrScale == 0) {
                for (yint x = 0; x < xSize; ++x) {
                    (*p)[y][x] = 0;
                }
            } else {
                __m256 mult = _mm256_set1_ps(discrScale);
                f.Read(row.data(), xSize * sizeof(row[0]));
                UnpackFp16Array(p->GetRow(y), row.data(), xSize, mult);
            }
        }
    }
    // row disp matrices
    TVector<TModelMatrixRowDisp *> rdMatrices;
    GetRowDispMatrices(pParams, &rdMatrices);
    for (TModelMatrixRowDisp *p : rdMatrices) {
        TVector<float> rowDisp;
        rowDisp.resize(p->GetYSize());
        f.Read(rowDisp.data(), YSize(rowDisp) * sizeof(rowDisp[0]));
        float sumWeight;
        f.Read(&sumWeight, sizeof(sumWeight));
        p->SetRowDisp(rowDisp, sumWeight);
    }
}


void AddPackedModelParamsScaled(TModelParams *pParams, TBufferedStream &f, float scale, float rowDispScale)
{
    // model dim
    TModelDim modelDim;
    ReadStruct(f, modelDim);
    Y_VERIFY(modelDim == pParams->ModelDim);
    // bias
    TVector<float> bias;
    ReadStruct(f, bias);
    Y_VERIFY(bias == pParams->Bias);
    // model matrices
    TVector<ui16> row;
    TVector<TArray2D<float> *> allMatrices;
    GetParamMatrices(pParams, &allMatrices);
    for (TArray2D<float> *p : allMatrices) {
        yint xSize = p->GetXSize();
        yint ySize = p->GetYSize();
        row.resize(xSize);
        for (yint y = 0; y < ySize; ++y) {
            float discrScale = 0;
            f.Read(&discrScale, sizeof(discrScale));
            if (discrScale != 0) {
                __m256 mult = _mm256_set1_ps(discrScale * scale);
                f.Read(row.data(), xSize * sizeof(row[0]));
                AddPackedFp16Array(p->GetRow(y), row.data(), xSize, mult);
            }
        }
    }
    // row disp matrices
    TVector<TModelMatrixRowDisp *> rdMatrices;
    GetRowDispMatrices(pParams, &rdMatrices);
    for (TModelMatrixRowDisp *p : rdMatrices) {
        TVector<float> rowDisp;
        rowDisp.resize(p->GetYSize());
        f.Read(rowDisp.data(), YSize(rowDisp) * sizeof(rowDisp[0]));
        float sumWeight;
        f.Read(&sumWeight, sizeof(sumWeight));
        if (scale > 0) {
            p->AddRowDisp(rowDisp, sumWeight, rowDispScale);
        }
    }
}
