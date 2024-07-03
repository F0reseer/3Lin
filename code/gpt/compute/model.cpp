#include "stdafx.h"
#include "model.h"
#include <lib/cuda/cuda_arrays.h>
#include "par_matrix.h"

using namespace NCuda;


///////////////////////////////////////////////////////////////////////////////////////////////////
TAttentionParams::TAttentionParams(TIntrusivePtr<TCPUMatrixAdd> cpuMatrixAdd, TIntrusivePtr<TModelMatrixScale> matrixScale,
    yint dim, yint qDim, yint ttDim, const TModelDim::TAttentionPosParams &layerParams,
    EModelMatrixQuant quant)
{
    QK = CreateModelMatrix(cpuMatrixAdd, matrixScale, MODEL_DISCR_SCALE, dim, qDim, MM_DISP_MATRIX, quant, MM_SYNC_GRADIENT);
    QV = CreateModelMatrix(cpuMatrixAdd, matrixScale, MODEL_DISCR_SCALE, dim, qDim, MM_DISP_MATRIX, quant, MM_SYNC_GRADIENT);
    K = CreateModelMatrix(cpuMatrixAdd, matrixScale, MODEL_DISCR_SCALE, dim, ttDim, MM_DISP_MATRIX, quant, MM_SYNC_GRADIENT);
    V = CreateModelMatrix(cpuMatrixAdd, matrixScale, MODEL_DISCR_SCALE, dim, ttDim, MM_DISP_MATRIX, quant, MM_SYNC_GRADIENT);
    Combiner = CreateModelMatrix(cpuMatrixAdd, matrixScale, MODEL_DISCR_SCALE, GetCombinerWidth(ttDim), dim, MM_DISP_MATRIX, quant, MM_SYNC_GRADIENT);
    AlibiSlope = layerParams.AlibiSlope;
    AlibiHyper = layerParams.AlibiHyper;
    AttentionWidthId = layerParams.AttentionWidthId;
}

void TAttentionParams::GetParams(TModelParams::TAttentionMatrices *p, TModelDim::TAttentionPosParams *pLayerParams)
{
    QK->GetData(&p->QK);
    QV->GetData(&p->QV);
    K->GetData(&p->K);
    V->GetData(&p->V);
    Combiner->GetData(&p->Combiner);
    pLayerParams->AlibiSlope = AlibiSlope;
    pLayerParams->AlibiHyper = AlibiHyper;
    pLayerParams->AttentionWidthId = AttentionWidthId;
}

void TAttentionParams::SetParams(const TModelParams::TAttentionMatrices &att, const TModelDim::TAttentionPosParams &layerParams)
{
    QK->SetData(att.QK);
    QV->SetData(att.QV);
    K->SetData(att.K);
    V->SetData(att.V);
    Combiner->SetData(att.Combiner);
    AlibiSlope = layerParams.AlibiSlope;
    AlibiHyper = layerParams.AlibiHyper;
    AttentionWidthId = layerParams.AttentionWidthId;
}

void TAttentionParams::GetGradient(TModelParams::TAttentionMatrices *p, TModelDim::TAttentionPosParams *pLayerParams)
{
    QK->GetDeltaData(&p->QK);
    QV->GetDeltaData(&p->QV);
    K->GetDeltaData(&p->K);
    V->GetDeltaData(&p->V);
    Combiner->GetDeltaData(&p->Combiner);
    pLayerParams->AlibiSlope = AlibiSlope;
    pLayerParams->AlibiHyper = AlibiHyper;
    pLayerParams->AttentionWidthId = AttentionWidthId;
}


///////////////////////////////////////////////////////////////////////////////////////////////////
class TModel : public IModel
{
    TModelDim ModelDim;
    TIntrusivePtr<TCPUMatrixAdd> MatrixAdd;

    TIntrusivePtr<TModelMatrixScale> MatrixScale;
    TIntrusivePtr<TModelMatrix> LabelEmbed;
    TVector<TVector<TIntrusivePtr<TAttentionParams>>> LayerArr;
    TIntrusivePtr<TModelMatrix> FinalLayer;
    TVector<float> Bias;
    TIntrusivePtr<IMMDeltaHookGen> DeltaHookGen;


    void Allocate(yint deviceCount, const TModelDim &md)
    {
        ModelDim = md;
        yint attentionCount = md.GetAttentionCount();
        const yint MATRIX_PER_ATTENTION = 5;
        const yint MATRIX_PER_MODEL = 2;
        yint maxMatrixCount = DivCeil(attentionCount * MATRIX_PER_ATTENTION + MATRIX_PER_MODEL, 32) * 32;

        EModelMatrixQuant quant = MM_QUANT_NONE;
        if (ModelDim.HasFlag(MPF_SIM_QUANT_2BIT)) {
            quant = MM_QUANT_2BIT;
        } else if (ModelDim.HasFlag(MPF_SIM_QUANT_4BIT)) {
            quant = MM_QUANT_4BIT;
        }

        MatrixScale = new TModelMatrixScale(maxMatrixCount);
        MatrixAdd = new TCPUMatrixAdd(deviceCount, maxMatrixCount, DeltaHookGen.Get());
        FinalLayer = CreateModelMatrix(MatrixAdd, MatrixScale, MODEL_DISCR_SCALE, md.Dim, md.VocabSize, MM_DISP_ROW, MM_QUANT_NONE, MM_SYNC_GRADIENT);
        LabelEmbed = CreateModelMatrix(MatrixAdd, MatrixScale, MODEL_DISCR_SCALE, md.Dim, md.LabelCount, MM_DISP_ROW, MM_QUANT_NONE, MM_STALE_GRADIENT);
        LayerArr.resize(YSize(md.Layers));
        for (yint d = 0; d < YSize(md.Layers); ++d) {
            for (yint at = 0; at < YSize(md.Layers[d]); ++at) {
                LayerArr[d].push_back(new TAttentionParams(MatrixAdd, MatrixScale, md.Dim, md.QDim, md.TTDim, md.Layers[d][at], quant));
            }
        }
        ClearPodArray(&Bias, md.VocabSize);
        MatrixAdd->LaunchWorkers();
    }

public:
    TModel(yint deviceCount, const TModelParams &modelParams, IMMDeltaHookGen *deltaHookGen)
        : DeltaHookGen(deltaHookGen)
    {
        Allocate(deviceCount, modelParams.ModelDim);
        SetParamsImpl(modelParams);
    }

    void GetParamsImpl(TModelParams *p) override
    {
        p->ModelDim = ModelDim;
        LabelEmbed->GetData(&p->LabelEmbed);
        yint depth = YSize(LayerArr);
        p->LayerArr.resize(depth);
        for (yint d = 0; d < depth; ++d) {
            yint count = YSize(LayerArr[d]);
            p->LayerArr[d].resize(count);
            for (yint k = 0; k < count; ++k) {
                LayerArr[d][k]->GetParams(&p->LayerArr[d][k], &p->ModelDim.Layers[d][k]);
            }
        }
        FinalLayer->GetData(&p->FinalLayer);
        p->Bias = Bias;
    }

    void SetParamsImpl(const TModelParams &p) override
    {
        Y_VERIFY(ModelDim == p.GetModelDim());
        LabelEmbed->SetData(p.LabelEmbed);
        yint depth = YSize(p.LayerArr);
        LayerArr.resize(depth);
        for (yint d = 0; d < depth; ++d) {
            yint count = YSize(p.LayerArr[d]);
            LayerArr[d].resize(count);
            for (yint k = 0; k < count; ++k) {
                LayerArr[d][k]->SetParams(p.LayerArr[d][k], p.ModelDim.Layers[d][k]);
            }
        }
        FinalLayer->SetData(p.FinalLayer);
        Bias = p.Bias;
        //NeedCopyToDevice = true;
    }

    void GetGradientImpl(TModelParams *p) override
    {
        p->ModelDim = ModelDim;
        LabelEmbed->GetDeltaData(&p->LabelEmbed);
        yint depth = YSize(LayerArr);
        p->LayerArr.resize(depth);
        for (yint d = 0; d < depth; ++d) {
            yint count = YSize(LayerArr[d]);
            p->LayerArr[d].resize(count);
            for (yint k = 0; k < count; ++k) {
                LayerArr[d][k]->GetGradient(&p->LayerArr[d][k], &p->ModelDim.Layers[d][k]);
            }
        }
        FinalLayer->GetDeltaData(&p->FinalLayer);
        ClearPodArray(&p->Bias, YSize(Bias));
    }

    TModelDim GetModelDim() override
    {
        return ModelDim;
    }

    TModelMatrix *GetLabelEmbed() override
    {
        return LabelEmbed.Get();
    }

    const TAttentionParams &GetAttention(yint d, yint k) override
    {
        return *LayerArr[d][k];
    }

    TModelMatrix *GetFinalLayer() override
    {
        return FinalLayer.Get();
    }

    const TVector<float> &GetBias() override
    {
        return Bias;
    }

    TModelMatrixScale *GetMatrixScale() override
    {
        return MatrixScale.Get();
    }

    yint GetDeviceCount() override
    {
        return MatrixAdd->GetDeviceCount();
    }

    void StartIteration(const TTrainingStep &step, EAddToModel addToModel) override
    {
        MatrixAdd->StartIteration(step, addToModel);
    }

    void WaitCompute() override
    {
        MatrixAdd->Wait();
    }

    void ResetIterCount() override
    {
        MatrixAdd->ResetIterCount();
    }
};



TIntrusivePtr<IModel> CreateModel(yint deviceCount, const TModelParams &params, IMMDeltaHookGen *deltaHookGen)
{
    return new TModel(deviceCount, params, deltaHookGen);
}
