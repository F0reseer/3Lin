#pragma once
#include <gpt/att/nodes_batch.h>
#include "model_params.h"
#include "par_matrix.h"


// model matrices discretization step
//constexpr float DISCR_SCALE = 1.f / 16;
//constexpr float DISCR_SCALE = 1.f / 24;
constexpr float DISCR_SCALE = 1.f / 32;

constexpr int MAIN_DEVICE = 0;

using NCuda::TModelMatrix;
using NCuda::TCPUMatrixAdd;
using NCuda::TModelMatrixScale;
using NCuda::EModelMatrixQuant;

///////////////////////////////////////////////////////////////////////////////////////////////////
struct TAttentionParams : public TThrRefBase
{
    TIntrusivePtr<TModelMatrix> QK;
    TIntrusivePtr<TModelMatrix> QV;
    TIntrusivePtr<TModelMatrix> K;
    TIntrusivePtr<TModelMatrix> V;
    TIntrusivePtr<TModelMatrix> Combiner;
    float AlibiSlope = 0;
    float AlibiHyper = 0;
    bool WideLayer = true;

    TAttentionParams(TIntrusivePtr<TCPUMatrixAdd> cpuMatrixAdd, TIntrusivePtr<TModelMatrixScale> matrixScale,
        yint dim, yint qDim, yint ttDim, const TModelDim::TAttentionPosParams &layerParams,
        EModelMatrixQuant quant);
    void GetParams(TModelParams::TAttentionMatrices *p, TModelDim::TAttentionPosParams *pLayerParams);
    void SetParams(const TModelParams::TAttentionMatrices &att, const TModelDim::TAttentionPosParams &layerParams);
    void GetGradient(TModelParams::TAttentionMatrices *p, TModelDim::TAttentionPosParams *pLayerParams);
};


struct IModel : public TThrRefBase
{
    // assume no operations in fly
    virtual void GetParamsImpl(TModelParams *p) = 0;
    virtual void SetParamsImpl(const TModelParams &p) = 0;
    virtual void GetGradientImpl(TModelParams *p) = 0;
    // retrieve param storage
    virtual TModelDim GetModelDim() = 0;
    virtual TModelMatrix *GetLabelEmbed() = 0;
    virtual const TAttentionParams &GetAttention(yint d, yint k) = 0;
    virtual TModelMatrix *GetFinalLayer() = 0;
    virtual const TVector<float> &GetBias() = 0;
    virtual TModelMatrixScale *GetMatrixScale() = 0;
    //
    virtual yint GetDeviceCount() = 0;
    virtual void StartIteration(float step) = 0;
    virtual void WaitCompute() = 0;
    virtual void ResetIterCount() = 0;
};


///////////////////////////////////////////////////////////////////////////////////////////////////
struct IComputeContext : public TThrRefBase
{
    virtual yint GetDeviceCount() = 0;
    virtual TModelDim GetModelDim() = 0;
    virtual void GetParams(TModelParams *p) = 0;
    virtual void SetParams(const TModelParams &p) = 0;
    virtual void GetGradient(TModelParams *p) = 0;

    virtual TNodesBatch &GetNodes(yint deviceId) = 0;
    virtual TVector<ui32> &GetDropTable(yint deviceId) = 0;
    virtual void Init(yint deviceId) = 0;
    virtual void ComputeFinalStateVectors(TVector<TVector<float>> *pStateVectors) = 0;
    virtual void ComputeFragmentPredictions(TVector<TVector<float>> *pPrediction) = 0;
    virtual float ComputeScore() = 0;
    virtual void Backprop(float step) = 0;
};


struct IMMDeltaHookGen;
TIntrusivePtr<IModel> CreateModel(yint deviceCount, const TModelParams &params, IMMDeltaHookGen *deltaHookGen);

inline TIntrusivePtr<IModel> CreateModel(yint deviceCount, const TModelParams &params)
{
    return CreateModel(deviceCount, params, nullptr);
}
