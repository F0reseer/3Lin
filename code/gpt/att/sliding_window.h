#pragma once
#include <gpt/rng/xrng.h>
#include "nodes_batch.h"
#include <gpt/model_params/model_dim.h>


void InitModelDim(TModelDim *pRes, const TString &modelDimStr, EAlibi alibi, yint vocabSize, ui64 flags);
yint GetNodeCount(yint len);

struct TFragment;


enum {
    ATT_GRAPH_TRAIN_LOSS,
    ATT_GRAPH_TEST_LOSS,
};

void InitLabelData(const TModelDim &modelDim, TXRng &rng, float tokenDrop,
    const TVector<TFragment> &fragArr, yint lossType,
    TNodesBatch *pNodes);

// process set of fragments and init context
template <class TComputeContext>
inline void MakeTrain(TXRng &rng, const TVector<TFragment> &fragArr,
    float tokenDrop, float channelDrop,
    TComputeContext *pCtx, yint deviceId, TVector<TNodeTarget> *pTarget)
{
    TNodesBatch &nodes = pCtx->GetNodes(deviceId);
    TVector<ui32> &dropTable = pCtx->GetDropTable(deviceId);
    TModelDim modelDim = pCtx->GetModelDim();
    InitLabelData(modelDim, rng, tokenDrop, fragArr, ATT_GRAPH_TRAIN_LOSS, &nodes);
    MakeDropTable(rng, modelDim, &dropTable, channelDrop);
    pCtx->Init(deviceId);
    if (pTarget) {
        *pTarget = nodes.Target;
    }
}


extern TXRng NopRng;

template <class TComputeContext>
void MakeTest(const TVector<TFragment> &fragArr, TComputeContext *pCtx, yint deviceId)
{
    TNodesBatch &nodes = pCtx->GetNodes(deviceId);
    TVector<ui32> &dropTable = pCtx->GetDropTable(deviceId);
    TModelDim modelDim = pCtx->GetModelDim();
    InitLabelData(modelDim, NopRng, 1., fragArr, ATT_GRAPH_TEST_LOSS, &nodes);
    dropTable.resize(0);
    dropTable.resize(CalcDropTableSize(modelDim), ~0);
    pCtx->Init(deviceId);
}


// call when not interested in target loss computation
template <class TComputeContext>
inline void MakeTrain(TXRng &rng, const TVector<TFragment> &fragArr, float tokenDrop, float channelDrop, TComputeContext *pCtx, yint deviceId)
{
    MakeTrain(rng, fragArr, tokenDrop, channelDrop, pCtx, deviceId, nullptr);
}
