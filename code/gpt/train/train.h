#pragma once
#include <gpt/data/bpe.h>
#include <gpt/data/data.h>
#include <gpt/compute/model.h>
#include <gpt/train_config/train_config.h>
#include <lib/hp_timer/hp_timer.h>
#include <lib/random/mersenne.h>
#include <lib/random/rand_utils.h>


double CalcModelErr(const TVector<TFragment> &fragArr, IComputeContext *pCtx);
double CalcModelErr(const TVector<TVector<TFragment>> &batchArr, IComputeContext *pCtx);
double CalcTargetLoss(const TVector<TVector<float>> &predArr, const TVector<TNodeTarget> &target);


///////////////////////////////////////////////////////////////////////////////////////////////////
class TTrainContext
{
    TDataset &Data;
    TTrainConfig Config;
    TVector<TVector<TFragment>> ScoreTrainBatches;
    TVector<TVector<TFragment>> ScoreTestBatches;
    bool SaveModel = false;
    yint MaxIters = 1000;
    yint EvalInterval = 100;

public:
    TTrainContext(TDataset &data, const TTrainConfig &cfg, bool saveModel, yint maxIters, yint evalInterval)
        : Data(data), Config(cfg), SaveModel(saveModel), MaxIters(maxIters), EvalInterval(evalInterval)
    {
    }
    TDataset &GetData() { return Data; }
    const TTrainConfig &GetConfig() const { return Config; }
    const TVector<TVector<TFragment>> &GetScoreTrainBatches() const { return ScoreTrainBatches; }
    const TVector<TVector<TFragment>> &GetScoreTestBatches() const { return ScoreTestBatches; }
    float GetCompression() const { return Data.GetCompression(); }
    bool IsSaveModel() const { return SaveModel; }
    yint GetMaxNodeCount() const { return Config.GetMaxNodeCount(); }
    yint GetMaxIters() const { return MaxIters; }
    yint GetEvalInterval() const { return EvalInterval; }
    TTrainingStep GetStep(yint iter) const
    {
        return Config.GetStep(iter, MaxIters);
    }

    void MakeScoreBatches(yint batchCount, yint batchSize, yint len)
    {
        TXRng chkRng(1313);
        if (Data.HasTest()) {
            for (yint k = 0; k < batchCount; ++k) {
                TVector<TFragment> &testBatch = *ScoreTestBatches.insert(ScoreTestBatches.end());
                for (yint k = 0; k < batchSize; ++k) {
                    TFragment frag;
                    Data.MakeFragment(TDataset::TEST, chkRng, len, &frag);
                    testBatch.push_back(frag);
                }
            }
        }
        for (yint k = 0; k < batchCount; ++k) {
            TVector<TFragment> &trainBatch = *ScoreTrainBatches.insert(ScoreTrainBatches.end());
            for (yint k = 0; k < batchSize; ++k) {
                TFragment frag;
                Data.MakeFragment(TDataset::TRAIN, chkRng, len, &frag);
                trainBatch.push_back(frag);
            }
        }
    }

    void MakeTrainBatches(TXRng &rng, TVector<TFragment> *pRes) const
    {
        TVector<TFragment> &fragArr = *pRes;
        for (yint k = 0; k < Config.TrainBatchSize; ++k) {
            yint len = Config.TrainFragLen;
            TFragment frag;
            Data.MakeFragment(TDataset::TRAIN, rng, len, &frag);
            fragArr.push_back(frag);
        }
    }
};
