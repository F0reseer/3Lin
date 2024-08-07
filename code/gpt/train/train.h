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
    TIntrusivePtr<IDataSource> Data;
    TTrainConfig Config;
    TVector<TVector<TFragment>> ScoreTrainBatches;
    TVector<TVector<TFragment>> ScoreTestBatches;
    bool SaveModel = false;
    yint MaxIters = 1000;
    yint EvalInterval = 100;

public:
    TTrainContext(TIntrusivePtr<IDataSource> data, const TTrainConfig &cfg, bool saveModel, yint maxIters, yint evalInterval)
        : Data(data), Config(cfg), SaveModel(saveModel), MaxIters(maxIters), EvalInterval(evalInterval)
    {
    }
    const TTrainConfig &GetConfig() const { return Config; }
    const TVector<TVector<TFragment>> &GetScoreTrainBatches() const { return ScoreTrainBatches; }
    const TVector<TVector<TFragment>> &GetScoreTestBatches() const { return ScoreTestBatches; }
    float GetCompression() const { return Data->GetStats().Compression; }
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
        if (Data->GetStats().HasTest) {
            yint chkRngSeed = 1313;
            for (yint k = 0; k < batchCount; ++k) {
                TVector<TFragment> &testBatch = *ScoreTestBatches.insert(ScoreTestBatches.end());
                Data->SampleFragments(IDataSource::TEST, chkRngSeed++, batchSize, len, &testBatch);
            }
        }
        yint chkRngSeed = 31313;
        for (yint k = 0; k < batchCount; ++k) {
            TVector<TFragment> &trainBatch = *ScoreTrainBatches.insert(ScoreTrainBatches.end());
            Data->SampleFragments(IDataSource::TRAIN, chkRngSeed++, batchSize, len, &trainBatch);
        }
    }

    void SampleTrainBatches(yint rngSeed, yint deviceCount, TVector<TVector<TFragment>> *pRes) const
    {
        TVector<TFragment> allFrags;
        yint count = Config.TrainBatchSize;
        Data->SampleFragments(IDataSource::TRAIN, rngSeed, count * deviceCount, Config.TrainFragLen, &allFrags);
        pRes->resize(deviceCount);
        for (yint deviceId = 0; deviceId < deviceCount; ++deviceId) {
            for (yint k = 0; k < count; ++k) {
                (*pRes)[deviceId].push_back(allFrags[deviceId * count + k]);
            }
        }
    }
};
