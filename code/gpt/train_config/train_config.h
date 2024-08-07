#pragma once
#include "train_step.h"


///////////////////////////////////////////////////////////////////////////////////////////////////
struct TTrainConfig
{
    yint BatchAccumulate = 1;
    yint TrainBatchSize = 64;
    yint TrainFragLen = 63;
    float TokenDrop = 0.9f;
    float ChannelDrop = 0.9f;
    TTrainingStep Step;
    float LRTail = 0;
    // SAVELOAD - serialized as POD

public:
    TTrainConfig() {}
    TTrainConfig(const TString &trainConfig, const TString &dropConfig);
    TString GetTrainConfig();
    TString GetDropConfig();

public:
    yint GetMaxNodeCount() const;
    bool DoAccumulate(yint iter) const
    {
        return (iter % BatchAccumulate) < BatchAccumulate - 1;
    }

    TTrainingStep GetStep(yint iter, yint maxIters) const
    {
        TTrainingStep res = Step;
        //if (iter < 100) {
        //    float scale = (iter + 1.) / 100;
        //    res.ScaleRate(scale);
        //}
        if (LRTail != 0) {
            float longFrac = (iter < maxIters) ? iter / (maxIters + 0.f) : 1;
            float scale = Min<float>(1, (1 - longFrac) * LRTail);
            res.ScaleRate(scale);
        }
        return res;
    }
};
