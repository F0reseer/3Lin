#include "stdafx.h"
#include "train_config.h"
#include <lib/config/config.h>


///////////////////////////////////////////////////////////////////////////////////////////////////
TTrainConfig::TTrainConfig(const TString &trainConfig, const TString &dropConfig)
{
    double slide = 1;
    TStringParams tc(trainConfig);
    for (auto &param : tc.Params) {
        if (param.Name == "a") {
            BatchAccumulate = param.Value;
        } else if (param.Name == "b") {
            TrainBatchSize = param.Value;
        } else if (param.Name == "f") {
            TrainFragLen = param.Value - 1;
        } else {
            Y_VERIFY(0 && "unknown param");
        }
    }

    TStringParams dc(dropConfig);
    for (auto &param : dc.Params) {
        if (param.Name == "drop") {
            TokenDrop = param.Value;
        } else if (param.Name == "ch") {
            ChannelDrop = param.Value;
        } else if (param.Name == "lr") {
            Step.Rate = param.Value;
        } else if (param.Name == "reg") {
            // ballpark estimate of optimal reg:
            //   reg = train_size / batch_size * 0.1
            // learning rate does not change optimal reg
            // dropout does not seem to change optimal reg
            // channel dropout does seem to change optimal reg, however ch1 wins
            if (param.Value > 0) {
                Step.L2Reg = 1 / param.Value;
            } else {
                Step.L2Reg = 0;
            }
        } else if (param.Name == "tail") {
            LRTail = param.Value;
        } else {
            Y_VERIFY(0 && "unknown param");
        }
    }
}

yint TTrainConfig::GetMaxNodeCount() const
{
    return (TrainFragLen + 1) * TrainBatchSize;
}

TString TTrainConfig::GetTrainConfig()
{
    TTrainConfig rr;
    TString res;
    if (BatchAccumulate != 1) {
        res += Sprintf("a%g", BatchAccumulate * 1.);
    }
    res += Sprintf("b%gf%g", TrainBatchSize * 1., TrainFragLen + 1.);
    return res;
}

TString TTrainConfig::GetDropConfig()
{
    TTrainConfig rr;
    TString res = Sprintf("drop%gch%g", TokenDrop, ChannelDrop);
    if (Step.Rate != rr.Step.Rate) {
        res += Sprintf("lr%g", Step.Rate);
    }
    if (Step.L2Reg != rr.Step.L2Reg) {
        res += Sprintf("reg%g", 1 / Step.L2Reg);
    }
    if (LRTail != rr.LRTail) {
        res += Sprintf("tail%g", LRTail);
    }
    return res;
}
