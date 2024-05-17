#include "stdafx.h"
#include "train.h"
#include <gpt/att/sliding_window.h>
#include <lib/config/config.h>


const double LOSS_SCALE = 1;
//const double LOSS_SCALE = 1 / log(0.5); // bpc, bits per character


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
        } else if (param.Name == "w") {
            Window.Limit = param.Value;
        } else if (param.Name == "ww") {
            Window.LimitWide = param.Value;
        } else if (param.Name == "slide") {
            slide = param.Value;
        }
    }
    TrainFragLen = Window.LimitWide * slide - 1;

    TStringParams dc(dropConfig);
    for (auto &param : dc.Params) {
        if (param.Name == "drop") {
            TokenDrop = param.Value;
        } else if (param.Name == "ch") {
            ChannelDrop = param.Value;
        } else if (param.Name == "lr") {
            LearningRate = param.Value;
        } else if (param.Name == "tail") {
            LRTail = param.Value;
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
    res += Sprintf("b%gww%g", TrainBatchSize * 1., Window.LimitWide * 1.);
    if (Window.Limit != rr.Window.Limit) {
        res += Sprintf("w%g", Window.Limit * 1.);
    }
    double slide = (TrainFragLen + 1.) / Window.LimitWide;
    if (slide != 1) {
        res += Sprintf("slide%g", slide);
    }
    return res;
}

TString TTrainConfig::GetDropConfig()
{
    TTrainConfig rr;
    TString res = Sprintf("drop%gch%g", TokenDrop, ChannelDrop);
    if (LearningRate != rr.LearningRate) {
        res += Sprintf("lr%g", LearningRate);
    }
    if (LRTail != rr.LRTail) {
        res += Sprintf("tail%g", LRTail);
    }
    return res;
}



///////////////////////////////////////////////////////////////////////////////////////////////////
// 
double CalcModelErr(const TVector<TFragment> &fragArr, const TWindowSizeLimit &window, IComputeContext *pCtx)
{
    if (fragArr.empty()) {
        return 0;
    }
    MakeTest(window, fragArr, pCtx, MAIN_DEVICE);
    return pCtx->ComputeScore();
}


double CalcModelErr(const TVector<TVector<TFragment>> &batchArr, const TWindowSizeLimit &window, IComputeContext *pCtx)
{
    if (batchArr.empty()) {
        return 0;
    }
    double err = 0;
    for (const TVector<TFragment> &b : batchArr) {
        err += CalcModelErr(b, window, pCtx);
    }
    return err / YSize(batchArr);
}


double CalcTargetLoss(const TVector<TVector<float>> &predArr, const TVector<TNodeTarget> &target)
{
    double res = 0;
    for (const TNodeTarget &nt : target) {
        res += log(predArr[nt.Node][nt.TargetId]);
    }
    return res;
}


