#pragma once

struct TModelParams;
class TTrainContext;

namespace NNetTrain
{
void RunWorker(yint port);
void RunMaster(yint startIteration, yint deviceCount, const TVector<TString> &workerArr, const TTrainContext &trainCtx, const TModelParams &params);
}
