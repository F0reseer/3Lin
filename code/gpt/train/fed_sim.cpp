#include "stdafx.h"
#include "train.h"
#include "fed_sim.h"
#include <gpt/att/sliding_window.h>
#include <gpt/data/data.h>
#include <gpt/compute/gpt_cuda.cuh>


namespace NFedSim
{

const bool SAVE_MODEL = false;

#ifdef NDEBUG
const yint EVAL_INTERVAL = 100;
const yint EVAL_BATCH_COUNT = 20;
//const yint MAX_ITERS = 300; // meta iter size
const yint MAX_ITERS = 1000; // meta iter size
#else
const yint EVAL_INTERVAL = 1;
const yint EVAL_BATCH_COUNT = 2;
const yint MAX_ITERS = 10; // meta iter size
#endif


struct TCommonContext
{
    TModelParams Params;
    TTokenizer Tokenizer;
};

struct TAgentContext : public TThrRefBase
{
    TDataset Data;
    TModelParams Tune;
    TTrainContext TrainCtx;
    //TOFStream Log;

    TAgentContext(const TTrainConfig &cfg, const TString &logName)
        : TrainCtx(Data, cfg, SAVE_MODEL, MAX_ITERS, EVAL_INTERVAL)
        //, Log(logName.c_str())
    {
    }
};


static void CreateCommon(TCommonContext *p)
{
    TXRng rng(1313);
    TString modelDimsString = "e256d65";
    ui64 modelFlags = MPF_TUNE_FINAL_LAYER | MPF_TUNE_EMBED;

    Serialize(true, "D:/tokenizers/5k.bin", p->Tokenizer);
    yint vocabSize = p->Tokenizer.GetVocabSize();

    // initialize model params
    TVector<float> bias;
    ClearPodArray(&bias, vocabSize);

    TModelDim modelDim;
    InitModelDim(&modelDim, modelDimsString, ALIBI_V3, vocabSize, modelFlags);
    p->Params = TModelParams();
    InitModel(&p->Params, rng, modelDim, COMBINER_INIT_ZERO, bias);
    Y_ASSERT(p->Tokenizer.HasDocStartToken());
    p->Params.ModelDim.SetDocStartToken(p->Tokenizer.GetDocStartToken());
}


static TIntrusivePtr<TAgentContext> CreateAgent(const TTrainConfig &tc, const TCommonContext &common, const TVector<TVector<char>> &docSet, const TString &logName)
{
    bool usePPM = false;
    float weight = 1;
    float testFraction = 0.05f;

    TIntrusivePtr<TAgentContext> p = new TAgentContext(tc, logName);
    // params
    p->Tune = common.Params;
    Scale(&p->Tune, 0, 0);

    // load data
    AddDocset(new TDatasetBuilder(&p->Data, usePPM, common.Tokenizer), common.Tokenizer, docSet, weight, testFraction);

    // create batches for train & test score compute, can use different sizes
    const yint batchSize = tc.TrainBatchSize;
    const yint fragLen = tc.TrainFragLen;
    p->TrainCtx.MakeScoreBatches(EVAL_BATCH_COUNT, batchSize, fragLen);

    return p;
}


///////////////////////////////////////////////////////////////////////////////////////////////////
struct TFedTrain
{
    TCommonContext Common;
    TVector<TIntrusivePtr<TAgentContext>> AgentArr;

    TFedTrain()
    {
        CreateCommon(&Common);
    }

    void AddAgent(const TTrainConfig &tc, const TString &binFileName, const TString &logName)
    {
        TVector<TVector<char>> docSet;
        LoadDocumentSetFromBin(&docSet, binFileName);
        AgentArr.push_back(CreateAgent(tc, Common, docSet, logName));
    }
};


void Train(TFedTrain *p)
{
    TFedTrain &fed = *p;

    yint deviceCount = 1;
    yint maxNodeCount = fed.AgentArr[0]->TrainCtx.GetMaxNodeCount();
    yint agentCount = YSize(fed.AgentArr);

    // create model
    TIntrusivePtr<IModel> pModel = CreateModel(deviceCount, fed.Common.Params);
    TIntrusivePtr<IComputeContext> pCtx = NCUDA_GPT::CreateContext(pModel, maxNodeCount);

    TVector<TModelParams> paramTrackArr;
    TVector<TModelParams> avrgParamTrackArr;
    avrgParamTrackArr.push_back(fed.Common.Params);
    //TOFStream log("d:/fed_log.txt");

    bool UPDATE_LAG = true;
    //bool UPDATE_LAG = false;

    //bool USE_TRACK_AVRG = true;
    bool USE_TRACK_AVRG = false;

    TModelParams prevSumDelta;
    TVector<TModelParams> prevTuneSub;
    prevTuneSub.resize(agentCount);

    for (yint metaIter = 0;; ++metaIter) {
        paramTrackArr.push_back(fed.Common.Params);
        {
            TAgentContext &agent = *fed.AgentArr[0];
            const TTrainContext &trainCtx = agent.TrainCtx;
            TVector<TModelParams> &trackArr = USE_TRACK_AVRG ? avrgParamTrackArr : paramTrackArr;
            //log << metaIter;
            yint hist[] = { 1, 2, 5, 10 };
            yint tLast = YSize(trackArr) - 1;
            for (yint histIdx = 0; histIdx < ARRAY_SIZE(hist); ++histIdx) {
                yint len = hist[histIdx];
                TModelParams sum = trackArr[tLast];
                for (yint t = tLast - len + 1; t < tLast; ++t) {
                    AddScaled(&sum, trackArr[Max<yint>(0, t)], 1., 0);
                }
                Scale(&sum, 1. / len, 0);
                pCtx->SetParams(sum);
                float trainErr = CalcModelErr(trainCtx.GetScoreTrainBatches(), pCtx.Get()) * trainCtx.GetCompression();
                float testErr = CalcModelErr(trainCtx.GetScoreTestBatches(), pCtx.Get()) * trainCtx.GetCompression();
                DebugPrintf("len %g, train err %g, test err %g\n", len * 1., trainErr, testErr); fflush(0);
                //log << "\t" << trainErr << "\t" << testErr;
                if (histIdx == ARRAY_SIZE(hist) - 1) {
                    //log << testErr;
                }
            }
            //log << "\n";
            DebugPrintf("\n");
        }


        TModelParams sumDelta = fed.Common.Params;
        Scale(&sumDelta, 0, 0);

        TModelParams avrgTrack = fed.Common.Params;
        Scale(&avrgTrack, 0, 0);

        for (yint agentId = 0; agentId < agentCount; ++agentId) {
            DebugPrintf("meta iter %g, agent %g\n", metaIter * 1., agentId * 1.);
            TAgentContext &agent = *fed.AgentArr[agentId];
            const TTrainContext &trainCtx = agent.TrainCtx;
            const TTrainConfig &tc = trainCtx.GetConfig();

            {
                TModelParams startParams = agent.Tune;
                AddScaled(&startParams, fed.Common.Params, 1., 0);
                pCtx->SetParams(startParams);
            }

            TModelParams sumParams = agent.Tune;
            yint sumWeight = 1;

            for (yint iter = 0; iter <= trainCtx.GetMaxIters(); ++iter) {
                // generate train fragments
                ui64 rngSeed = ((1313 + agentId) * 0xc949d7c7509e6557ULL + metaIter) * 0x9ae16a3b2f90404fULL + iter;
                TXRng iterRng(rngSeed);
                for (yint deviceId = 0; deviceId < deviceCount; ++deviceId) {
                    TVector<TFragment> fragArr;
                    trainCtx.MakeTrainBatches(iterRng, &fragArr);
                    MakeTrain(iterRng, fragArr, tc.TokenDrop, tc.ChannelDrop, pCtx.Get(), deviceId);
                }
                EAddToModel addToModel = GRADIENT_APPLY;
                pCtx->Backprop(trainCtx.GetStep(iter), addToModel);

                if (USE_TRACK_AVRG && (iter % 10) == 0) {
                    TModelParams xx;
                    pCtx->GetParams(&xx);
                    AddScaled(&sumParams, xx, 1., 0);
                    sumWeight += 1;
                }
            }
            AddScaled(&avrgTrack, sumParams, 1. / agentCount / sumWeight, 0);

            TModelParams newParams;
            pCtx->GetParams(&newParams);
            TModelParams delta = newParams;
            AddScaled(&delta, fed.Common.Params, -1, 0);
            float globalWeight = 1. / agentCount; // weight of this agent

            if (UPDATE_LAG) {
                // 0.5 is required for convergence for 1 step sumDelta lag
                globalWeight *= 0.5;
            }
            AddScaled(&sumDelta, delta, globalWeight, 0);
            agent.Tune = delta;
            if (UPDATE_LAG) {
                //Scale(&agent.Tune, 1 - globalWeight); // can not do this with lagged updates - weight is unknown at this point
                //Scale(&agent.Tune, 0.5f); // subtract 0.5 ours, add 0.5 global
                // requires keeping sent delta
                TModelParams newSub = agent.Tune;
                Scale(&newSub, -globalWeight, 0);
                if (!prevTuneSub[agentId].IsEmpty()) {
                    AddScaled(&agent.Tune, prevTuneSub[agentId], 1, 0);
                }
                prevTuneSub[agentId] = newSub;
            } else {
                Scale(&agent.Tune, 1 - globalWeight, 1);
            }
        }
        avrgParamTrackArr.push_back(avrgTrack);

        // add common delta (sim lag)
        if (UPDATE_LAG) {
            if (!prevSumDelta.IsEmpty()) {
                AddScaled(&fed.Common.Params, prevSumDelta, 1, 0);
            }
            prevSumDelta = sumDelta;
        } else {
            AddScaled(&fed.Common.Params, sumDelta, 1, 0);
        }
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////

void Run()
{
#ifdef _MSC_VER
    SetConsoleCP(CP_UTF8);
    SetConsoleOutputCP(CP_UTF8);
    SetProcessAffinityMask(GetCurrentProcess(), 0xffff); // use P-cores, mask is cpu dependent
#endif

    TString trainConfig = "b64f64";
    //TString trainConfig = "b256f64";
    //TString dropConfig = "drop1ch1";
    TString dropConfig = "drop1ch1reg1000";
    TTrainConfig tc(trainConfig, dropConfig);

    TFedTrain fed;
    fed.AddAgent(tc, "D:/text/cultura_y/27.bin", "d:/fed_log1.txt");
    fed.AddAgent(tc, "D:/text/cultura_y/27.bin", "d:/fed_log2.txt");
    fed.AddAgent(tc, "D:/text/cultura_y/27.bin", "d:/fed_log3.txt");
    fed.AddAgent(tc, "D:/text/cultura_y/27.bin", "d:/fed_log4.txt");
    fed.AddAgent(tc, "D:/text/cultura_y/27.bin", "d:/fed_log5.txt");
    //for (yint k = 0; k < 25; ++k) {
    //    fed.AddAgent(tc, "D:/text/cultura_y/27.bin", Sprintf("d:/fed_log%g.txt", k + 1.));
    //}
    //fed.AddAgent(tc, "D:/text/Gutenberg/7.bin", "d:/fed_log2.txt"); // en
    //fed.AddAgent(tc, "D:/text/librusec/83.bin", "d:/fed_log2.txt");
    Train(&fed);
}
}
