#include "stdafx.h"
#include "bin_class.h"
#include <gpt/data/data.h>
#include <gpt/compute/model.h>
#include <gpt/compute/gpt_cuda.cuh>
#include <lib/features_txt/doc_info.h>
#include <lib/features_txt/make_bin_features.h>
#include <lib/random/rand_utils.h>
#include <lib/file/dir.h>


namespace NBinClass
{
//const TString MODEL_DIMS = "e256q128tt64";
const TString MODEL_DIMS = "e512q128tt64";

//const yint LAYER_COUNT = 0;
const yint LAYER_COUNT = 1;
//const yint LAYER_COUNT = 4;
//const yint LAYER_COUNT = 16;
//const yint LAYER_COUNT = 32;
//const yint LAYER_COUNT = 64;
//const yint LAYER_COUNT = 128;

//const yint HEAD_COUNT = 1;
//const yint HEAD_COUNT = 4;
const yint HEAD_COUNT = 16;
//const yint HEAD_COUNT = 64;

const float TOKEN_DROP = 1;
//const float TOKEN_DROP = 0.9f;
//const float TOKEN_DROP = 0.8f;
//const float TOKEN_DROP = 0.65f;
//const float TOKEN_DROP = 0.5f;

const float CHANNEL_DROP = 1;
//const float CHANNEL_DROP = 0.95f;
//const float CHANNEL_DROP = 0.9f;
//const float CHANNEL_DROP = 0.8f;

//const float LEARNING_RATE = 0.1f;
//const float LEARNING_RATE = 0.01f;
//const float LEARNING_RATE = 0.003f;
const float LEARNING_RATE = 0.001f;
//const float LEARNING_RATE = 0.0003f;
//const float LEARNING_RATE = 0.0001f;
const yint BUFFER_LEN = 65536;

///////////////////////////////////////////////////////////////////////////////////////////////////
// attention graph
static yint BCGetLabelCount(yint vocabSize, yint featureCount)
{
    (void)vocabSize;
    return featureCount * 2 + 1;
    //return featureCount;
}

template <class TMyRng>
static void BCGenerateAttentionGraph(
    const TModelDim &modelDim, TMyRng &rng, float tokenDrop,
    const TVector<TVector<bool>> &features, const TVector<TBPEToken> &target, yint lossStart,
    const TVector<yint> &shIndex,
    TVector<TVector<TLabelIndex>> *pLabels, TVector<TVector<TAttentionSpan>> *pAtt, TVector<TNodeTarget> *pTargetArr)
{
    yint len = YSize(shIndex);
    yint featureCount = YSize(features);
    pLabels->resize(len);
    pAtt->resize(len);
    for (yint z = 0; z < len; ++z) {
        yint t = shIndex[z];
        for (yint f = 0; f < featureCount; ++f) {
            if (rng.GenRandReal3() <= tokenDrop) {
                // make gaps to fill by training
                (*pLabels)[z].push_back(1 + f * 2 + features[f][t]);
            }
        }
        // makes worse somehow
        //if (z > 0) {
        //    (*pAtt)[z].push_back(TAttentionSpan(0, z - 1));
        //}
        if (t >= lossStart) {
            pTargetArr->push_back(TNodeTarget(z, target[t]));
        }
    }
}



///////////////////////////////////////////////////////////////////////////////////////////////////
// make train/test contexts

static void BCInitLabelData(const TModelDim &modelDim, TXRng &rng, float tokenDrop,
    yint shuffleCount,
    const TVector<TVector<bool>> &features, const TVector<TBPEToken> &target, yint totalSize, yint trainSize,
    TNodesBatch *pNodes)
{
    yint attentionWidthCount = modelDim.GetAttentionWidthCount();
    pNodes->Init(attentionWidthCount);

    // att sink
    {
        TVector<TLabelIndex> startToken;
        startToken.push_back(0);
        TVector<TVector<TAttentionSpan>> rrArr;
        rrArr.resize(attentionWidthCount);
        pNodes->AddSample(-1, startToken, rrArr);
    }

    for (yint k = 0; k < shuffleCount; ++k) {
        yint ptr = pNodes->GetNodeCount();
        TVector<yint> shTrain;
        for (yint i = 0; i < trainSize; ++i) {
            shTrain.push_back(i);
        }
        TVector<yint> shTest;
        for (yint i = trainSize; i < totalSize; ++i) {
            shTest.push_back(i);
        }
        Shuffle(shTrain.begin(), shTrain.end(), rng); // required for reliable results
        Shuffle(shTest.begin(), shTest.end(), rng);
        TVector<yint> shIndex = shTrain;
        shIndex.insert(shIndex.end(), shTest.begin(), shTest.end());

        TVector<TVector<TLabelIndex>> fragLabels;
        TVector<TVector<TAttentionSpan>> fragAttSpans;
        TVector<TNodeTarget> fragTargets;
        BCGenerateAttentionGraph(modelDim, rng, tokenDrop,
            features, target, trainSize,
            shIndex,
            &fragLabels, &fragAttSpans, &fragTargets);

        yint nodeCount = YSize(fragLabels);
        Y_ASSERT(nodeCount == YSize(fragAttSpans));
        for (yint t = 0; t < nodeCount; ++t) {
            TVector<TAttentionSpan> rr = fragAttSpans[t];
            for (TAttentionSpan &span : rr) {
                span.Shift(ptr);
            }
            rr.push_back(TAttentionSpan(0, 0)); // add attention to attention sink
            TVector< TVector<TAttentionSpan>> rrArr;
            rrArr.resize(attentionWidthCount, rr); // single element is sufficient
            pNodes->AddSample(0, fragLabels[t], rrArr);
        }

        for (TNodeTarget nt : fragTargets) {
            nt.Node += ptr;
            pNodes->Target.push_back(nt);
        }
    }
}


static void BCMakeTrain(TXRng &rng,
    yint shuffleCount,
    const TVector<TVector<bool>> &features, const TVector<TBPEToken> &target, yint trainSize,
    float tokenDrop, float channelDrop,
    IComputeContext *pCtx)
{
    TNodesBatch &nodes = pCtx->GetNodes(MAIN_DEVICE);
    TVector<ui32> &dropTable = pCtx->GetDropTable(MAIN_DEVICE);
    TModelDim modelDim = pCtx->GetModelDim();
    BCInitLabelData(modelDim, rng, tokenDrop, shuffleCount, features, target, trainSize, 0, &nodes);
    MakeDropTable(rng, modelDim, &dropTable, channelDrop);
    pCtx->Init(MAIN_DEVICE);
}


static void BCMakeTest(TXRng &rng,
    yint shuffleCount,
    const TVector<TVector<bool>> &features, const TVector<TBPEToken> &target, yint totalSize, yint trainSize,
    IComputeContext *pCtx, TVector<TNodeTarget> *pTarget)
{
    TNodesBatch &nodes = pCtx->GetNodes(MAIN_DEVICE);
    TVector<ui32> &dropTable = pCtx->GetDropTable(MAIN_DEVICE);
    TModelDim modelDim = pCtx->GetModelDim();
    BCInitLabelData(modelDim, rng, 1., shuffleCount, features, target, totalSize, trainSize, &nodes);
    dropTable.resize(0);
    dropTable.resize(CalcDropTableSize(modelDim), ~0);
    pCtx->Init(MAIN_DEVICE);
    if (pTarget) {
        *pTarget = nodes.Target;
    }
}


static double CalcModelErr(const TVector<TVector<bool>> &features, const TVector<TBPEToken> &target, yint totalSize, yint trainSize, IComputeContext *pCtx)
{
    TXRng rng(1313);
    double err = 0;
    double count = 0;
    yint shuffleCount = 1;
    for (yint batch = 0; batch < 2; ++batch) {
        TVector<TNodeTarget> targetArr;
        BCMakeTest(rng, shuffleCount, features, target, totalSize, trainSize, pCtx, &targetArr);
        TVector<TVector<float>> predArr;
        pCtx->ComputeFragmentPredictions(&predArr);
        for (const TNodeTarget &nt : targetArr) {
            err += log(predArr[nt.Node][nt.TargetId]);
        }
        count += YSize(targetArr);
    }
    return err / count;
}


void Run()
{
    ChDir("d:/111");
    ////r250_init(1313);
    //r250_init(1316);
    r250_init(1317);
    //r250_init(GetTickCount());

    printf("Loading learn data\n");
    vector<TDocInfo> docInfos;

    //int xxx = docInfos.size();
    ////LoadData(&docInfos, "f500.learn");
    LoadData(&docInfos, "features.txt");
    //yint learnSampleCount = docInfos.size() - xxx;
    ////LoadData(&docInfos, "f500.test");
    LoadData(&docInfos, "featuresTest.txt");

    //r250_init(1313);
    //LoadData(&docInfos, "loanLearn.txt");
    //printf("warning, no dataset shuffle\n");
    for (int i = 0; i < docInfos.size(); ++i) {
        swap(docInfos[i], docInfos[i + r250n(YSize(docInfos) - i)]);
    }

    // MatrixNet
    vector<TBPEToken> target;
    //vector<int> queryId, groupId;
    vector<vector<bool> > features;
    ExtractBoolsFromDocInfo(docInfos, &features);
    printf("%d features\n", (int)YSize(features));

    for (yint i = 0; i < YSize(docInfos); ++i) {
        bool bRelev = (docInfos[i].fRelev > 0.07);
        target.push_back(bRelev);
    }
    Shuffle(&features);
    //features.resize(32);
    features.resize(78);

    //yint learnSampleCount = YSize(docInfos) / 2;
    //yint learnSampleCount = 30000;
    yint learnSampleCount = 20000;
    //yint learnSampleCount = 1000;

    yint testSampleCount = 5000;
    yint shuffleCount = 1; // how many time we sample orders

    TVector<float> biasArr;
    biasArr.resize(2, 0.);
    {
        TXRng rng(1313); // 60.??
        //TXRng rng(1314); // 60.??
        //TXRng rng(1315); // 60.40

        TModelDim modelDim;
        yint vocabSize = 2;
        InitModelDim(&modelDim, MODEL_DIMS, ALIBI_NONE, vocabSize, BCGetLabelCount(vocabSize, YSize(features)), MPF_NOFLAGS);

        modelDim.Layers.resize(LAYER_COUNT);
        for (TVector<TModelDim::TAttentionPosParams> &lpArr : modelDim.Layers) {
            for (yint h = 0; h < HEAD_COUNT; ++h) {
                lpArr.push_back(TModelDim::TAttentionPosParams(0., 0., 0));
            }
        }

        TModelParams startParams;
        InitModel(&startParams, rng, modelDim, COMBINER_INIT_ZERO, biasArr);
        DebugPrintf("Model size %gM\n", CountModelSize(startParams) / 1000000.);

        //{
        //    TArray2D<float> &comb = startParams.FinalLayer;
        //    for (yint y = 0; y < comb.GetYSize(); ++y) {
        //        for (yint x = 0; x < comb.GetXSize(); ++x) {
        //            comb[y][x] *= 4;
        //        }
        //    }
        //}

        TIntrusivePtr<IModel> pModel = CreateModel(1, startParams);
        TIntrusivePtr<IComputeContext> pCtx = NCUDA_GPT::CreateContext(pModel, BUFFER_LEN);
        Y_VERIFY(pCtx->GetDeviceCount() == 1);

        const yint EVAL_INTERVAL = 10;

        yint batchCount = 0;
        for (yint iter = 0; ; ++iter) {
            if ((iter % EVAL_INTERVAL) == 0) {
                double trainErr = CalcModelErr(features, target, learnSampleCount, 0, pCtx.Get());
                double testErr = CalcModelErr(features, target, learnSampleCount + testSampleCount, learnSampleCount, pCtx.Get());
                DebugPrintf("iter %g, train %g test %g\n", iter * 1., trainErr, testErr);
            }

            const yint TRAIN_BATCHES = 1;
            //const yint TRAIN_BATCHES = 4;
            //const yint TRAIN_BATCHES = 16;
            for (yint k = TRAIN_BATCHES - 1; k >= 0; --k) {
                EAddToModel addToModel = (k == 0) ? GRADIENT_APPLY : GRADIENT_ACCUMULATE;
                BCMakeTrain(rng, shuffleCount, features, target, learnSampleCount, TOKEN_DROP, CHANNEL_DROP, pCtx.Get());
                pCtx->Backprop(TTrainingStep(LEARNING_RATE, 0), addToModel);
            }
        }
    }
}
}
