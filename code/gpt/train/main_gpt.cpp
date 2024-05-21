#include "stdafx.h"
#include <gpt/data/data.h>
#include <gpt/data/bpe.h>
#include <gpt/att/sliding_window.h>
#include <gpt/model/gpt_cpu.h>
#include <gpt/model/gpt_cuda.cuh>
#include "bin_class.h"
#include "train.h"
#include "net_train.h"
#include <gpt/model/par_delta_accum.h>
#include <lib/random/rand_utils.h>
#include <lib/hp_timer/hp_timer.h>
#include <lib/config/config.h>
#include <lib/config/cfg_file.h>


static TString CONFIG =
    " SAVE_MODEL = false"
    " MAX_ITERS = 2000000"
    " EVAL_INTERVAL = 1000"
    " EVAL_ITERS = 20"
    //" USE_PPM = true"
    // batch, window, sliding window
    " TRAIN_CONFIG = 'b64ww64'"
    //" TRAIN_CONFIG = 'b64ww256'"
    //" TRAIN_CONFIG = 'b64ww64slide4'"
    // dropout, learning rate
    " DROP_CONFIG = 'drop0.9ch0.9'"
    //" DROP_CONFIG = 'drop0.8ch0.8'"
    // model width, depth
    //" MODEL_DIMS = 'e256d1'"
    " MODEL_DIMS = 'e256d64'" // 25M, default
    //" MODEL_DIMS = 'e512d64'" // 50M
    // load data, create model, train
    " make_char_dataset('wiki7_filter.txt')"
    //" save_dataset('d:/dataset.bin')"
    //" load_dataset('d:/dataset.bin')"
    " create_model(MPF_TAIL_LOSS)"
    //" create_model(MPF_TAIL_LOSS, MPF_TUNE_FINAL_LAYER, MPF_TUNE_EMBED)"
    " train()\n"
    //" compute_exact_test(135000,10000)\n"
    ;


//// this train run achieves loss of 0.7185 on test or about 1.04 bpc on enwik8
//static TString CONFIG =
//    " MAX_ITERS = 500000"
//    " EVAL_INTERVAL = 1000"
//    " EVAL_ITERS = 20"
//    //" USE_PPM = true"
//    //" TRAIN_CONFIG = 'b16ww1024'"
//    //" TRAIN_CONFIG = 'b16ww1024w1024'"
//    //" TRAIN_CONFIG = 'b4ww1024slide4'"
//    " TRAIN_CONFIG = 'b4ww4096'"
//    //" TRAIN_CONFIG = 'a4b4ww4096'"
//    //" DROP_CONFIG = 'drop0.9ch0.9'"
//    " DROP_CONFIG = 'drop0.9ch0.9tail3'"
//    //" DROP_CONFIG = 'drop0.8ch0.8'"
//    " MODEL_DIMS = 'e512d64'"
//    " make_char_dataset('D:/111enwiki9/enwik8')"
//    //" load_checkpoint(250000)\n"
//    " create_model(MPF_TAIL_LOSS, MPF_TUNE_FINAL_LAYER, MPF_TUNE_EMBED)"
//    " train()\n"
//    " compute_exact_test(500000,0)\n"
//    ;


//// train gpt2 small model size on owt
//static TString CONFIG =
//    " MAX_ITERS = 2000000"
//    " EVAL_INTERVAL = 1000"
//    " EVAL_ITERS = 20"
//    //" USE_PPM = true"
//    " TRAIN_CONFIG = 'a4b16ww1024'"
//    " DROP_CONFIG = 'drop1ch1tail5'"
//    " MODEL_DIMS = 'e512tt128d86'" // match 124M param model (gpt2-small, they do not count final layer) on OWT
//    //" MODEL_DIMS = 'e512tt256d45'"
//    " set_vocab_size(50257)"
//    " load_tokenized_train('D:/111enwiki9/gpt2_train.bin')"
//    " load_tokenized_test('D:/111enwiki9/gpt2_test.bin')"
//    " create_model(MPF_TUNE_FINAL_LAYER, MPF_TUNE_EMBED)"
//    " train()\n"
//    ;


//// distributed ru model train
//static TString CONFIG =
//    " DEVICE_COUNT = 4"
//    " MAX_ITERS = 1500000"
//    " EVAL_INTERVAL = 3000"
//    " EVAL_ITERS = 5"
//    //" USE_PPM = true"
//    //" TRAIN_CONFIG = 'b16ww1024'"
//    " TRAIN_CONFIG = 'b32ww1024'"
//    //" DROP_CONFIG = 'drop1ch1'"
//    " DROP_CONFIG = 'drop1ch1tail5'"
//    //" MODEL_DIMS = 'e512tt128d64'"
//    " MODEL_DIMS = 'e512tt256d64'"
//    " load_tokenizer('d:/tokenizers/50k.bin')"
//    //" load_tokenizer('d:/tokenizers/5k.bin')"
//    " load_indexed_docset_folder('D:/text/Gutenberg/', 1)"
//    " load_indexed_docset_folder('D:/text/open_web_text/', 1)"
//    " load_indexed_docset_folder('D:/text/librusec/', 1)"
//    " load_indexed_docset_folder('D:/text/cultura_y/', 1)"
//    //" create_model(MPF_TAIL_LOSS, MPF_TUNE_FINAL_LAYER, MPF_TUNE_EMBED)"
//    //" load_checkpoint(1134000)\n"
//    //" train()\n"
//    " net_train('d:/workers_net.txt')\n"
//    //" compute_exact_test(56000)\n"
//    ;


// index datasets
//static TString CONFIG =
//    " USE_PPM = true"
//    " TEST_FRACTION = 0"
//    " load_tokenizer('d:/tokenizers/50k.bin')"
//    " index_docset_folder('D:/text/Gutenberg/')"
//    " index_docset_folder('D:/text/open_web_text/')"
//    " index_docset_folder('D:/text/librusec/')"
//    " index_docset_folder('D:/text/cultura_y/')"
//    ;


///////////////////////////////////////////////////////////////////////////////////////////////////
static void TestGradient(const TModelParams &params, const TTrainConfig &tc, TDataset &data)
{
    TXRng rng(1313);

    TVector<TFragment> dataBatch;
    for (yint k = 0; k < 1; ++k) {
        TFragment frag;
        data.MakeFragment(TDataset::TRAIN, rng, tc.TrainFragLen, &frag);
        dataBatch.push_back(frag);

    }

    TIntrusivePtr<IModel> pModel = CreateModel(1, params);
    TIntrusivePtr<IComputeContext> pCtx = NCPU_GPT::CreateContext(pModel, tc.GetMaxNodeCount());

    yint labelCount = params.GetModelDim().LabelCount;
    pCtx->SetParams(params);

    const float NO_DROP = 1;

    TVector<TNodeTarget> target;
    MakeTrain(rng, dataBatch, NO_DROP, NO_DROP, tc.Window, pCtx.Get(), MAIN_DEVICE, &target);

    TVector<TVector<float>> predArr;
    pCtx->ComputeFragmentPredictions(&predArr);
    double loss = CalcTargetLoss(predArr, target);
    pCtx->Backprop(tc.LearningRate);
    TModelParams grad;
    pCtx->GetGradient(&grad);

    //for (;;) {
    //    double TARGET_STEP = 0.00001;
    //    //double TARGET_STEP = 0.001;

    //    yint label = rng.Uniform(labelCount);
    //    yint h1 = rng.Uniform(10);
    //    yint h2 = rng.Uniform(10);
    //    yint d = (LAYER_COUNT > 0) ? rng.Uniform(LAYER_COUNT) : 0;

    //    double trueGrad = grad.LabelEmbed[label][h1];
    //    //double trueGrad = grad.FinalLayer[h1][h2];
    //    //double trueGrad = grad.LayerArr[d].Step1.KQV[h1][h2];
    //    //double trueGrad = grad.LayerArr[d].Step1.Combiner[h1][h2];
    //    if (trueGrad == 0) {
    //        continue;
    //    }
    //    TModelParams chk = params;
    //    double mult = TARGET_STEP / trueGrad;
    //    chk.LabelEmbed[label][h1] += mult;
    //    //chk.FinalLayer[h1][h2] += mult;
    //    //chk.LayerArr[d].Step1.KQV[h1][h2] += mult;
    //    //chk.LayerArr[d].Step1.Combiner[h1][h2] += mult;
    //    pModel->SetParams(chk);

    //    pCtx->ComputeFragmentPredictions(&predArr);
    //    double chkLoss = CalcTargetLoss(predArr, target);
    //    double deltaLoss = (chkLoss - loss);
    //    DebugPrintf("(d%g %g %g), grad %g, delta loss %g, ratio %g\n", d * 1., h1 * 1., h2 * 1., trueGrad, deltaLoss, deltaLoss / TARGET_STEP);
    //}

    //double gradScale = 1 / sqrt(CalcSum2(grad));
    //Scale(&grad, gradScale);
    double gradSum2 = CalcSum2(grad);

    double sumErr = 0;
    double sumCount = 0;
    for (;;) {
        TModelParams g1 = grad;
        Scale(&g1, 1 / gradSum2);
        Randomize(rng, &g1);

        const double STEP = 0.1;

        TModelParams chk = params;
        AddScaled(&chk, g1, (float)STEP);

        pCtx->SetParams(chk);

        pCtx->ComputeFragmentPredictions(&predArr);
        double chkLoss = CalcTargetLoss(predArr, target);
        double trueDelta = (chkLoss - loss);
        double expectDelta = CalcDot(g1, grad) * STEP;
        sumErr += Sqr((trueDelta - expectDelta) / STEP);
        sumCount += 1;
        DebugPrintf("%g - %g, avrg err %g\n", trueDelta, expectDelta, sqrt(sumErr / sumCount));
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// compute model params distribution
static void ComputeMatrixParamDistr(TModelParams *pParams)
{
    for (yint d = 0; d < YSize(pParams->LayerArr); ++d) {
        for (TModelParams::TAttentionMatrices &att : pParams->LayerArr[d]) {
            //TArray2D<float> &matr = att.Combiner;
            //TArray2D<float> &matr = att.K;
            TArray2D<float> &matr = att.V;
            yint xSize = matr.GetXSize();
            yint ySize = matr.GetYSize();
            double sum2 = 0;
            for (yint y = 0; y < ySize; ++y) {
                for (yint x = 0; x < xSize; ++x) {
                    sum2 += Sqr(matr[y][x]);
                }
            }
            double sko = sqrt(sum2 / xSize / ySize);
            double count3 = 0;
            double count5 = 0;
            double count7 = 0;
            for (yint y = 0; y < ySize; ++y) {
                for (yint x = 0; x < xSize; ++x) {
                    double val = fabs(matr[y][x] / sko);
                    if (val > 3) {
                        count3 += 1;
                    }
                    if (val > 5) {
                        count5 += 1;
                    }
                    if (val > 7) {
                        count7 += 1;
                    }
                }
            }
            double scale = 100. / xSize / ySize;
            DebugPrintf("depth %g, sko %g, 3sko %g%%, 5sko %g%%, 7sko %g%%\n", d + 1., sko, count3 * scale, count5 * scale, count7 * scale);
        }
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// compute score on test set
static void ComputeExactTest(TDataset &data, yint finishIter, yint iterInterval)
{
    //TString pathTemplate = "D:/models/win4k/eden_gpt_%.8gk.bin";
    TString pathTemplate = "D:/eden_gpt_%.8gk.bin";
    //TString pathTemplate = "D:/models/gpt2/eden_gpt_%.8gk.bin";
    //TString pathTemplate = "D:/models/gpt2-batch64/eden_gpt_%.8gk.bin";

    // model averaging boosts perf on test significantly
    int startIter = finishIter - iterInterval;
    double modelCount = 1;
    TModelParams sumParams;
    Serialize(true, Sprintf(pathTemplate.c_str(), startIter / 1000.), sumParams);
    const int STEP = 1000;
    //const int STEP = 100;
    for (int iter = startIter + STEP; iter <= finishIter; iter += STEP) {
        TModelParams params;
        Serialize(true, Sprintf(pathTemplate.c_str(), iter / 1000.), params);
        AddScaled(&sumParams, params, 1);
        modelCount += 1;
        printf(".");
    }
    printf("\n");
    Scale(&sumParams, 1 / modelCount);
    //ComputeMatrixParamDistr(&startParams);

    yint fragLen = sumParams.ModelDim.FragLen;
    TWindowSizeLimit window = sumParams.ModelDim.Window;
    //yint testBatchSize = BUFFER_LEN / GetNodeCount(fragLen);
    //yint testBatchSize = 4;
    yint testBatchSize = 1;

    TIntrusivePtr<IModel> pModel = CreateModel(1, sumParams);
    TIntrusivePtr<IComputeContext> pCtx = NCUDA_GPT::CreateContext(pModel, testBatchSize * GetNodeCount(fragLen));
    double sumTestErr = 0;
    double sumCount = 0;
    int rSeed = 31331;
    for (yint iter = 1;; ++rSeed, ++iter) {
        TXRng rng(rSeed);
        TVector<TFragment> batchArr;
        for (yint b = 0; b < testBatchSize; ++b) {
            TFragment frag;
            data.MakeFragment(TDataset::TEST, rng, fragLen, &frag);
            batchArr.push_back(frag);
        }
        float testErr = CalcModelErr(batchArr, window, pCtx.Get()) * data.GetCompression();
        if (isnan(testErr)) {
            DebugPrintf("rseed %g, score is nan\n", rSeed * 1.);
        }
        sumTestErr += testErr;
        sumCount += 1;
        if ((iter % 100) == 0) {
            DebugPrintf("iter %gk, avrg test score %g\n", iter / 1000., sumTestErr / sumCount);
        }
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// check if results are reproducible
template <class T>
double CalcDiff(const TVector<TVector<T>> &cpuPredArr, const TVector<TVector<T>> &gpuPredArr)
{
    double totalDiff = 0;
    for (yint t = 0; t < YSize(cpuPredArr); ++t) {
        for (yint k = 0; k < YSize(cpuPredArr[t]); ++k) {
            totalDiff += Sqr(cpuPredArr[t][k] - gpuPredArr[t][k]);
        }
    }
    return sqrt(totalDiff / YSize(cpuPredArr) / YSize(cpuPredArr[0]));
}


static bool TestMatch(const TArray2D<float> &a, const TArray2D<float> &b)
{
    for (yint y = 0; y < a.GetYSize(); ++y) {
        for (yint x = 0; x < a.GetXSize(); ++x) {
            if (a[y][x] != b[y][x]) {
                printf("%g != %g  (%x %x)\n", a[y][x], b[y][x], *(int*)&a[y][x], *(int*)&b[y][x]);
                return false;
            }
        }
    }
    return true;
}

void TestReproducibility(const TTrainContext &trainCtx, IComputeContext *pCtx, TXRng &rng, const TVector<TFragment> &fragArr)
{
    const TTrainConfig &tc = trainCtx.GetConfig();

    TModelParams point1;
    pCtx->GetParams(&point1);
    pCtx->SetParams(point1);

    TXRng chkRng = rng;
    TVector<TNodeTarget> batchTarget;
    MakeTrain(rng, fragArr, tc.TokenDrop, tc.ChannelDrop, tc.Window, pCtx, MAIN_DEVICE, &batchTarget);
    pCtx->Backprop(tc.LearningRate);

    TModelParams point2;
    pCtx->GetParams(&point2);

    for (yint testId = 0; testId < 5; ++testId) {
        pCtx->SetParams(point1);

        pCtx->Backprop(tc.LearningRate);

        TModelParams chk;
        pCtx->GetParams(&chk);

        bool hasMismatch = false;
        if (!TestMatch(chk.LabelEmbed.Matr, point2.LabelEmbed.Matr)) {
            printf("Label embed mismatch\n");
            hasMismatch = true;
        }
        for (yint d = 0; d < YSize(point2.LayerArr); ++d) {
            for (yint k = 0; k < YSize(point2.LayerArr[d]); ++k) {
                const TModelParams::TAttentionMatrices &att1 = point2.LayerArr[d][k];
                const TModelParams::TAttentionMatrices &att2 = chk.LayerArr[d][k];
                if (!TestMatch(att1.QK, att2.QK)) {
                    printf("Layer %g, att %g, QK mismatch\n", d * 1., k * 1.);
                    hasMismatch = true;
                }
                if (!TestMatch(att1.QV, att2.QV)) {
                    printf("Layer %g, att %g, QV mismatch\n", d * 1., k * 1.);
                    hasMismatch = true;
                }
                if (!TestMatch(att1.V, att2.V)) {
                    printf("Layer %g, att %g, V mismatch\n", d * 1., k * 1.);
                    hasMismatch = true;
                }
                if (!TestMatch(att1.K, att2.K)) {
                    printf("Layer %g, att %g, K mismatch\n", d * 1., k * 1.);
                    hasMismatch = true;
                }
                if (!TestMatch(att1.Combiner, att2.Combiner)) {
                    printf("Layer %g, att %g, Combiner mismatch\n", d * 1., k * 1.);
                    hasMismatch = true;
                }
            }
        }
        if (hasMismatch) {
            printf("attempt %g\n", testId + 1.);
            while (hasMismatch) {
                SchedYield();
            }
        }
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
void CheckCpuGpuMatch(const TTrainConfig &tc, TDataset &data)
{
    TVector<yint> attPerLayer;
    AttSquare(&attPerLayer, 16, 4);
    //AttSquare(&attPerLayer, 8, 1);
    //AttSquare(&attPerLayer, 1, 1);
    TXRng chkRng(1313);
    TModelParams params;
    yint vocabSize = data.GetVocabSize();
    //yint modelFlags = 0;
    yint modelFlags = MPF_TUNE_FINAL_LAYER | MPF_TUNE_EMBED;
    TString modelDims = "e256d16";
    InitModel(&params, chkRng,
        modelDims, vocabSize, GetLabelCount(modelFlags, vocabSize),
        ALIBI_V1, COMBINER_INIT_RANDOM,
        data.GetBias(), attPerLayer, modelFlags);
    //Serialize(true, "d:/eden_gpt_0.bin", startParams);
    //Serialize(true, "d:/eden_gpt_3k.bin", startParams);
    //startParams.LayerArr.resize(1);
    //startParams.ModelDim.Layers.resize(1);
    const yint CHECK_BATCH_SIZE = 1;
    TWindowSizeLimit checkWindow(64, 64);
    const yint CHECK_FRAG_LEN = checkWindow.LimitWide - 1;
    const float CHECK_CHANNEL_DROP = 1;

    TIntrusivePtr<IModel> cpuModel = CreateModel(1, params);
    TIntrusivePtr<IComputeContext> cpuCtx = NCPU_GPT::CreateContext(cpuModel, CHECK_BATCH_SIZE * GetNodeCount(tc.TrainFragLen));

    TIntrusivePtr<IModel> gpuModel = CreateModel(1, params);
    TIntrusivePtr<IComputeContext> gpuCtx = NCUDA_GPT::CreateContext(gpuModel, CHECK_BATCH_SIZE * GetNodeCount(tc.TrainFragLen));

    TFragment frag;
    data.MakeFragment(TDataset::TRAIN, chkRng, CHECK_FRAG_LEN, &frag);
    TVector<TFragment> xxFrag;
    xxFrag.push_back(frag);

    MakeTest(checkWindow, xxFrag, cpuCtx.Get(), MAIN_DEVICE);
    MakeTest(checkWindow, xxFrag, gpuCtx.Get(), MAIN_DEVICE);

    TVector<TVector<float>> cpuPredArr;
    cpuCtx->ComputeFragmentPredictions(&cpuPredArr);
    TVector<TVector<float>> gpuPredArr;
    gpuCtx->ComputeFragmentPredictions(&gpuPredArr);

    int t = 15;
    //int t = 0;
    for (yint k = 0; k < 5; ++k) {
        DebugPrintf("%g - %g\n", cpuPredArr[t][k], gpuPredArr[t][k]);
    }
    DebugPrintf("\nDiff %g bp\n", CalcDiff(cpuPredArr, gpuPredArr) * 10000);

    TXRng cpuRng = chkRng;
    TXRng gpuRng = chkRng;
    MakeTrain(cpuRng, xxFrag, tc.TokenDrop, CHECK_CHANNEL_DROP, checkWindow, cpuCtx.Get(), MAIN_DEVICE);
    cpuCtx->Backprop(tc.LearningRate * 10);
    cpuCtx->ComputeFragmentPredictions(&cpuPredArr);

    MakeTrain(gpuRng, xxFrag, tc.TokenDrop, CHECK_CHANNEL_DROP, checkWindow, gpuCtx.Get(), MAIN_DEVICE);
    gpuCtx->Backprop(tc.LearningRate * 10);
    gpuCtx->ComputeFragmentPredictions(&gpuPredArr);

    DebugPrintf("\nAfter backprop\n");
    for (yint k = 0; k < 5; ++k) {
        DebugPrintf("%g - %g\n", cpuPredArr[t][k], gpuPredArr[t][k]);
    }
    DebugPrintf("\nDiff %g bp\n\n", CalcDiff(cpuPredArr, gpuPredArr) * 10000);
    for (;;) {
        SchedYield();
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////
struct TMainGpt
{
    TDataset Data;
    TTokenizer Tokenizer;
    TModelParams StartParams;
    TIntrusivePtr<TDatasetBuilder> DataBuild;
    yint VocabSize = 0;
    yint OverrideDocStartToken = -1;


    void CreateModel(const TConfigFile::TOp &op, const TString &modelDimsString, bool usePPM)
    {
        TXRng rng(1313);

        Y_VERIFY(VocabSize > 0 && "unknown vocab size, no train data?");
        FinishDatasetBuild();
        ui64 modelFlags = 0;
        if (usePPM) {
            modelFlags |= MPF_PPM;
        }
        for (const TString &flag : op.Args) {
            if (flag == "MPF_HASHED_EMBED") {
                modelFlags |= MPF_HASHED_EMBED;
            } else if (flag == "MPF_TUNE_FINAL_LAYER") {
                modelFlags |= MPF_TUNE_FINAL_LAYER;
            } else if (flag == "MPF_TUNE_EMBED") {
                modelFlags |= MPF_TUNE_EMBED;
            } else if (flag == "MPF_TAIL_LOSS") {
                modelFlags |= MPF_TAIL_LOSS;
            } else if (flag == "MPF_SIM_QUANT_2BIT") {
                modelFlags |= MPF_SIM_QUANT_2BIT;
            } else if (flag == "MPF_GROK_BINARY_OP") {
                modelFlags |= MPF_GROK_BINARY_OP;
            } else {
                DebugPrintf("unknown model flag %s\n", flag.c_str());
            }
        }
        // initialize model params
        TStringParams modelDims(modelDimsString);
        yint layerCount = modelDims.GetParam("d", 64);

        TVector<yint> attPerLayer;
        AttSquare(&attPerLayer, layerCount, 1);
        //// comparable results, faster due to higher parallelism
        //AttSquare(&attPerLayer, 8, 1);
        //AttSquare(&attPerLayer, (layerCount - 8) / 4, 4);

        StartParams = TModelParams();
        InitModel(&StartParams, rng,
            modelDimsString, VocabSize, GetLabelCount(modelFlags, VocabSize),
            ALIBI_V1, COMBINER_INIT_ZERO,
            Data.GetBias(), attPerLayer, modelFlags);
        if (OverrideDocStartToken >= 0) {
            StartParams.ModelDim.SetDocStartToken(OverrideDocStartToken);
        } else if (Tokenizer.HasDocStartToken()) {
            StartParams.ModelDim.SetDocStartToken(Tokenizer.GetDocStartToken());
        } else {
            Y_ASSERT(!StartParams.ModelDim.HasFlag(MPF_USE_DOC_START_TOKEN));
        }
    }

    void CreateDatasetBuilders(bool usePPM)
    {
        Y_VERIFY(!Tokenizer.IsEmpty());
        Y_VERIFY(StartParams.IsEmpty());
        if (DataBuild.Get() == 0) {
            DataBuild = new TDatasetBuilder(&Data, usePPM, Tokenizer);
        }
    }

    void CreateDatasetBuilders(yint vocabSize, bool usePPM)
    {
        Y_VERIFY(StartParams.IsEmpty());
        Y_VERIFY(vocabSize > 0);
        if (DataBuild.Get() == 0) {
            DataBuild = new TDatasetBuilder(&Data, usePPM, vocabSize, -1);
        } else {
            Y_VERIFY(Data.GetVocabSize() == vocabSize);
        }
    }

    void FinishDatasetBuild()
    {
        DataBuild = 0;
    }

    void VerifyVocabSize()
    {
        Y_VERIFY(Tokenizer.GetVocabSize() == 0 || Tokenizer.GetVocabSize() == VocabSize);
        Y_VERIFY(StartParams.ModelDim.VocabSize == VocabSize);
        Y_VERIFY(Data.GetVocabSize() == VocabSize);
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
void TrainModel(yint startIteration, yint deviceCount, const TTrainContext &trainCtx, const TModelParams &params)
{
    const TTrainConfig &tc = trainCtx.GetConfig();

#ifdef _MSC_VER
    SetConsoleCP(CP_UTF8);
    SetConsoleOutputCP(CP_UTF8);
    SetProcessAffinityMask(GetCurrentProcess(), 0xffff); // use P-cores, mask is cpu dependent
#endif

    // create model
    TIntrusivePtr<TMMDeltaAccumulateGen> deltaHookGen = new TMMDeltaAccumulateGen;
    TIntrusivePtr<IModel> pModel = CreateModel(deviceCount, params, deltaHookGen.Get());
    //TIntrusivePtr<IComputeContext> pCtx = NCPU_GPT::CreateContext(pModel, trainCtx.GetMaxNodeCount());
    TIntrusivePtr<IComputeContext> pCtx = NCUDA_GPT::CreateContext(pModel, trainCtx.GetMaxNodeCount());

    //TOFStream fTrainLog("d:/train_log.txt");
    NHPTimer::STime tStart;
    NHPTimer::GetTime(&tStart);
    yint batchCount = 0;
    for (yint iter = startIteration; iter <= trainCtx.GetMaxIters(); ++iter) {
        if ((iter % trainCtx.GetEvalInterval()) == 0) {
            if (trainCtx.IsSaveModel()) {
                TModelParams params;
                pCtx->GetParams(&params);
                Serialize(false, Sprintf("d:/eden_gpt_%.8gk.bin", iter / 1000.), params);
            }
            float trainErr = CalcModelErr(trainCtx.GetScoreTrainBatches(), tc.Window, pCtx.Get()) * trainCtx.GetCompression();
            float testErr = CalcModelErr(trainCtx.GetScoreTestBatches(), tc.Window, pCtx.Get()) * trainCtx.GetCompression();
            if (testErr != 0) {
                DebugPrintf("iter %.8gk, %g sec, train err %g, test err %g\n", iter / 1000., NHPTimer::GetTimePassed(&tStart), trainErr, testErr); fflush(0);
            } else {
                DebugPrintf("iter %.8gk, %g sec, train err %g\n", iter / 1000., NHPTimer::GetTimePassed(&tStart), trainErr); fflush(0);
            }
            //fTrainLog << trainErr << "\t" << testErr << Endl;
        }

        // accumulate several batches
        EAddToModel addToModel = tc.DoAccumulate(iter) ? GRADIENT_ACCUMULATE : GRADIENT_APPLY;
        deltaHookGen->SetAddToModel(addToModel);

        // generate train fragments
        TXRng iterRng(iter);
        for (yint deviceId = 0; deviceId < deviceCount; ++deviceId) {
            TVector<TFragment> fragArr;
            yint batchId = iter * deviceCount + deviceId;
            trainCtx.MakeTrainBatches(iterRng, batchId, &fragArr);
            MakeTrain(iterRng, fragArr, tc.TokenDrop, tc.ChannelDrop, tc.Window, pCtx.Get(), deviceId);
        }
        pCtx->Backprop(trainCtx.GetStep(iter));

        //printf("Iter %.8gk\n", iter / 1000.);
        //TestReproducibility(trainCtx, pCtx.Get(), iterRng, fragArr);

        //{ // test if we improve train batch
        //    TVector<TVector<float>> predArr;
        //    pCtx->ComputeFragmentPredictions(&predArr);
        //    double errBefore = 0;
        //    for (TNodeTarget &nt : batchTarget) {
        //        errBefore += log(predArr[nt.Node][nt.TargetId]);
        //    }
        //    pCtx->Backprop(batchTarget, LEARNING_RATE, GRADIENT_APPLY);
        //    pCtx->ComputeFragmentPredictions(&predArr);
        //    double errAfter = 0;
        //    for (TNodeTarget &nt : batchTarget) {
        //        errAfter += log(predArr[nt.Node][nt.TargetId]);
        //    }
        //    DebugPrintf("loss before %g after %g\n", errBefore, errAfter);
        //}
    }
}


static void ReadNonEmptyLines(TVector<TString> *pRes, const TString &fName)
{
    TSeqReader fr(fName);
    Y_VERIFY(fr.IsValid() && "file not found");
    while (!fr.IsEof()) {
        TString sz = fr.ReadLine();
        if (!sz.empty()) {
            pRes->push_back(sz);
        }
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
void TestMatMul();
//void Repack();

int main(int argc, char **argv)
{
    //TestMatMul();
    //Repack();
    //GenerateArithmetic();
    //GenerateArithmetic97();
    //NBinClass::Run();
    //return 0;

    TOpt cmdline("c:w:", argc, argv);
    for (const TOpt::TParam &param : cmdline.Params) {
        if (param.Name == "c") {
            DebugPrintf("Executing script %s\n", param.Args[0].c_str());
            TVector<char> cfg;
            ReadWholeFile(param.Args[0], &cfg);
            CONFIG = cfg.data();
        }
        if (param.Name == "w") {
            NNetTrain::RunWorker(atoi(param.Args[0].c_str()));
            return 0;
        }
    }

    TMainGpt gpt;

    TConfigFile cfg;
    ParseConfig(&cfg, CONFIG);
    yint deviceCount = 1;
    yint startIteration = 0;
    bool saveModel = true;
    bool usePPM = false;
    yint maxIters = 2000000;
    yint evalInterval = 1000;
    yint evalIters = 20;
    float testFraction = 0.05f;
    TString trainConfig = "b64w64";
    TString dropConfig = "drop1ch1";
    TString modelDimsString = "e256d64";
    for (const TConfigFile::TOp &op : cfg.OpArr) {
        if (op.Op == CFG_OP_ASSIGNMENT) {
            if (op.Dst == "MAX_ITERS") {
                maxIters = atof(op.Args[0].c_str());
            } else if (op.Dst == "DEVICE_COUNT") {
                deviceCount = atof(op.Args[0].c_str());
                Y_VERIFY(deviceCount >= 1 && deviceCount < 100);
            } else if (op.Dst == "EVAL_INTERVAL") {
                evalInterval = atof(op.Args[0].c_str());
            } else if (op.Dst == "EVAL_ITERS") {
                evalIters = atof(op.Args[0].c_str());
            } else if (op.Dst == "TEST_FRACTION") {
                testFraction = atof(op.Args[0].c_str());
            } else if (op.Dst == "TRAIN_CONFIG") {
                trainConfig = op.Args[0];
            } else if (op.Dst == "DROP_CONFIG") {
                dropConfig = op.Args[0];
            } else if (op.Dst == "SAVE_MODEL") {
                saveModel = (IsYes(op.Args[0]));
            } else if (op.Dst == "USE_PPM") {
                usePPM = (IsYes(op.Args[0]));
            } else if (op.Dst == "MODEL_DIMS") {
                Y_VERIFY(gpt.StartParams.IsEmpty() && "model dimenstion are useless, model already created");
                modelDimsString = op.Args[0];
            } else {
                DebugPrintf("unknown config variable %s\n", op.Dst.c_str());
            }

        } else if (op.Op == CFG_OP_CALL) {
        // model ops
            if (op.Dst == "create_model") {
                gpt.CreateModel(op, modelDimsString, usePPM);

            } else if (op.Dst == "load_model") {
                Y_VERIFY(YSize(op.Args) == 1);
                DebugPrintf("Load model %s\n", op.Args[0].c_str());
                Serialize(true, op.Args[0], gpt.StartParams);
                Y_VERIFY(!gpt.StartParams.IsEmpty());

            } else if (op.Dst == "load_checkpoint") {
                Y_VERIFY(YSize(op.Args) == 1);
                startIteration = atoi(op.Args[0].c_str());
                DebugPrintf("Load checkpoint %gk\n", startIteration / 1000.);
                Serialize(true, Sprintf("d:/eden_gpt_%.8gk.bin", startIteration / 1000.), gpt.StartParams);
                Y_VERIFY(!gpt.StartParams.IsEmpty());

        // tokenizer ops
            } else if (op.Dst == "set_vocab_size") {
                Y_VERIFY(YSize(op.Args) == 1);
                gpt.VocabSize = atoi(op.Args[0].c_str());

            } else if (op.Dst == "set_doc_start_token") {
                Y_VERIFY(YSize(op.Args) == 1);
                gpt.OverrideDocStartToken = atoi(op.Args[0].c_str());

            } else if (op.Dst == "load_tokenizer") {
                Y_VERIFY(YSize(op.Args) == 1);
                Serialize(true, op.Args[0], gpt.Tokenizer);
                gpt.VocabSize = gpt.Tokenizer.GetVocabSize();

            } else if (op.Dst == "make_byte_tokenizer") {
                gpt.Tokenizer.MakeByteEncoder(TTokenizer::TK_CHAR);
                gpt.VocabSize = gpt.Tokenizer.GetVocabSize();

        // dataset ops
            } else if (op.Dst == "make_char_dataset") {
                Y_VERIFY(gpt.StartParams.IsEmpty());
                Y_VERIFY(gpt.Tokenizer.IsEmpty());
                Y_VERIFY(gpt.DataBuild.Get() == 0);
                TVector<char> text;
                LoadDocument(&text, op.Args[0]);
                MakeCharDataset(&gpt.Data, &gpt.Tokenizer, text, testFraction, usePPM);
                gpt.VocabSize = gpt.Tokenizer.GetVocabSize();

            } else if (op.Dst == "load_tokenized_train" || op.Dst == "load_tokenized_test") {
                Y_VERIFY(gpt.StartParams.IsEmpty());
                TVector<TBPEToken> data;
                LoadTokenized(op.Args[0], &data);
                gpt.CreateDatasetBuilders(gpt.VocabSize, usePPM);
                float ltTestFraction = (op.Dst == "load_tokenized_train") ? 0 : 1;
                TDatasetParams params(gpt.VocabSize);
                params.CountDocset(data, 0, YSize(data), ltTestFraction);
                float weight = 1;
                gpt.DataBuild->AddTokenizedDocset(data, params, weight);

            } else if (op.Dst == "load_text" || op.Dst == "load_folder" || op.Dst == "load_docset") {
                Y_VERIFY(gpt.StartParams.IsEmpty());
                Y_VERIFY(!gpt.Tokenizer.IsEmpty());
                Y_VERIFY(YSize(op.Args) > 0);
                TVector<TVector<char>> docSet;
                if (op.Dst == "load_text") {
                    docSet.resize(1);
                    LoadDocument(&docSet[0], op.Args[0]);
                } else if (op.Dst == "load_folder") {
                    LoadDocumentSetFromFiles(&docSet, op.Args[0]);
                } else if (op.Dst == "load_docset") {
                    LoadDocumentSetFromBin(&docSet, op.Args[0]);
                }
                gpt.CreateDatasetBuilders(usePPM);
                float weight = (YSize(op.Args) > 1) ? atof(op.Args[1].c_str()) : 1;
                AddDocset(gpt.DataBuild.Get(), gpt.Tokenizer, docSet, weight, testFraction);

            } else if (op.Dst == "load_indexed_docset_folder") {
                Y_VERIFY(YSize(op.Args) > 0);
                Y_VERIFY(!gpt.Tokenizer.IsEmpty());
                gpt.CreateDatasetBuilders(usePPM);
                float weight = (YSize(op.Args) > 1) ? atof(op.Args[1].c_str()) : 1;
                AddIndexedDocset(gpt.DataBuild.Get(), op.Args[0], weight);

            } else if (op.Dst == "index_docset_folder") {
                Y_VERIFY(YSize(op.Args) == 1);
                Y_VERIFY(!gpt.Tokenizer.IsEmpty());
                IndexDocsetDir(op.Args[0], gpt.Tokenizer, usePPM, testFraction);

            } else if (op.Dst == "save_dataset") {
                Y_VERIFY(YSize(op.Args) == 1);
                gpt.FinishDatasetBuild();
                Serialize(false, op.Args[0], gpt.Data);

            } else if (op.Dst == "load_dataset") {
                Y_VERIFY(YSize(op.Args) == 1);
                Y_VERIFY(gpt.DataBuild.Get() == 0);
                Serialize(true, op.Args[0], gpt.Data);

        // process ops
            } else if (op.Dst == "train" || op.Dst == "net_train") {
                Y_VERIFY(!gpt.StartParams.IsEmpty());
                gpt.FinishDatasetBuild();
                gpt.VerifyVocabSize();
                TTrainConfig tc(trainConfig, dropConfig);
                TTrainContext trainCtx(gpt.Data, tc, saveModel, maxIters, evalInterval);

                DebugPrintf("%s %s %s 0x%x, size %gM\n",
                    modelDimsString.c_str(),
                    tc.GetTrainConfig().c_str(),
                    tc.GetDropConfig().c_str(),
                    (int)gpt.StartParams.ModelDim.Flags,
                    CountModelSize(gpt.StartParams) / 1000000.);

                // create batches for train & test score compute, can use different sizes
                const yint batchSize = tc.TrainBatchSize;
                const yint fragLen = tc.TrainFragLen;
                trainCtx.MakeScoreBatches(evalIters, batchSize, fragLen);

                // keep train params
                gpt.StartParams.ModelDim.Window = tc.Window;
                gpt.StartParams.ModelDim.FragLen = tc.TrainFragLen;

                if (op.Dst == "train") {
                    TrainModel(startIteration, deviceCount, trainCtx, gpt.StartParams);
                } else if (op.Dst == "net_train") {
                    Y_VERIFY(YSize(op.Args) == 1);
                    TVector<TString> workerArr;
                    ReadNonEmptyLines(&workerArr, op.Args[0]);
                    NNetTrain::RunMaster(startIteration, deviceCount, workerArr, trainCtx, gpt.StartParams);
                } else {
                    Y_ASSERT(0);
                }

            } else if (op.Dst == "compute_exact_test") {
                yint finishIter = YSize(op.Args) > 0 ? atoi(op.Args[0].c_str()) : 50000;
                yint iterInterval = YSize(op.Args) > 1 ? atoi(op.Args[1].c_str()) : 0;
                gpt.FinishDatasetBuild();
                ComputeExactTest(gpt.Data, finishIter, iterInterval);

            } else if (op.Dst == "check_cpu_gpu_match") {
                gpt.FinishDatasetBuild();
                TTrainConfig tc(trainConfig, dropConfig);
                CheckCpuGpuMatch(tc, gpt.Data);

            } else if (op.Dst == "test_gradient") {
                gpt.FinishDatasetBuild();
                TTrainConfig tc(trainConfig, dropConfig);
                TestGradient(gpt.StartParams, tc, gpt.Data);

            } else {
                DebugPrintf("unknown function %s\n", op.Dst.c_str());
                abort();
            }
        }
    }
    return 0;
}
