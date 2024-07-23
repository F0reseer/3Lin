#include "stdafx.h"
#include "bin_class.h"
#include "train.h"
#include "net_train.h"
#include "fed_sim.h"
#include "cpu_infer.h"
#include <gpt/data/data.h>
#include <gpt/data/bpe.h>
#include <gpt/att/sliding_window.h>
#include <gpt/compute/gpt_cpu.h>
#include <gpt/compute/gpt_cuda.cuh>
#include <gpt/data_config/data_config.h>
#include <lib/random/rand_utils.h>
#include <lib/hp_timer/hp_timer.h>
#include <lib/config/config.h>


static TString TRAIN_SCRIPT =
    //" SAVE_MODEL = false"
    " MAX_ITERS = 2000000"
    " EVAL_INTERVAL = 100"
    " EVAL_BATCH_COUNT = 20"
    // batch, window, sliding window
    " TRAIN_CONFIG = 'b32f513'" // 16k samples
    " set_vocab_size(70000)"
    " load_bert_train('train_batches')"
    " load_bert_test('test_batches')"
    " DROP_CONFIG = 'drop1ch1'"
    //" MODEL_DIMS = 'e256d1w512'"
    " MODEL_DIMS = 'e512tt128d60w512'" // 150M
    " create_model(MPF_MLM_BERT, MPF_TUNE_FINAL_LAYER, MPF_TUNE_EMBED)"
    //" load_checkpoint(2000)"
    " train()\n"
    ;

//static TString TRAIN_SCRIPT =
//    " SAVE_MODEL = false"
//    " MAX_ITERS = 2000000"
//    " EVAL_INTERVAL = 1000"
//    " EVAL_BATCH_COUNT = 20"
//    //" USE_PPM = true"
//    // batch, window, sliding window
//    " TRAIN_CONFIG = 'b64f64'"
//    //" TRAIN_CONFIG = 'b256f64'"
//    //" TRAIN_CONFIG = 'b64f256'"
//    //" TRAIN_CONFIG = 'b4f4096'"
//    // dropout, learning rate
//    //" DROP_CONFIG = 'drop0.9ch0.9'"
//    " DROP_CONFIG = 'drop0.9ch0.9reg2000'"
//    //" DROP_CONFIG = 'drop0.8ch0.8'"
//    // model width, depth
//    //" MODEL_DIMS = 'e256d1'"
//    " MODEL_DIMS = 'e256d65'" // 25M, default
//    //" MODEL_DIMS = 'e512d65'" // 50M
//    //" MODEL_DIMS = 'e2048tt256d96w4096'"
//    // load data, create model, train
//    " make_char_dataset('D:/111enwiki9/wiki7_filter.txt')"
//    //" save_dataset('d:/dataset.bin')"
//    //" load_dataset('d:/dataset.bin')"
//    //" create_model(MPF_TAIL_LOSS)"
//    " create_model(MPF_TAIL_LOSS, MPF_TUNE_FINAL_LAYER, MPF_TUNE_EMBED)"
//    //" create_model(MPF_MLM_BERT, MPF_TUNE_FINAL_LAYER, MPF_TUNE_EMBED)"
//    //" load_checkpoint(150000)"
//    " train()\n"
//    //" net_train('d:/workers_local.txt')\n"
//    //" compute_exact_test(100000,10000)\n"
//    //" load_model('D:/models/fed_small/model_192.bin')"
//    //" compute_exact_test()\n"
//    ;


//// grok binary ops
//static TString TRAIN_SCRIPT =
//    " SAVE_MODEL = false"
//    " MAX_ITERS = 2000000"
//    " EVAL_INTERVAL = 1000"
//    " EVAL_BATCH_COUNT = 20"
//    " TEST_FRACTION = 0.01"
//    // batch, window, sliding window
//    " TRAIN_CONFIG = 'b64f64'"
//    //" TRAIN_CONFIG = 'b256f64'"
//    //" TRAIN_CONFIG = 'b4f4096'"
//    //" TRAIN_CONFIG = 'b1f32768'"
//    //" TRAIN_CONFIG = 'b16f1024'"
//    //" TRAIN_CONFIG = 'b1f100000'"
//    " DROP_CONFIG = 'drop1ch1'"
//    //" MODEL_DIMS = 'e256d16'"
//    " MODEL_DIMS = 'e512d16'"
//    //" MODEL_DIMS = 'e1024tt128d16'"
//    //" MODEL_DIMS = 'e256d32'"
//    // load data, create model, train
//    " make_char_dataset('D:/arith97.txt')"
//    " create_model(MPF_GROK_BINARY_OP, MPF_TUNE_FINAL_LAYER, MPF_TUNE_EMBED)"
//    " train()\n"
//    ;


//// this train run achieves loss of ??? on test or about ??? bpc on enwik8 (approx 0.71 after 750k iterations, avrg over last 50k iterations)
//static TString TRAIN_SCRIPT =
//    //" MAX_ITERS = 500000"
//    " MAX_ITERS = 2500000"
//    " EVAL_INTERVAL = 1000"
//    " EVAL_BATCH_COUNT = 20"
//    //" USE_PPM = true"
//    //" TRAIN_CONFIG = 'b16f1024'"
//    " TRAIN_CONFIG = 'b4f4096'"
//    //" DROP_CONFIG = 'drop0.9ch0.9'"
//    //" DROP_CONFIG = 'drop0.9ch0.9tail3'"
//    //" DROP_CONFIG = 'drop0.8ch0.8'"
//    //" DROP_CONFIG = 'drop0.9ch0.9reg2000'"
//    " DROP_CONFIG = 'drop0.9ch0.9reg10000'"
//    " MODEL_DIMS = 'e512d65w4096'"
//    " make_char_dataset('D:/111enwiki9/enwik8')"
//    //" load_checkpoint(587000)\n"
//    " create_model(MPF_TAIL_LOSS, MPF_TUNE_FINAL_LAYER, MPF_TUNE_EMBED)"
//    " train()\n"
//    //" compute_exact_test(742000,50000)\n"
//    ;


//// enwik9 tests
//static TString TRAIN_SCRIPT =
//    " SAVE_MODEL = false"
//    " EVAL_INTERVAL = 1000"
//    " EVAL_BATCH_COUNT = 20"
//    " TRAIN_CONFIG = 'b8f1024'"
//    " DROP_CONFIG = 'drop1ch1'"
//    " MODEL_DIMS = 'e256d65w1024'"
//    " make_char_dataset('D:/111enwiki9/enwik9')"
//    " create_model(MPF_TAIL_LOSS, MPF_TUNE_FINAL_LAYER, MPF_TUNE_EMBED)"
//    " train()\n"
//    ;


//// arith test
//static TString TRAIN_SCRIPT =
//    " SAVE_MODEL = false"
//    " EVAL_INTERVAL = 1000"
//    " EVAL_BATCH_COUNT = 20"
//    " TRAIN_CONFIG = 'b64f64'"
//    " DROP_CONFIG = 'drop1ch1'"
//    " MODEL_DIMS = 'e256d65'"
//    " make_char_dataset('D:/111enwiki9/arith.txt')"
//    " create_model(MPF_TAIL_LOSS)"
//    " train()\n"
//    ;


//// train gpt2 small model size on owt
//static TString TRAIN_SCRIPT =
//    " MAX_ITERS = 2000000"
//    " EVAL_INTERVAL = 1000"
//    " EVAL_BATCH_COUNT = 20"
//    //" USE_PPM = true"
//    " TRAIN_CONFIG = 'a4b16f1024'"
//    " DROP_CONFIG = 'drop1ch1tail5'"
//    " MODEL_DIMS = 'e512tt128d86w1024'" // match 124M param model (gpt2-small, they do not count final layer) on OWT
//    //" MODEL_DIMS = 'e512tt256d45w1024'"
//    " set_vocab_size(50257)"
//    " load_tokenized_train('D:/111enwiki9/gpt2_train.bin')"
//    " load_tokenized_test('D:/111enwiki9/gpt2_test.bin')"
//    " create_model(MPF_TUNE_FINAL_LAYER, MPF_TUNE_EMBED)"
//    " train()\n"
//    ;


//// fed reference run
//static TString TRAIN_SCRIPT =
//    " SAVE_MODEL = false"
//    " EVAL_INTERVAL = 100"
//    " EVAL_BATCH_COUNT = 20"
//    //" TRAIN_CONFIG = 'b64f64'"
//    " TRAIN_CONFIG = 'b256f64'"
//    //" DROP_CONFIG = 'drop1ch1reg2000'"
//    //" DROP_CONFIG = 'drop1ch1'"
//    //" MODEL_DIMS = 'e256d65w64'"
//    //" MODEL_DIMS = 'e512tt128d65w64'"
//    //" MODEL_DIMS = 'e512tt128d250w64'"
//    //" MODEL_DIMS = 'e512tt256d135w64'"
//    //" MODEL_DIMS = 'e1024tt128d110w64'"
//    //" MODEL_DIMS = 'e1024tt192d75w64'"
//    //" MODEL_DIMS = 'e1024tt256d55w64'"
//    //" MODEL_DIMS = 'e1536tt128d60w64'"
//    //" MODEL_DIMS = 'e256tt128d130w64'"
//    //" MODEL_DIMS = 'e512tt128d35w64'"
//    //" MODEL_DIMS = 'e1024tt128d35w64'"
//    //" MODEL_DIMS = 'e1024tt128d65w64'"
//    //" MODEL_DIMS = 'e2048tt128d35w64'"
//    //" MODEL_DIMS = 'e512tt256d80w64'"
//    //" MODEL_DIMS = 'e2048tt128d65w64'"
//    //" MODEL_DIMS = 'e2048tt256d130w64'"
//    //" MODEL_DIMS = 'e2048tt384d90w64'"
//    //" MODEL_DIMS = 'e2048tt512d70w64'"
//    //" MODEL_DIMS = 'e4096tt256d55w64'"
//    //" MODEL_DIMS = 'e4096tt192d75w64'"
//    //" MODEL_DIMS = 'e1024tt256d280w64'"
//    " MODEL_DIMS = 'e4096tt512d150w64'"
//    //" load_tokenizer('D:/tokenizers/5k.bin')"
//    //" load_docset('D:/text/cultura_y/27.bin')" // 83M tokens
//    " load_tokenizer('d:/tokenizers/50k.bin')"
//    " load_indexed_docset_folder('D:/text/librusec/', 1)"
//    //" create_model(MPF_TUNE_FINAL_LAYER, MPF_TUNE_EMBED)"
//    " create_model(MPF_TUNE_FINAL_LAYER, MPF_TUNE_EMBED, MPF_COMBINE_LAYERS)"
//    //" load_checkpoint(460)\n"
//    " train()\n"
//    //" compute_exact_test(15300, 0000)\n"
//    ;


//// distributed ru model train
//static TString TRAIN_SCRIPT =
//    " DEVICE_COUNT = 1"
//    " MAX_ITERS = 1500000"
//    " EVAL_INTERVAL = 1000"
//    " EVAL_BATCH_COUNT = 5"
//    //" USE_PPM = true"
//    " TRAIN_CONFIG = 'a4b16f1024'"
//    //" TRAIN_CONFIG = 'b32f1024'"
//    " DROP_CONFIG = 'drop1ch1'"
//    //" DROP_CONFIG = 'drop1ch1tail5'"
//    //" MODEL_DIMS = 'e512tt128d65w1024'"
//    //" MODEL_DIMS = 'e512tt256d65w1024'"
//    " MODEL_DIMS = 'e1024tt256d65w1024'" // 420M
//    " load_tokenizer('d:/tokenizers/50k.bin')"
//    " load_indexed_docset_folder('D:/text/Gutenberg/', 1)"
//    " load_indexed_docset_folder('D:/text/open_web_text/', 1)"
//    " load_indexed_docset_folder('D:/text/librusec/', 1)"
//    " load_indexed_docset_folder('D:/text/cultura_y/', 1)"
//    " create_model(MPF_TAIL_LOSS, MPF_TUNE_FINAL_LAYER, MPF_TUNE_EMBED)"
//    " load_checkpoint(323000)\n"
//    " train()\n"
//    //" net_train('d:/workers_net.txt')\n"
//    //" compute_exact_test(56000)\n"
//    ;


//// index datasets
//static TString TRAIN_SCRIPT =
//    " USE_PPM = true"
//    " TEST_FRACTION = 0"
//    " load_tokenizer('d:/tokenizers/50k.bin')"
//    " index_docset_folder('D:/text/Gutenberg/')"
//    " index_docset_folder('D:/text/open_web_text/')"
//    " index_docset_folder('D:/text/librusec/')"
//    " index_docset_folder('D:/text/cultura_y/')"
//    ;


//static TString TRAIN_SCRIPT =
//    " make_char_dataset('D:/111enwiki9/wiki7_filter.txt')"
//    " check_cpu_gpu_match()\n"
//    ;



///////////////////////////////////////////////////////////////////////////////////////////////////
static void TestGradient(const TModelParams &params, const TTrainConfig &tc, TDataset &data)
{
    // have to remove gradient scaling and normalization for exact measurements
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
    MakeTrain(rng, dataBatch, NO_DROP, NO_DROP, pCtx.Get(), MAIN_DEVICE, &target);

    TVector<TVector<float>> predArr;
    pCtx->ComputeFragmentPredictions(&predArr);
    double loss = CalcTargetLoss(predArr, target);
    pCtx->Backprop(tc.Step, GRADIENT_APPLY);
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
        Scale(&g1, 1 / gradSum2, 0);
        Randomize(rng, &g1);

        const double STEP = 0.1;

        TModelParams chk = params;
        AddScaled(&chk, g1, (float)STEP, 0);

        pCtx->SetParams(chk);

        pCtx->ComputeFragmentPredictions(&predArr);
        double chkLoss = CalcTargetLoss(predArr, target);
        double trueDelta = (chkLoss - loss);
        double expectDelta = CalcDot(g1, grad) * STEP;
        DebugPrintf("loss %g, chk %g, trueDelta %g, expect %g\n", loss, chkLoss, trueDelta, expectDelta);
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
static void ComputeAverageModel(TModelParams *p, yint finishIter, yint iterInterval)
{
    TString pathTemplate = "eden_gpt_%.8gk.bin";

    // model averaging boosts perf on test significantly
    int startIter = finishIter - iterInterval;
    double modelCount = 1;
    TModelParams &sumParams = *p;
    Serialize(true, Sprintf(pathTemplate.c_str(), startIter / 1000.), sumParams);
    const int STEP = 1000;
    //const int STEP = 100;
    for (int iter = startIter + STEP; iter <= finishIter; iter += STEP) {
        TModelParams params;
        Serialize(true, Sprintf(pathTemplate.c_str(), iter / 1000.), params);
        AddScaled(&sumParams, params, 1, 1);
        modelCount += 1;
        printf(".");
    }
    printf("\n");
    Scale(&sumParams, 1 / modelCount, 1 / modelCount);
    //ComputeMatrixParamDistr(&startParams);
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// compute score on test set
static void ComputeExactTest(TDataset &data, const TModelParams &params)
{
    yint fragLen = params.ModelDim.FragLen;
    //yint testBatchSize = BUFFER_LEN / GetNodeCount(fragLen);
    //yint testBatchSize = 4;
    yint testBatchSize = 1;

    TIntrusivePtr<IModel> pModel = CreateModel(1, params);
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
        float testErr = CalcModelErr(batchArr, pCtx.Get()) * data.GetCompression();
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
    MakeTrain(rng, fragArr, tc.TokenDrop, tc.ChannelDrop, pCtx, MAIN_DEVICE, &batchTarget);
    pCtx->Backprop(tc.Step, GRADIENT_APPLY);

    TModelParams point2;
    pCtx->GetParams(&point2);

    for (yint testId = 0; testId < 5; ++testId) {
        pCtx->SetParams(point1);

        pCtx->Backprop(tc.Step, GRADIENT_APPLY);

        TModelParams chk;
        pCtx->GetParams(&chk);

        bool hasMismatch = false;
        if (!TestMatch(chk.LabelEmbed.GetMatrix(), point2.LabelEmbed.GetMatrix())) {
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
    // TVecFloat in gpt_cuda.cu must be fp16 for better match
    TXRng chkRng(1313);
    TModelParams params;
    yint vocabSize = data.GetVocabSize();
    //yint modelFlags = 0;
    yint modelFlags = MPF_TUNE_FINAL_LAYER | MPF_TUNE_EMBED;
    //TString modelDimStr = "e256d1w64";
    TString modelDimStr = "e256d6w64";
    TModelDim modelDim;
    InitModelDim(&modelDim, modelDimStr, ALIBI_V3, vocabSize, modelFlags);
    InitModel(&params, chkRng, modelDim, COMBINER_INIT_RANDOM, data.GetBias());
    //Serialize(true, "eden_gpt_0.bin", startParams);
    //Serialize(true, "eden_gpt_3k.bin", startParams);
    //startParams.LayerArr.resize(1);
    //startParams.ModelDim.Layers.resize(1);
    const yint CHECK_BATCH_SIZE = 1;
    const yint CHECK_FRAG_LEN = 64 - 1;
    const float CHECK_CHANNEL_DROP = 1;

    TIntrusivePtr<IModel> cpuModel = CreateModel(1, params);
    TIntrusivePtr<IComputeContext> cpuCtx = NCPU_GPT::CreateContext(cpuModel, CHECK_BATCH_SIZE * GetNodeCount(tc.TrainFragLen));

    TIntrusivePtr<IModel> gpuModel = CreateModel(1, params);
    TIntrusivePtr<IComputeContext> gpuCtx = NCUDA_GPT::CreateContext(gpuModel, CHECK_BATCH_SIZE * GetNodeCount(tc.TrainFragLen));

    TFragment frag;
    data.MakeFragment(TDataset::TRAIN, chkRng, CHECK_FRAG_LEN, &frag);
    TVector<TFragment> xxFrag;
    xxFrag.push_back(frag);

    MakeTest(xxFrag, cpuCtx.Get(), MAIN_DEVICE);
    MakeTest(xxFrag, gpuCtx.Get(), MAIN_DEVICE);

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
    TTrainingStep largeStep = tc.Step;
    largeStep.ScaleRate(10);
    MakeTrain(cpuRng, xxFrag, tc.TokenDrop, CHECK_CHANNEL_DROP, cpuCtx.Get(), MAIN_DEVICE);
    cpuCtx->Backprop(largeStep, GRADIENT_APPLY);
    cpuCtx->ComputeFragmentPredictions(&cpuPredArr);

    MakeTrain(gpuRng, xxFrag, tc.TokenDrop, CHECK_CHANNEL_DROP, gpuCtx.Get(), MAIN_DEVICE);
    gpuCtx->Backprop(largeStep, GRADIENT_APPLY);
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
void TrainModel(yint startIteration, yint deviceCount, const TTrainContext &trainCtx, TIntrusivePtr<TModelParamsHolder> pParams)
{
    const TTrainConfig &tc = trainCtx.GetConfig();

#ifdef _MSC_VER
    SetConsoleCP(CP_UTF8);
    SetConsoleOutputCP(CP_UTF8);
    SetProcessAffinityMask(GetCurrentProcess(), 0xffff); // use P-cores, mask is cpu dependent
#endif

    // create model
    TIntrusivePtr<IModel> pModel = CreateModel(deviceCount, pParams->Params);
    pParams = 0;
    //TIntrusivePtr<IComputeContext> pCtx = NCPU_GPT::CreateContext(pModel, trainCtx.GetMaxNodeCount());
    TIntrusivePtr<IComputeContext> pCtx = NCUDA_GPT::CreateContext(pModel, trainCtx.GetMaxNodeCount());

    //TOFStream fTrainLog("train_log.txt");
    NHPTimer::STime tStart;
    NHPTimer::GetTime(&tStart);
    for (yint iter = startIteration; iter <= trainCtx.GetMaxIters(); ++iter) {
        if ((iter % trainCtx.GetEvalInterval()) == 0) {
            if (trainCtx.IsSaveModel()) {
                TModelParams params;
                pCtx->GetParams(&params);
                Serialize(false, Sprintf("eden_gpt_%.8gk.bin", iter / 1000.), params);
            }
            float trainErr = CalcModelErr(trainCtx.GetScoreTrainBatches(), pCtx.Get()) * trainCtx.GetCompression();
            float testErr = CalcModelErr(trainCtx.GetScoreTestBatches(), pCtx.Get()) * trainCtx.GetCompression();
            if (testErr != 0) {
                DebugPrintf("iter %.8gk, %g sec, train err %g, test err %g\n", iter / 1000., NHPTimer::GetTimePassed(&tStart), trainErr, testErr); fflush(0);
            } else {
                DebugPrintf("iter %.8gk, %g sec, train err %g\n", iter / 1000., NHPTimer::GetTimePassed(&tStart), trainErr); fflush(0);
            }
            //fTrainLog << trainErr << "\t" << testErr << Endl;
        }

        // accumulate several batches
        EAddToModel addToModel = tc.DoAccumulate(iter) ? GRADIENT_ACCUMULATE : GRADIENT_APPLY;

        // generate train fragments
        TXRng iterRng(iter);
        for (yint deviceId = 0; deviceId < deviceCount; ++deviceId) {
            TVector<TFragment> fragArr;
            trainCtx.MakeTrainBatches(iterRng, &fragArr);
            MakeTrain(iterRng, fragArr, tc.TokenDrop, tc.ChannelDrop, pCtx.Get(), deviceId);
        }
        pCtx->Backprop(trainCtx.GetStep(iter), addToModel);

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


///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////
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


class TTrainScriptParser : public TTrainDataConfigParser
{
    yint DeviceCount = 1;
    yint StartIteration = 0;
    bool SaveModel = true;
    yint MaxIters = 2000000;
    yint EvalInterval = 1000;
    yint EvalBatchCount = 20;

private:
    void ParseScriptOp(const TConfigFile::TOp &op) override
    {
        if (op.Op == CFG_OP_ASSIGNMENT) {
            if (op.Dst == "MAX_ITERS") {
                MaxIters = atof(op.Args[0].c_str());
            } else if (op.Dst == "DEVICE_COUNT") {
                DeviceCount = atof(op.Args[0].c_str());
                Y_VERIFY(DeviceCount >= 1 && DeviceCount < 100);
            } else if (op.Dst == "EVAL_INTERVAL") {
                EvalInterval = atof(op.Args[0].c_str());
            } else if (op.Dst == "EVAL_BATCH_COUNT") {
                EvalBatchCount = atof(op.Args[0].c_str());
            } else if (op.Dst == "SAVE_MODEL") {
                SaveModel = (IsYes(op.Args[0]));
            } else {
                DebugPrintf("unknown config variable %s\n", op.Dst.c_str());
            }

        } else if (op.Op == CFG_OP_CALL) {
            if (op.Dst == "load_checkpoint") {
                Y_VERIFY(YSize(op.Args) == 1);
                StartIteration = atoi(op.Args[0].c_str());
                DebugPrintf("Load checkpoint %gk\n", StartIteration / 1000.);
                Data.StartParams = new TModelParamsHolder();
                Serialize(true, Sprintf("eden_gpt_%.8gk.bin", StartIteration / 1000.), Data.StartParams->Params);
                Y_VERIFY(!Data.StartParams->Params.IsEmpty());

            // process ops
            } else if (op.Dst == "train" || op.Dst == "net_train") {
                Y_VERIFY(!Data.StartParams->Params.IsEmpty());
                Data.FinishDatasetBuild();
                Data.VerifyVocabSize();
                TTrainConfig tc(TrainConfig, DropConfig);
                TTrainContext trainCtx(Data.Data, tc, SaveModel, MaxIters, EvalInterval);

                DebugPrintf("%s %s %s 0x%x, size %gM\n",
                    GetModelDimsString(Data.StartParams->Params.GetModelDim()).c_str(),
                    tc.GetTrainConfig().c_str(),
                    tc.GetDropConfig().c_str(),
                    (int)Data.StartParams->Params.ModelDim.Flags,
                    CountModelSize(Data.StartParams->Params) / 1000000.);

                // create batches for train & test score compute, can use different sizes
                const yint batchSize = tc.TrainBatchSize;
                const yint fragLen = tc.TrainFragLen;
                trainCtx.MakeScoreBatches(EvalBatchCount, batchSize, fragLen);

                // keep train params
                Data.StartParams->Params.ModelDim.FragLen = tc.TrainFragLen;

                if (op.Dst == "train") {
                    TrainModel(StartIteration, DeviceCount, trainCtx, Data.StartParams.Release());
                } else if (op.Dst == "net_train") {
                    Y_VERIFY(YSize(op.Args) == 1);
                    TVector<TString> workerArr;
                    ReadNonEmptyLines(&workerArr, op.Args[0]);
                    NNetTrain::RunMaster(StartIteration, DeviceCount, workerArr, trainCtx, Data.StartParams.Release());
                } else {
                    Y_ASSERT(0);
                }

            } else if (op.Dst == "compute_exact_test") {
                Data.FinishDatasetBuild();
                TModelParams params;
                if (op.Args.empty()) {
                    params = Data.StartParams->Params;
                } else {
                    yint finishIter = atoi(op.Args[0].c_str());
                    yint iterInterval = YSize(op.Args) > 1 ? atoi(op.Args[1].c_str()) : 0;
                    ComputeAverageModel(&params, finishIter, iterInterval);
                }
                ComputeExactTest(Data.Data, params);

            } else if (op.Dst == "check_cpu_gpu_match") {
                Data.FinishDatasetBuild();
                TTrainConfig tc(TrainConfig, DropConfig);
                CheckCpuGpuMatch(tc, Data.Data);

            } else if (op.Dst == "test_gradient") {
                Data.FinishDatasetBuild();
                TTrainConfig tc(TrainConfig, DropConfig);
                TestGradient(Data.StartParams->Params, tc, Data.Data);

            } else {
                DebugPrintf("unknown function %s\n", op.Dst.c_str());
                abort();
            }
        }
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
extern yint MatrixAddWorkerThreadCount;

void TestMatMul();
//void Repack();

int main(int argc, char **argv)
{
    //TestMatMul();
    //Repack();
    //GenerateArithmetic();
    //GenerateArithmetic97();
    //NBinClass::Run();
    //NFedSim::Run();
    //NCPUInfer::Check();
    //return 0;

    TOpt cmdline("c:w:t:", argc, argv);
    for (const TOpt::TParam &param : cmdline.Params) {
        if (param.Name == "c") {
            DebugPrintf("Executing script %s\n", param.Args[0].c_str());
            TVector<char> cfg;
            Y_VERIFY(ReadWholeFile(param.Args[0], &cfg));
            TRAIN_SCRIPT = cfg.data();
        } else if (param.Name == "w") {
            NNetTrain::RunWorker(atoi(param.Args[0].c_str()));
            return 0;
        } else if (param.Name == "t") {
            MatrixAddWorkerThreadCount = atoi(param.Args[0].c_str());
        }
    }

    // execute config script
    TTrainScriptParser gpt;
    gpt.ParseScript(TRAIN_SCRIPT);

    return 0;
}
