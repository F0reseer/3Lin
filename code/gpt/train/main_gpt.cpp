#include "stdafx.h"
#include "train.h"
#include "net_train.h"
#include "mmlu_score.h"
#include <gpt/data/data.h>
#include <gpt/data/bpe.h>
#include <gpt/att/sliding_window.h>
#include <gpt/compute/gpt_cpu.h>
#include <gpt/compute/gpt_cuda.cuh>
#include <gpt/data_config/data_config.h>
#include <gpt/model_config/model_config.h>
#include <lib/random/rand_utils.h>
#include <lib/hp_timer/hp_timer.h>
#include <lib/config/config.h>


static TString TRAIN_SCRIPT = "";

#ifdef _win_
static TString WorkingFolder = "d:/";
#else
static TString WorkingFolder = "";
#endif


///////////////////////////////////////////////////////////////////////////////////////////////////
static void TestGradient(const TModelParams &params, const TTrainConfig &tc, TIntrusivePtr<IDataSource> data)
{
    // have to remove gradient scaling and normalization for exact measurements
    TXRng rng(1313);

    TVector<TFragment> dataBatch;
    ui64 rngSeed = 31313;
    data->SampleFragments(IDataSource::TRAIN, rngSeed, 1, tc.TrainFragLen, &dataBatch);

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
    TString pathTemplate = WorkingFolder + "eden_gpt_%.8gk.bin";
    //TString pathTemplate = WorkingFolder + "models/fed_small/model_%.8g.bin ";

    // model averaging boosts perf on test significantly
    int startIter = finishIter - iterInterval;
    double modelCount = 1;
    TModelParams &sumParams = *p;
    Serialize(IO_READ, Sprintf(pathTemplate.c_str(), startIter / 1000.), sumParams);
    const int STEP = 1000;
    //const int STEP = 100;
    for (int iter = startIter + STEP; iter <= finishIter; iter += STEP) {
        TModelParams params;
        Serialize(IO_READ, Sprintf(pathTemplate.c_str(), iter / 1000.), params);
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
static void ComputeExactTest(TIntrusivePtr<IDataSource> data, const TModelParams &params)
{
    yint fragLen = params.ModelDim.FragLen;
    //yint testBatchSize = BUFFER_LEN / GetNodeCount(fragLen);
    //yint testBatchSize = 4;
    yint testBatchSize = 1;

    TIntrusivePtr<IModel> pModel = CreateModel(1, params);
    TIntrusivePtr<IComputeContext> pCtx = NCUDA_GPT::CreateContext(pModel, testBatchSize * GetNodeCount(fragLen));
    double sumTestErr = 0;
    double sumCount = 0;
    int rngSeed = 31331;
    for (yint iter = 1;; ++rngSeed, ++iter) {
        TVector<TFragment> batchArr;
        data->SampleFragments(IDataSource::TEST, rngSeed, testBatchSize, fragLen, &batchArr);
        float testErr = CalcModelErr(batchArr, pCtx.Get()) * data->GetStats().Compression;
        if (isnan(testErr)) {
            DebugPrintf("rseed %g, score is nan\n", rngSeed * 1.);
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
void CheckCpuGpuMatch(const TTrainConfig &tc, TIntrusivePtr<IDataSource> data)
{
    // TVecFloat in gpt_cuda.cu must be fp16 for better match
    TXRng chkRng(1313);
    TModelParams params;
    yint vocabSize = data->GetStats().VocabSize;
    //yint modelFlags = 0;
    yint modelFlags = MPF_TUNE_FINAL_LAYER | MPF_TUNE_EMBED;
    //TString modelDimStr = "e256d1w64";
    //TString modelDimStr = "e256d6w64";
    TString modelDimStr = "e512h2d6w64";
    TModelDim modelDim;
    InitModelDim(&modelDim, modelDimStr, ALIBI_V3, vocabSize, modelFlags);
    InitModel(&params, chkRng, modelDim, COMBINER_INIT_RANDOM, data->GetStats().Bias);
    //Serialize(true, WorkingFolder + "eden_gpt_0.bin", startParams);
    //Serialize(true, WorkingFolder + "eden_gpt_3k.bin", startParams);
    //startParams.LayerArr.resize(1);
    //startParams.ModelDim.Layers.resize(1);
    const yint CHECK_BATCH_SIZE = 1;
    const yint CHECK_FRAG_LEN = 64 - 1;
    const float CHECK_CHANNEL_DROP = 1;

    TIntrusivePtr<IModel> cpuModel = CreateModel(1, params);
    TIntrusivePtr<IComputeContext> cpuCtx = NCPU_GPT::CreateContext(cpuModel, CHECK_BATCH_SIZE * GetNodeCount(tc.TrainFragLen));

    TIntrusivePtr<IModel> gpuModel = CreateModel(1, params);
    TIntrusivePtr<IComputeContext> gpuCtx = NCUDA_GPT::CreateContext(gpuModel, CHECK_BATCH_SIZE * GetNodeCount(tc.TrainFragLen));

    TVector<TFragment> fragArr;
    data->SampleFragments(IDataSource::TRAIN, 1313, 1, CHECK_FRAG_LEN, &fragArr);

    MakeTest(fragArr, cpuCtx.Get(), MAIN_DEVICE);
    MakeTest(fragArr, gpuCtx.Get(), MAIN_DEVICE);

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
    MakeTrain(cpuRng, fragArr, tc.TokenDrop, CHECK_CHANNEL_DROP, cpuCtx.Get(), MAIN_DEVICE);
    cpuCtx->Backprop(largeStep, GRADIENT_APPLY);
    cpuCtx->ComputeFragmentPredictions(&cpuPredArr);

    MakeTrain(gpuRng, fragArr, tc.TokenDrop, CHECK_CHANNEL_DROP, gpuCtx.Get(), MAIN_DEVICE);
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

    //TOFStream fTrainLog(WorkingFolder + "train_log.txt");
    NHPTimer::STime tStart;
    NHPTimer::GetTime(&tStart);
    for (yint iter = startIteration; iter <= trainCtx.GetMaxIters(); ++iter) {
        if ((iter % trainCtx.GetEvalInterval()) == 0) {
            if (trainCtx.IsSaveModel()) {
                TModelParams params;
                pCtx->GetParams(&params);
                Serialize(IO_WRITE, Sprintf((WorkingFolder + "eden_gpt_%.8gk.bin").c_str(), iter / 1000.), params);
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
        ui64 rngSeed = (iter + 0xbadf00d) * 0x39ef28172812ull;
        TVector<TVector<TFragment>> fragArr;
        trainCtx.SampleTrainBatches(rngSeed, deviceCount, &fragArr);
        for (yint deviceId = 0; deviceId < deviceCount; ++deviceId) {
            MakeTrain(iterRng, fragArr[deviceId], tc.TokenDrop, tc.ChannelDrop, pCtx.Get(), deviceId);
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


class TTrainScriptParser
{
    TTrainModelConfigParser TrainCfg;
    yint DeviceCount = 1;
    yint StartIteration = 0;
    bool SaveModel = true;
    yint MaxIters = 2000000;
    yint EvalInterval = 1000;
    yint EvalBatchCount = 20;

public:
    void ParseScript(const TVector<TConfigFile::TOp> &opArr, yint *pOpPtr, TIntrusivePtr<IDataSource> data)
    {
        Y_VERIFY(data.Get() != 0);
        for (yint &ptr = *pOpPtr; ptr < YSize(opArr); ++ptr) {
            const TConfigFile::TOp &op = opArr[ptr];
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
                } else if (TrainCfg.ParseScriptOp(op, data)) {
                    ;
                } else {
                    DebugPrintf("unknown config variable %s\n", op.Dst.c_str());
                }

            } else if (op.Op == CFG_OP_CALL) {
                if (op.Dst == "load_checkpoint") {
                    Y_VERIFY(YSize(op.Args) == 1);
                    StartIteration = atoi(op.Args[0].c_str());
                    DebugPrintf("Load checkpoint %gk\n", StartIteration / 1000.);
                    TrainCfg.StartParams = new TModelParamsHolder();
                    Serialize(IO_READ, Sprintf((WorkingFolder + "eden_gpt_%.8gk.bin").c_str(), StartIteration / 1000.), TrainCfg.StartParams->Params);
                    Y_VERIFY(!TrainCfg.StartParams->Params.IsEmpty());

                    // process ops
                } else if (op.Dst == "train" || op.Dst == "net_train") {
                    Y_VERIFY(!TrainCfg.StartParams->Params.IsEmpty());
                    TTrainConfig tc(TrainCfg.TrainConfig, TrainCfg.DropConfig);
                    TTrainContext trainCtx(data, tc, SaveModel, MaxIters, EvalInterval);

                    DebugPrintf("%s %s %s 0x%x, size %gM\n",
                        GetModelDimsString(TrainCfg.StartParams->Params.GetModelDim()).c_str(),
                        tc.GetTrainConfig().c_str(),
                        tc.GetDropConfig().c_str(),
                        (int)TrainCfg.StartParams->Params.ModelDim.Flags,
                        CountModelSize(TrainCfg.StartParams->Params) / 1000000.);

                    // create batches for train & test score compute, can use different sizes
                    const yint batchSize = tc.TrainBatchSize;
                    const yint fragLen = tc.TrainFragLen;
                    trainCtx.MakeScoreBatches(EvalBatchCount, batchSize, fragLen);

                    // keep train params
                    TrainCfg.StartParams->Params.ModelDim.FragLen = tc.TrainFragLen;

                    if (op.Dst == "train") {
                        TrainModel(StartIteration, DeviceCount, trainCtx, TrainCfg.StartParams.Release());
                    } else if (op.Dst == "net_train") {
                        Y_VERIFY(YSize(op.Args) == 1);
                        TVector<TString> workerArr;
                        ReadNonEmptyLines(&workerArr, op.Args[0]);
                        NNetTrain::RunMaster(StartIteration, DeviceCount, workerArr, trainCtx, TrainCfg.StartParams.Release());
                    } else {
                        Y_ASSERT(0);
                    }

                } else if (op.Dst == "compute_exact_test") {
                    TModelParams params;
                    if (op.Args.empty()) {
                        params = TrainCfg.StartParams->Params;
                    } else {
                        yint finishIter = atoi(op.Args[0].c_str());
                        yint iterInterval = YSize(op.Args) > 1 ? atoi(op.Args[1].c_str()) : 0;
                        ComputeAverageModel(&params, finishIter, iterInterval);
                    }
                    ComputeExactTest(data, params);

                } else if (op.Dst == "compute_choice_score") {
                    Y_VERIFY(YSize(op.Args) == 1);
                    ComputeChoiceScore(TrainCfg.StartParams->Params, op.Args[0]);

                } else if (op.Dst == "check_cpu_gpu_match") {
                    TTrainConfig tc(TrainCfg.TrainConfig, TrainCfg.DropConfig);
                    CheckCpuGpuMatch(tc, data);

                } else if (op.Dst == "test_gradient") {
                    TTrainConfig tc(TrainCfg.TrainConfig, TrainCfg.DropConfig);
                    TestGradient(TrainCfg.StartParams->Params, tc, data);

                } else if (TrainCfg.ParseScriptOp(op, data)) {

                } else {
                    DebugPrintf("unknown function %s\n", op.Dst.c_str());
                    abort();
                }
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
    //return 0;

    TOpt cmdline("s:w:t:", argc, argv);
    for (const TOpt::TParam &param : cmdline.Params) {
        if (param.Name == "s") {
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
    TConfigFile cfg;
    ParseConfig(&cfg, TRAIN_SCRIPT);
    yint ptr = 0;
    TDataSourceConfigParser dataCfg;
    dataCfg.ParseScript(cfg.OpArr, &ptr);
    if (ptr == YSize(cfg.OpArr)) {
        return 0;
    }

    TTrainScriptParser trainScript;
    trainScript.ParseScript(cfg.OpArr, &ptr, dataCfg.GetDataset());

    return 0;
}
