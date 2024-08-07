#include "stdafx.h"
#include "mmlu_score.h"
#include <gpt/data/data.h>
#include <gpt/compute/gpt_cuda.cuh>
#include <gpt/att/sliding_window.h>


static bool ReadVec(TFileStream &f, TVector<ui16> *p)
{
    ui16 len = 0;
    if (f.Read(&len, 2) != 2) {
        return false;
    }
    p->resize(len);
    f.Read(p->data(), len * 2);
    return true;
}


struct TChoiceSample
{
    TVector<TFragment> FragArr;
    yint Correct = 0;
    yint CtxSize = 0;

    yint GetNodesRequired() const
    {
        yint res = 0;
        for (const TFragment &frag : FragArr) {
            res += YSize(frag.Text) + 1;
        }
        return res;
    }
};


void ComputeChoiceScore(const TModelParams &params, const TString &queryFile)
{
    TFileStream f(IO_READ, queryFile);
    Y_VERIFY(f.IsValid() && "file not found");

    constexpr yint MAX_NODE_COUNT = 16384;
    TIntrusivePtr<IModel> pModel = CreateModel(1, params);
    TIntrusivePtr<IComputeContext> pCtx = NCUDA_GPT::CreateContext(pModel, MAX_NODE_COUNT);
    pCtx->SetParams(params);

    TVector<TChoiceSample> allSamples;
    for (;;) {
        TVector<ui16> ctx;
        if (!ReadVec(f, &ctx)) {
            break;
        }
        TChoiceSample sample;
        sample.CtxSize = YSize(ctx);
        ui16 num_cont = 0;
        f.Read(&num_cont, 2);
        for (yint k = 0; k < num_cont; ++k) {
            TVector<ui16> cont;
            ReadVec(f, &cont);
            // make fragment
            TFragment frag;
            for (TBPEToken x : ctx) {
                frag.Text.push_back(x);
            }
            for (TBPEToken x : cont) {
                frag.Text.push_back(x);
            }
            Y_VERIFY(!frag.Text.empty());
            sample.FragArr.push_back(frag);
        }
        ui16 correct = 0;
        f.Read(&correct, 2);
        sample.Correct = correct;
        Y_VERIFY(!f.IsFailed() && "file corrupted");
        allSamples.push_back(sample);
    }

    double totalSamples = 0;
    double correctSamples = 0;
    yint sampleId = 0;
    while (sampleId < YSize(allSamples)) {
        TVector<TFragment> batchFragArr;
        yint begSampleId = sampleId;
        yint nodeCount = 0;
        while (sampleId < YSize(allSamples)) {
            const TChoiceSample &s = allSamples[sampleId];
            nodeCount += s.GetNodesRequired();
            if (nodeCount > MAX_NODE_COUNT) {
                break;
            }
            batchFragArr.insert(batchFragArr.end(), s.FragArr.begin(), s.FragArr.end());
            ++sampleId;
            //break;
        }

        MakeTest(batchFragArr, pCtx.Get(), MAIN_DEVICE);
        TVector<TVector<float>> predArr;
        pCtx->ComputeFragmentPredictions(&predArr);

        yint ptr = 0;
        for (yint x = begSampleId; x < sampleId; ++x) {
            const TChoiceSample &s = allSamples[x];

            yint topChoice = 0;
            double topScore = -1e38;
            for (yint k = 0; k < YSize(s.FragArr); ++k) {
                const TFragment &frag = s.FragArr[k];
                double sumLoss = 0;
                double count = 0;
                for (yint t = s.CtxSize; t < YSize(frag.Text); ++t) {
                    sumLoss += log(predArr[ptr + t][frag.Text[t]]);
                    count += 1;
                }
                double score = sumLoss / count;
                if (score > topScore) {
                    topScore = score;
                    topChoice = k;
                }
                ptr += YSize(frag.Text) + 1;
            }

            if (topChoice == s.Correct) {
                correctSamples += 1;
                printf("+"); fflush(0);
            } else {
                printf("."); fflush(0);
            }
            totalSamples += 1;
        }
    }
    printf("\n");
    DebugPrintf("%g%% correct\n", correctSamples * 100. / totalSamples);
    fflush(0);
}
