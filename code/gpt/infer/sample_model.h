#pragma once
#include <gpt/data/bpe.h>
#include <gpt/model_params/model_params.h>
#include <gpt/compute/gpt_cuda.cuh>
#include <gpt/att/sliding_window.h>


struct TSamplingModel
{
    TIntrusivePtr<IModel> Model;
    TIntrusivePtr<IComputeContext> Ctx;
    TTokenizer Tokenizer;
    yint MaxLen = 0;
    bool UsePPM = false;

    void Init(const TModelParams &params, const TTokenizer &tokenizer)
    {
        Tokenizer = tokenizer;
        Model = CreateModel(1, params);
        MaxLen = params.ModelDim.FragLen;
        Ctx = NCUDA_GPT::CreateContext(Model, GetNodeCount(MaxLen));
        UsePPM = params.ModelDim.HasFlag(MPF_PPM);
    }
};

TString SampleFromModel(TXRng &rng, TSamplingModel &model, const TString &prefix);
//TString GenerateFromModel(TXRng &rng, TTrainContext *pTrainCtx, const TModelParams &params, yint genLen, yint limitWindow, yint fragLen);
//TString BeamSampleFromModel(TXRng &rng, TTrainContext *pTrainCtx, const TModelParams &params, yint genLen, yint limitWindow, yint fragLen);
