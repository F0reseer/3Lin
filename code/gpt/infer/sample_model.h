#pragma once
#include <gpt/data/bpe.h>
#include <gpt/model/model_params.h>
#include <gpt/model/gpt_cuda.cuh>
#include <gpt/att/sliding_window.h>


struct TSamplingModel
{
    TIntrusivePtr<IModel> Model;
    TIntrusivePtr<IComputeContext> Ctx;
    TTokenizer Tokenizer;
    TWindowSizeLimit Window;
    yint MaxLen = 0;
    bool UsePPM = false;

    TSamplingModel(const TModelParams &params, const TTokenizer &tokenizer) : Tokenizer(tokenizer)
    {
        Model = CreateModel(1, params);
        Window = params.ModelDim.Window;
        MaxLen = params.ModelDim.FragLen;
        Ctx = NCUDA_GPT::CreateContext(Model, GetNodeCount(MaxLen));
        UsePPM = params.ModelDim.HasFlag(MPF_PPM);
    }
};

TString SampleFromModel(TXRng &rng, TSamplingModel &model, const TString &prefix);
//TString GenerateFromModel(TXRng &rng, TTrainContext *pTrainCtx, const TModelParams &params, yint genLen, yint limitWindow, yint fragLen);
//TString BeamSampleFromModel(TXRng &rng, TTrainContext *pTrainCtx, const TModelParams &params, yint genLen, yint limitWindow, yint fragLen);
