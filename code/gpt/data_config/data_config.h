#pragma once
#include <gpt/data/data.h>
#include <gpt/model_params/model_params.h>
#include <gpt/att/sliding_window.h>
#include <lib/config/cfg_file.h>
#include <lib/config/config.h>


struct TTrainDataConfig
{
    TDataset Data;
    TTokenizer Tokenizer;
    TIntrusivePtr<TModelParamsHolder> StartParams;
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
            } else if (flag == "MPF_SIM_QUANT_4BIT") {
                modelFlags |= MPF_SIM_QUANT_4BIT;
            } else if (flag == "MPF_GROK_BINARY_OP") {
                modelFlags |= MPF_GROK_BINARY_OP;
            } else if (flag == "MPF_COMBINE_LAYERS") {
                modelFlags |= MPF_COMBINE_LAYERS;
            } else if (flag == "MPF_MLM_BERT") {
                modelFlags |= MPF_MLM_BERT;
            } else {
                DebugPrintf("unknown model flag %s\n", flag.c_str());
            }
        }
        // initialize model params
        StartParams = new TModelParamsHolder();
        TModelDim modelDim;
        InitModelDim(&modelDim, modelDimsString, ALIBI_V3, VocabSize, modelFlags);
        InitModel(&StartParams->Params, rng, modelDim, COMBINER_INIT_ZERO, Data.GetBias());
        if (OverrideDocStartToken >= 0) {
            StartParams->Params.ModelDim.SetDocStartToken(OverrideDocStartToken);
        } else if (Tokenizer.HasDocStartToken()) {
            StartParams->Params.ModelDim.SetDocStartToken(Tokenizer.GetDocStartToken());
        } else {
            Y_ASSERT(!StartParams->Params.ModelDim.HasFlag(MPF_USE_DOC_START_TOKEN));
        }
    }

    void CreateDatasetBuilders(bool usePPM)
    {
        Y_VERIFY(!Tokenizer.IsEmpty());
        Y_VERIFY(StartParams == nullptr);
        if (DataBuild.Get() == 0) {
            DataBuild = new TDatasetBuilder(&Data, usePPM, Tokenizer);
        }
    }

    void CreateDatasetBuilders(yint vocabSize, bool usePPM)
    {
        Y_VERIFY(StartParams == nullptr);
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
        Y_VERIFY(StartParams->Params.ModelDim.VocabSize == VocabSize);
        Y_VERIFY(Data.GetVocabSize() == VocabSize);
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
struct TTrainDataConfigParser
{
    TTrainDataConfig Data;
    // params
    bool UsePPM = false;
    float TestFraction = 0.05f;
    TString ModelDimsString = "e256d65";
    TString TrainConfig = "b64f64";
    TString DropConfig = "drop1ch1";


public:
    void ParseScript(const TString &configText);
    virtual void ParseScriptOp(const TConfigFile::TOp &op) {}
};
