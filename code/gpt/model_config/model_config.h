#pragma once
#include <gpt/data/data.h>
#include <gpt/model_params/model_params.h>
#include <lib/config/cfg_file.h>
//#include <lib/config/config.h>



///////////////////////////////////////////////////////////////////////////////////////////////////
struct TTrainModelConfigParser
{
    TIntrusivePtr<TModelParamsHolder> StartParams;
    // params
    TString ModelDimsString = "e256d65";
    TString TrainConfig = "b64f64";
    TString DropConfig = "drop1ch1";

public:
    bool ParseScriptOp(const TConfigFile::TOp &op, TIntrusivePtr<IDataSource> data);
};
