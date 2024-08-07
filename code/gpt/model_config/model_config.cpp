#include "stdafx.h"
#include "model_config.h"
#include <gpt/att/sliding_window.h>


static TIntrusivePtr<TModelParamsHolder> CreateModel(const TConfigFile::TOp &op, const TString &modelDimsString, const IDataSource::TDataStats &stats)
{
    TXRng rng(1313);

    Y_VERIFY(stats.VocabSize > 0 && "unknown vocab size");
    ui64 modelFlags = 0;
    if (stats.UsePPM) {
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
        } else if (flag == "MPF_ABS_POSITIONS") {
            modelFlags |= MPF_ABS_POSITIONS;
        } else {
            DebugPrintf("unknown model flag %s\n", flag.c_str());
        }
    }
    // initialize model params
    TIntrusivePtr<TModelParamsHolder> res = new TModelParamsHolder();
    TModelDim modelDim;
    InitModelDim(&modelDim, modelDimsString, ALIBI_V3, stats.VocabSize, modelFlags);
    InitModel(&res->Params, rng, modelDim, COMBINER_INIT_ZERO, stats.Bias);
    //InitModel(&res->Params, rng, modelDim, COMBINER_INIT_RANDOM, stats.Bias);
    if (stats.DocStartToken>= 0) {
        res->Params.ModelDim.SetDocStartToken(stats.DocStartToken);
    } else {
        Y_ASSERT(!res->Params.ModelDim.HasFlag(MPF_USE_DOC_START_TOKEN));
    }
    return res;
}


bool TTrainModelConfigParser::ParseScriptOp(const TConfigFile::TOp &op, TIntrusivePtr<IDataSource> data)
{
    if (op.Op == CFG_OP_ASSIGNMENT) {
        if (op.Dst == "TRAIN_CONFIG") {
            TrainConfig = op.Args[0];
        } else if (op.Dst == "DROP_CONFIG") {
            DropConfig = op.Args[0];
        } else if (op.Dst == "MODEL_DIMS") {
            Y_VERIFY(StartParams == nullptr && "model dimenstion are useless, model already created");
            ModelDimsString = op.Args[0];
        } else {
            return false;
        }

    } else if (op.Op == CFG_OP_CALL) {
    // model ops
        if (op.Dst == "create_model") {
            StartParams = CreateModel(op, ModelDimsString, data->GetStats());

        } else if (op.Dst == "load_model") {
            Y_VERIFY(YSize(op.Args) == 1);
            DebugPrintf("Load model %s\n", op.Args[0].c_str());
            StartParams = new TModelParamsHolder();
            Serialize(IO_READ, op.Args[0], StartParams->Params);
            Y_VERIFY(!StartParams->Params.IsEmpty());

        } else if (op.Dst == "load_fed_model16") {
            Y_VERIFY(YSize(op.Args) == 1);
            DebugPrintf("Load int8 model %s\n", op.Args[0].c_str());
            StartParams = new TModelParamsHolder();
            TFileStream f(IO_READ, op.Args[0]);
            TBufferedStream bufIO(IO_READ, f);
            float weight = 0;
            bufIO.Read(&weight, sizeof(weight));
            UnpackModelParams(&StartParams->Params, bufIO);
            Y_VERIFY(!StartParams->Params.IsEmpty());

        } else {
            return false;
        }
    }
    return true;
}
