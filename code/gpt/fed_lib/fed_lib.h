#pragma once
#include <lib/guid/guid.h>
#include <gpt/data/data.h>
#include <gpt/model_params/model_params.h>
#include <gpt/train_config/train_config.h>
#include <util/fast_io.h>
#include <util/mem_io.h>


extern TGuid FedToken;

const yint FED_HTTP_PORT = 18181;
const yint FED_GRAD_PORT = 18182;
const yint FED_DATA_PORT = 18183;


// have to scale delta to get convergence with lagged updates
const float FED_WEIGHT_SCALE = 0.5;


struct TFedParams
{
    TTrainConfig Config;
    float Compression = 0;
    SAVELOAD(Config, Compression);
};


///////////////////////////////////////////////////////////////////////////////////////////////////
struct TWeightedModelParamsPkt
{
    TMemStream Pkt;

    TWeightedModelParamsPkt() {}
    TWeightedModelParamsPkt(TVector<ui8> *data) : Pkt(data) {}
    void Swap(TVector<ui8> *data)
    {
        Pkt.Swap(data);
    }
    void Swap(TWeightedModelParamsPkt &x)
    {
        Pkt.Swap(x.Pkt);
    }
    void Read(const TString &filename)
    {
        TFileStream f(true, filename);
        TVector<ui8> buf;
        buf.resize(f.GetLength());
        f.Read(buf.data(), YSize(buf));
        Swap(&buf);
    }
    void Write(const TString &filename)
    {
        TFileStream f(false, filename);
        TVector<ui8> buf;
        Swap(&buf);
        f.Write(buf.data(), YSize(buf));
        Swap(&buf);
    }
};


void PackModelParams(TWeightedModelParamsPkt *p, TModelParams &params, float weight);
void UnpackModelParams(TModelParams *pParams, TWeightedModelParamsPkt &pkt);
void SetWeight(TWeightedModelParamsPkt &pkt, float weight);
float GetWeight(TWeightedModelParamsPkt &pkt);
void AddPackedModelParamsScaled(TModelParams *pParams, TWeightedModelParamsPkt &pkt, float scale, float rowDispScale);


