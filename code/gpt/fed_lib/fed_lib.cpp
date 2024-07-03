#include "stdafx.h"
#include "fed_lib.h"

TGuid FedToken(0xbadf00d, 0x31337, 0x9ece30, 0x31415926);


void PackModelParams(TWeightedModelParamsPkt *p, TModelParams &params, float weight)
{
    p->Pkt.Seek(0);
    TBufferedStream bufIO(p->Pkt, false);
    bufIO.Write(&weight, sizeof(weight));
    PackModelParams(bufIO, params);
}

void UnpackModelParams(TModelParams *pParams, TWeightedModelParamsPkt &pkt)
{
    float weight = 0;
    pkt.Pkt.Seek(0);
    TBufferedStream bufIO(pkt.Pkt, true);
    bufIO.Read(&weight, sizeof(weight));
    UnpackModelParams(pParams, bufIO);
}


void SetWeight(TWeightedModelParamsPkt &pkt, float weight)
{
    pkt.Pkt.Seek(0);
    pkt.Pkt.Write(&weight, sizeof(weight));
}

float GetWeight(TWeightedModelParamsPkt &pkt)
{
    float weight = 0;;
    if (pkt.Pkt.GetLength() >= sizeof(weight)) {
        pkt.Pkt.Seek(0);
        pkt.Pkt.Read(&weight, sizeof(weight));
    }
    return weight;
}


void AddPackedModelParamsScaled(TModelParams *pParams, TWeightedModelParamsPkt &pkt, float scale, float rowDispScale)
{
    float weight = 0;
    pkt.Pkt.Seek(0);
    TBufferedStream bufIO(pkt.Pkt, true);
    bufIO.Read(&weight, sizeof(weight));
    AddPackedModelParamsScaled(pParams, bufIO, scale, rowDispScale);
}
