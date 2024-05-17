#include "stdafx.h"
#include "model_dim.h"


void InitAlibi(TModelDim *p, EAlibi alibi)
{
    if (alibi == ALIBI_NONE) {
        // default values are fine
    } else if (alibi == ALIBI_V1) {
        yint k = 0;
        for (TVector<TModelDim::TAttentionPosParams> &lpArr : p->Layers) {
            for (TModelDim::TAttentionPosParams &lp : lpArr) {
                // HJ2 sync
                yint z = k / 2;
                if (k % 2 == 0) {
                    float k = ((z % 3) + 1) / 3.;
                    lp.AlibiHyper = 20 * k;
                } else if (k % 2 == 1) {
                    float k = ((z % 3)) / 2.;
                    lp.AlibiSlope = 0.5 * k;
                }
                lp.WideLayer = (lp.AlibiHyper == 0 && lp.AlibiSlope == 0);
                ++k;
            }
        }
    } else {
        Y_VERIFY("unknown alibi version");
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// drop table utils
yint CalcDropTableSize(const TModelDim &modelDim)
{
    return modelDim.Dim / 32;
}

void MakeDropTable(TXRng &rng, const TModelDim &modelDim, TVector<ui32> *pDropTable, float channelDrop)
{
    yint sz = CalcDropTableSize(modelDim);
    pDropTable->resize(sz);
    for (yint i = 0; i < sz; ++i) {
        ui32 mask = 0;
        for (int k = 0; k < 32; ++k) {
            if (rng.GenRandReal3() <= channelDrop) {
                mask |= 1 << k;
            }
        }
        (*pDropTable)[i] = mask;
    }
}
