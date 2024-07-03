#include "stdafx.h"
#include "model_dim.h"
#include <lib/config/config.h>


///////////////////////////////////////////////////////////////////////////////////////////////////
struct TModelDimString
{
    yint Dim = 256;
    yint QDim = 128;
    yint TTDim = 64;
    yint Depth = 64;
    yint WideLimitWindow = 64;

    TModelDimString() {}
    TModelDimString(const TString &modelDims);
};

TModelDimString::TModelDimString(const TString &modelDims)
{
    TStringParams sp(modelDims);
    for (TStringParams::TParam &param : sp.Params) {
        if (param.Name == "e") {
            Dim = param.Value;
        } else if (param.Name == "q") {
            QDim = param.Value;
        } else if (param.Name == "tt") {
            TTDim = param.Value;
        } else if (param.Name == "d") {
            Depth = param.Value;
        } else if (param.Name == "w") {
            WideLimitWindow = param.Value;
        }
    }
}


TString GetModelDimsString(const TModelDim &modelDim)
{
    TModelDimString defMD;
    TString res = Sprintf("e%g", modelDim.Dim * 1.);
    if (modelDim.QDim != defMD.QDim) {
        res += Sprintf("q%g", modelDim.QDim * 1.);
    }
    if (modelDim.TTDim != defMD.TTDim) {
        res += Sprintf("tt%g", modelDim.TTDim * 1.);
    }
    yint depth = YSize(modelDim.Layers);
    res += Sprintf("d%g", depth * 1.);
    yint wideLimit = modelDim.GetWideLimitWindow();
    if (wideLimit != defMD.WideLimitWindow) {
        res += Sprintf("w%g", wideLimit * 1.);
    }
    return res;
}


///////////////////////////////////////////////////////////////////////////////////////////////////
static void InitAlibi(TModelDim *p, EAlibi alibi, yint depth, bool combineLayers, yint wideLimitWindow)
{
    p->Layers.resize(depth);
    for (TVector<TModelDim::TAttentionPosParams> &lpArr : p->Layers) {
        lpArr.resize(1);
    }
    // useful in attention profiling
    if (depth == 1) {
        p->AttentionWidthArr.push_back(wideLimitWindow);
        p->Layers[0][0].AttentionWidthId = 0;
        return;
    }
    // configure position encoding
    if (alibi == ALIBI_NONE) {
        ClearPodArray(&p->AttentionWidthArr, 1);

    } else if (alibi == ALIBI_V1) {
        Y_VERIFY(!combineLayers);
        yint k = 0;
        p->AttentionWidthArr.resize(2);
        p->AttentionWidthArr[0] = 64;
        p->AttentionWidthArr[1] = wideLimitWindow;
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
                lp.AttentionWidthId = (lp.AlibiHyper == 0 && lp.AlibiSlope == 0) ? 1 : 0;
                ++k;
            }
        }

    } else if (alibi == ALIBI_V2) {
        yint widthPattern[] = {
            // start4, 1.1798, 1.0458, 9363
            0, 1, 2, 0, 3, 4,
        };
        yint start = 4;

        p->AttentionWidthArr.clear();
        p->AttentionWidthArr.push_back(1);
        p->AttentionWidthArr.push_back(4);
        p->AttentionWidthArr.push_back(16);
        p->AttentionWidthArr.push_back(64);
        p->AttentionWidthArr.push_back(wideLimitWindow);
        yint k = 0;
        for (TVector<TModelDim::TAttentionPosParams> &lpArr : p->Layers) {
            for (TModelDim::TAttentionPosParams &lp : lpArr) {
                yint ww = ARRAY_SIZE(widthPattern);
                if (k < start) {
                    lp.AttentionWidthId = 0;
                } else {
                    lp.AttentionWidthId = widthPattern[(k - start) % ww];
                }
                ++k;
            }
        }

    } else if (alibi == ALIBI_V2_YOCO) {
        const yint WIDE_ID = 4;
        yint widthPattern[] = {
            0, 1, 2, 0, 3, WIDE_ID,
        };
        yint start = 4;
        float WIDE_POSITION = 0.5f;
        p->AttentionWidthArr.clear();
        p->AttentionWidthArr.push_back(1);
        p->AttentionWidthArr.push_back(4);
        p->AttentionWidthArr.push_back(16);
        p->AttentionWidthArr.push_back(64);
        p->AttentionWidthArr.push_back(wideLimitWindow);
        yint total = 0;
        for (TVector<TModelDim::TAttentionPosParams> &lpArr : p->Layers) {
            total += YSize(lpArr);
        }
        yint k = 0;
        yint ptr = 0;
        bool hasCreatedWide = false;
        for (TVector<TModelDim::TAttentionPosParams> &lpArr : p->Layers) {
            for (TModelDim::TAttentionPosParams &lp : lpArr) {
                yint ww = ARRAY_SIZE(widthPattern);
                if (k < start) {
                    lp.AttentionWidthId = 0;
                } else {
                    yint id = widthPattern[ptr++ % ww];
                    if (k < total * WIDE_POSITION && id == WIDE_ID) {
                        id = widthPattern[ptr++ % ww];
                        Y_ASSERT(id != WIDE_ID);
                    }
                    if (id == WIDE_ID) {
                        if (!hasCreatedWide) {
                            id |= ATT_ID_CREATE_WIDE_FLAG;
                            hasCreatedWide = true;
                        }
                        id |= ATT_ID_USE_WIDE_FLAG;
                    }
                    lp.AttentionWidthId = id;
                }
                ++k;
            }
        }

    } else if (alibi == ALIBI_V3) {
        p->AttentionWidthArr.clear();
        p->AttentionWidthArr.push_back(1);
        p->AttentionWidthArr.push_back(4);
        p->AttentionWidthArr.push_back(16);
        p->AttentionWidthArr.push_back(64);
        p->AttentionWidthArr.push_back(wideLimitWindow);
        p->Layers.resize(0);
        yint groupSize = depth / 15;
        for (yint width = 1; width <= 5; ++width) {
            yint count = groupSize;
            if (width == 5) {
                count += (depth - groupSize * 15) / 5;
            }
            for (yint k = 0; k < count; ++k) {
                if (combineLayers) {
                    TVector<TModelDim::TAttentionPosParams> lpArr;
                    for (yint w = 0; w < width; ++w) {
                        lpArr.push_back(TModelDim::TAttentionPosParams(0., 0., w));
                    }
                    p->Layers.push_back(lpArr);

                } else {
                    for (yint w = 0; w < width; ++w) {
                        TVector<TModelDim::TAttentionPosParams> lpArr;
                        lpArr.push_back(TModelDim::TAttentionPosParams(0., 0., w));
                        p->Layers.push_back(lpArr);
                    }
                }
            }
        }

    } else {
        Y_VERIFY("unknown alibi version");
    }
}


void InitModelDim(TModelDim *pRes, const TString &modelDimStr, EAlibi alibi, yint vocabSize, yint labelCount, ui64 flags)
{
    TModelDimString dims(modelDimStr);

    TModelDim &modelDim = *pRes;
    modelDim = TModelDim();
    modelDim.Dim = dims.Dim;
    modelDim.QDim = dims.QDim;
    modelDim.TTDim = dims.TTDim;
    modelDim.LabelCount = labelCount;
    modelDim.VocabSize = vocabSize;
    modelDim.Flags = flags;
    InitAlibi(&modelDim, alibi, dims.Depth, modelDim.HasFlag(MPF_COMBINE_LAYERS), dims.WideLimitWindow);
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
