#include "stdafx.h"
#include "sliding_window.h"
#include <gpt/data/data.h>


const yint HASH_VOCAB_SIZE_LN = 11;
const yint HASH_VOCAB_SIZE = 1ull << HASH_VOCAB_SIZE_LN;
const yint HASH_VOCAB_COUNT = 3;

static void AddToken(bool hashedVocab, TVector<TLabelIndex> *p, yint token)
{
    if (hashedVocab) {
        ui64 hh = 0x9ae16a3b2f90404fULL;
        for (yint k = 0; k < HASH_VOCAB_COUNT; ++k) {
            hh = (hh + token) * 0xc949d7c7509e6557ULL;
            yint token_hash = hh >> (64 - HASH_VOCAB_SIZE_LN);
            p->push_back(token_hash);
        }
    } else {
        p->push_back(token);
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// attention graph
static bool IsHashedVocab(const TModelDim &modelDim)
{
    if (modelDim.HasFlag(MPF_HASHED_EMBED)) {
        return true;
    } else {
        return false;
    }
}

static yint GetLabelCount(const TModelDim &modelDim)
{
    yint res = 0;
    if (modelDim.HasFlag(MPF_MLM_BERT)) {
        yint wideLimitWindow = modelDim.GetWideLimitWindow();
        res = 1 + wideLimitWindow + (modelDim.VocabSize + 1);
    } else if (IsHashedVocab(modelDim)) {
        res = HASH_VOCAB_SIZE;
    } else {
        if (modelDim.HasFlag(MPF_PPM)) {
            res = 1 + 2 * (modelDim.VocabSize + 1);
        } else {
            res = 1 + 1 * (modelDim.VocabSize + 1);
        }
    }
    return res;
}

void InitModelDim(TModelDim *pRes, const TString &modelDimStr, EAlibi alibi, yint vocabSize, ui64 flags)
{
    InitModelDim(pRes, modelDimStr, alibi, vocabSize, 0, flags);
    pRes->LabelCount = GetLabelCount(*pRes);
}

yint GetNodeCount(yint len)
{
    return len + 1;
}


static void AddAttSpans(yint docStart, yint nodeId, yint limitWindow, TVector<TVector<TAttentionSpan>> *pAtt)
{
    yint attStart = Max<yint>(docStart, nodeId - limitWindow);
    yint attFinish = nodeId - 1;
    if (attFinish >= attStart) {
        (*pAtt)[nodeId].push_back(TAttentionSpan(attStart, attFinish));
    }
    if (attStart > 0) {
        (*pAtt)[nodeId].push_back(TAttentionSpan(0, 0)); // add attention to start token
    }
}


// process single fragment
static void GenerateAttentionGraph(
    const TModelDim &modelDim, TXRng &rng, float tokenDrop,
    const TFragment &frag, yint lossType,
    TVector<TVector<TLabelIndex>> *pLabels,
    TVector<TVector<TVector<TAttentionSpan>>> *pAttArr,
    TVector<TNodeTarget> *pTargetArr, TVector<yint> *pNodeToSampleIndex)
{
    bool isHashedVocab = IsHashedVocab(modelDim);
    yint len = YSize(frag.Text);
    pLabels->resize(len + 1);
    yint attentionWidthCount = modelDim.GetAttentionWidthCount();
    pAttArr->resize(attentionWidthCount);
    for (yint wa = 0; wa < attentionWidthCount; ++wa) {
        (*pAttArr)[wa].resize(len + 1);
    }
    pNodeToSampleIndex->resize(len + 1);
    // start token
    (*pLabels)[0].push_back(0);
    (*pNodeToSampleIndex)[0] = -1;

    if (modelDim.HasFlag(MPF_MLM_BERT)) {
        yint wideLimitWindow = modelDim.GetWideLimitWindow();
        Y_VERIFY(len <= wideLimitWindow && "absolute position encoding is impossible, sequence too long");
        while (len > 0 && frag.Text[len - 1] == 0) {
            (*pNodeToSampleIndex)[len] = -1;
            --len;
        }
        for (yint t = 0; t < len; ++t) {
            yint nodeId = t + 1;

            yint lblBase = 1;
            // position
            AddToken(isHashedVocab, &(*pLabels)[nodeId], lblBase + t);

            // add labels
            lblBase += wideLimitWindow;
            if (frag.Text[t] != UNDEFINED_TOKEN) {
                // make gaps to fill by training
                AddToken(isHashedVocab, &(*pLabels)[nodeId], lblBase + 1 + frag.Text[t]);
            } else {
                AddToken(isHashedVocab, &(*pLabels)[nodeId], lblBase + 0);
            }

            if (frag.Target[t] != UNDEFINED_TOKEN) {
                pTargetArr->push_back(TNodeTarget(nodeId, frag.Target[t]));
            }

            // add attention spans, same for all widths
            for (yint wa = 0; wa < attentionWidthCount; ++wa) {
                (*pAttArr)[wa][nodeId].push_back(TAttentionSpan(0, nodeId - 1));
                if (t < len - 1) {
                    (*pAttArr)[wa][nodeId].push_back(TAttentionSpan(nodeId + 1, len));
                }
            }

            (*pNodeToSampleIndex)[nodeId] = frag.Offset + t;
        }

    } else {
        // samples
        yint docStart = 0;
        for (yint t = 0; t < len; ++t) {
            yint nodeId = t + 1;

            // detect document start and limit attention to the document
            if (modelDim.HasFlag(MPF_USE_DOC_START_TOKEN)) {
                if (t > 0 && frag.Text[t] == modelDim.DocStartToken) {
                    docStart = nodeId;
                }
            }
            if (modelDim.HasFlag(MPF_GROK_BINARY_OP)) {
                if (t > 0 && frag.Text[t] == 0) {
                    docStart = nodeId;
                }
            }

            // add labels
            yint lblBase = 1;
            if (rng.GenRandReal3() <= tokenDrop) {
                // make gaps to fill by training
                AddToken(isHashedVocab, &(*pLabels)[nodeId], lblBase + 1 + frag.Text[t]);
            } else {
                AddToken(isHashedVocab, &(*pLabels)[nodeId], lblBase + 0);
            }

            // ppm features
            if (modelDim.HasFlag(MPF_PPM)) {
                lblBase += 1 + modelDim.VocabSize;
                if (frag.PPM1[t] != UNDEFINED_TOKEN) {
                    if (rng.GenRandReal3() <= tokenDrop) {
                        AddToken(isHashedVocab, &(*pLabels)[nodeId], lblBase + 1 + frag.PPM1[t]);
                    } else {
                        AddToken(isHashedVocab, &(*pLabels)[nodeId], lblBase + 0); // skip token
                    }
                }
                //lblBase += 1 + modelDim.VocabSize;
                //if (frag.PPM2[t] != UNDEFINED_TOKEN) {
                //    if (rng.GenRandReal3() <= tokenDrop) {
                //        AddToken(isHashedVocab, &(*pLabels)[nodeId], lblBase + 1 + frag.PPM2[t]);
                //    } else {
                //        AddToken(isHashedVocab, &(*pLabels)[nodeId], lblBase + 0); // skip token
                //    }
                //}
            }

            // add attention span
            for (yint wa = 0; wa < attentionWidthCount; ++wa) {
                yint limitWindow = modelDim.AttentionWidthArr[wa];
                AddAttSpans(docStart, nodeId, limitWindow, &(*pAttArr)[wa]);
            }

            // add loss
            if (modelDim.HasFlag(MPF_GROK_BINARY_OP)) {
                // special loss, target only binary op result, 0 is special token for this dataset meaning start of sample
                if (docStart > 0 && t + 1 < YSize(frag.Target) && frag.Target[t + 1] == 0) {
                    pTargetArr->push_back(TNodeTarget(nodeId, frag.Target[t]));
                }
            } else if (!frag.Target.empty()) {
                bool isLoss = true;
                if (modelDim.HasFlag(MPF_TAIL_LOSS)) {
                    isLoss = (t >= 0.5 * len); // account second half in reported loss
                }
                if (lossType == ATT_GRAPH_TRAIN_LOSS || (lossType == ATT_GRAPH_TEST_LOSS && isLoss)) {
                    pTargetArr->push_back(TNodeTarget(nodeId, frag.Target[t]));
                }
            }

            (*pNodeToSampleIndex)[nodeId] = frag.Offset + t;
        }
    }
}



///////////////////////////////////////////////////////////////////////////////////////////////////
// make train/test contexts

void InitLabelData(const TModelDim &modelDim, TXRng &rng, float tokenDrop,
    const TVector<TFragment> &fragArr, yint lossType,
    TNodesBatch *pNodes)
{
    pNodes->Init(modelDim.GetAttentionWidthCount());

    for (const TFragment &frag : fragArr) {
        yint ptr = pNodes->GetNodeCount();

        TVector<TVector<TLabelIndex>> fragLabels;
        TVector<TVector<TVector<TAttentionSpan>>> fragAttSpansArr;
        TVector<TNodeTarget> fragTargets;
        TVector<yint> fragNodeToSampleIndex;
        GenerateAttentionGraph(modelDim, rng, tokenDrop,
            frag, lossType,
            &fragLabels, &fragAttSpansArr, &fragTargets, &fragNodeToSampleIndex);

        yint nodeCount = YSize(fragLabels);
        for (yint t = 0; t < nodeCount; ++t) {
            TVector<TVector<TAttentionSpan>> rrArr;
            rrArr.resize(YSize(fragAttSpansArr));
            for (yint wa = 0; wa < YSize(fragAttSpansArr); ++wa) {
                Y_ASSERT(nodeCount == YSize(fragAttSpansArr[wa]));
                TVector<TAttentionSpan> rr = fragAttSpansArr[wa][t];
                for (TAttentionSpan &span : rr) {
                    span.Shift(ptr);
                }
                rrArr[wa] = rr;
            }
            pNodes->AddSample(fragNodeToSampleIndex[t], fragLabels[t], rrArr);
        }

        for (TNodeTarget nt : fragTargets) {
            nt.Node += ptr;
            pNodes->Target.push_back(nt);
        }
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// results are discarded, so we don't care about race conditions
TXRng NopRng;
