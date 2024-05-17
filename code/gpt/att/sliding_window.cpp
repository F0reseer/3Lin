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
static bool IsHashedVocab(ui64 mpfFlags, yint vocabSize)
{
    if (mpfFlags & MPF_HASHED_EMBED) {
        return true;
    } else {
        return false;
    }
}

yint GetLabelCount(ui64 mpfFlags, yint vocabSize)
{
    yint res = 0;
    if (IsHashedVocab(mpfFlags, vocabSize)) {
        res = HASH_VOCAB_SIZE;
    } else {
        if (mpfFlags & MPF_PPM) {
            res = 1 + 2 * (vocabSize + 1);
        } else {
            res = 1 + 1 * (vocabSize + 1);
        }
    }
    return res;
}

yint GetNodeCount(yint len)
{
    return len + 1;
}


static void AddAttSpans(yint docStart, yint nodeId, yint limitWindow, TVector<TVector<TAttentionSpan>> *pAtt)
{
    yint attStart = Max<yint>(docStart, nodeId - limitWindow + 1);
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
    const TModelDim &modelDim, TXRng &rng, float tokenDrop, const TWindowSizeLimit &window,
    const TFragment &frag, yint lossType,
    TVector<TVector<TLabelIndex>> *pLabels, TVector<TVector<TAttentionSpan>> *pAtt, TVector<TVector<TAttentionSpan>> *pWideAtt,
    TVector<TNodeTarget> *pTargetArr, TVector<yint> *pNodeToSampleIndex)
{
    bool isHashedVocab = IsHashedVocab(modelDim.Flags, modelDim.VocabSize);
    yint len = YSize(frag.Text);
    pLabels->resize(len + 1);
    pAtt->resize(len + 1);
    pWideAtt->resize(len + 1);
    pNodeToSampleIndex->resize(len + 1);
    // start token
    (*pLabels)[0].push_back(0);
    (*pNodeToSampleIndex)[0] = -1;
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
        AddAttSpans(docStart, nodeId, window.Limit, pAtt);
        AddAttSpans(docStart, nodeId, window.LimitWide, pWideAtt);

        // add loss
        if (modelDim.HasFlag(MPF_GROK_BINARY_OP)) {
            // special loss, target only binary op result, 0 is special token for this dataset meaning start of sample
            if (docStart > 0 && t +  1 < YSize(frag.Target) && frag.Target[t + 1] == 0) {
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



///////////////////////////////////////////////////////////////////////////////////////////////////
// make train/test contexts

void InitLabelData(const TModelDim &modelDim, TXRng &rng, float tokenDrop, const TWindowSizeLimit &window,
    const TVector<TFragment> &fragArr, yint lossType,
    TNodesBatch *pNodes)
{
    pNodes->Init();

    for (const TFragment &frag : fragArr) {
        yint ptr = pNodes->GetNodeCount();

        TVector<TVector<TLabelIndex>> fragLabels;
        TVector<TVector<TAttentionSpan>> fragAttSpans;
        TVector<TVector<TAttentionSpan>> fragWideAttSpans;
        TVector<TNodeTarget> fragTargets;
        TVector<yint> fragNodeToSampleIndex;
        GenerateAttentionGraph(modelDim, rng, tokenDrop, window,
            frag, lossType,
            &fragLabels, &fragAttSpans, &fragWideAttSpans, &fragTargets, &fragNodeToSampleIndex);

        yint nodeCount = YSize(fragLabels);
        Y_ASSERT(nodeCount == YSize(fragAttSpans));

        for (yint t = 0; t < nodeCount; ++t) {
            TVector<TAttentionSpan> rr = fragAttSpans[t];
            for (TAttentionSpan &span : rr) {
                span.Shift(ptr);
            }
            TVector<TAttentionSpan> rwide = fragWideAttSpans[t];
            for (TAttentionSpan &span : rwide) {
                span.Shift(ptr);
            }
            pNodes->AddSample(fragNodeToSampleIndex[t], fragLabels[t], rr, rwide);
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
