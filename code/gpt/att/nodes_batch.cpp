#include "stdafx.h"
#include "nodes_batch.h"

///////////////////////////////////////////////////////////////////////////////////////////////////
void TNodesBatch::Init()
{
    SampleIndex.resize(0);
    LabelArr.resize(0);
    LabelPtr.resize(0);
    LabelPtr.push_back(0);
    Target.resize(0);
    Att.Init();
    WideAtt.Init();
}


void TNodesBatch::AddSample(int idx, const TVector<TLabelIndex> &labels, const TVector<TAttentionSpan> &attSpans, const TVector<TAttentionSpan> &wideAttSpans)
{
    SampleIndex.push_back(idx);
    LabelArr.insert(LabelArr.end(), labels.begin(), labels.end());
    LabelPtr.push_back(YSize(LabelArr));
    Att.AddSpans(attSpans);
    Att.AddSample();
    WideAtt.AddSpans(wideAttSpans);
    WideAtt.AddSample();
}


