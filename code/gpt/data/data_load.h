#pragma once
#include "data.h"


void GenerateArithmetic();
void GenerateArithmetic97();


///////////////////////////////////////////////////////////////////////////////////////////////////
void LoadDocument(TVector<char> *pRes, const TString &fileName);
void LoadDocumentSetFromFiles(TVector<TVector<char>> *pRes, const TString &dir);
void LoadDocumentSetFromBin(TVector<TVector<char>> *pRes, const TString &fileName);
void LoadTokenized(const TString &fileName, yint tokenWidth, yint headerSize, TVector<TBPEToken> *p);
void SaveDocumentSetToBin(const TVector<TVector<char>> &textArr, const TString &fileName);


///////////////////////////////////////////////////////////////////////////////////////////////////
struct TDatasetWeightedSpan
{
    double Weight = 0;
    yint DocsetId = 0;
    yint SpanStart = 0;
    yint SpanFinish = 0;

    TDatasetWeightedSpan() {}
    TDatasetWeightedSpan(double w, yint id, yint start, yint finish) : Weight(w), DocsetId(id), SpanStart(start), SpanFinish(finish) {}
};


struct TDatasetParams
{
    TVector<double> FreqArr;
    yint TotalUtf8Chars = 0;
    yint TotalTokens = 0;
    yint BytesPerToken = 0;
    TVector<TDatasetWeightedSpan> TrainSpans;
    TVector<TDatasetWeightedSpan> TestSpans;
    SAVELOAD(FreqArr, TotalUtf8Chars, TotalTokens, BytesPerToken, TrainSpans, TestSpans);

    TDatasetParams() {}
    TDatasetParams(yint vocabSize)
    {
        ClearPodArray(&FreqArr, vocabSize);
        BytesPerToken = (vocabSize > 65530) ? 3 : 2;
    }
    void CountDocset(const TVector<TBPEToken> &data, yint offset, yint utf8charCount, float testFraction)
    {
        yint vocabSize = YSize(FreqArr);
        for (ui64 x : data) {
            Y_VERIFY(x < vocabSize);
            FreqArr[x] += 1;
        }
        yint len = YSize(data);
        if (testFraction == 0) {
            TrainSpans.push_back(TDatasetWeightedSpan(len, -1, offset, offset + len));
        } else if (testFraction == 1) {
            TestSpans.push_back(TDatasetWeightedSpan(len, -1, offset, offset + len));
        } else {
            // test is always last part of docset
            yint testLen = len * testFraction;
            yint trainLen = len - testLen;
            TrainSpans.push_back(TDatasetWeightedSpan(trainLen, -1, offset, offset + trainLen));
            TestSpans.push_back(TDatasetWeightedSpan(testLen, -1, offset + trainLen, offset + len));
        }
        TotalUtf8Chars += utf8charCount;
        TotalTokens += len;
    }
};


class TDataset : public IDataSource
{
    struct TDocumentSet
    {
        TVector<TBPEToken> Text;
        TVector<TBPEToken> PPM;
        TString IndexFilename;
        TString PPMIndexFilename;
        yint BytesPerToken = 0;
        SAVELOAD(Text, PPM, IndexFilename, PPMIndexFilename, BytesPerToken);
        TIntrusivePtr<TPackedBPETokenReader> Reader;
        TIntrusivePtr<TPackedBPETokenReader> PPMReader;

        void FillFragment(bool usePPM, yint offset, yint fragLen, TFragment *p)
        {
            *p = TFragment();
            p->Offset = offset;
            if (IndexFilename.empty()) {
                for (yint t = 0; t < fragLen; ++t) {
                    p->Text.push_back(Text[offset + t]);
                    if (usePPM) {
                        p->PPM1.push_back(PPM[offset + t]);
                    }
                    p->Target.push_back(Text[offset + t + 1]);
                }
            } else {
                if (Reader.Get() == 0) {
                    Reader = new TPackedBPETokenReader(IndexFilename, BytesPerToken);
                    if (!PPMIndexFilename.empty()) {
                        PPMReader = new TPackedBPETokenReader(PPMIndexFilename, BytesPerToken);
                    }
                }
                TVector<TBPEToken> buf;
                Reader->Read(offset, fragLen + 1, &buf);
                for (yint t = 0; t < fragLen; ++t) {
                    p->Text.push_back(buf[t]);
                    p->Target.push_back(buf[t + 1]);
                }
                if (usePPM) {
                    PPMReader->Read(offset, fragLen, &p->PPM1);
                }
            }
        }
    };

    TVector<TDocumentSet> DocsetArr;
    TDataStats Stats;
    TVector<TDatasetWeightedSpan> TrainSpans;
    TVector<TDatasetWeightedSpan> TestSpans;
public:
    SAVELOAD(DocsetArr, Stats, TrainSpans, TestSpans);

private:
    template <class TRng>
    void MakeRandomFragment(TRng &rng,
        yint docsetId, yint spanStart, yint spanFinish,
        yint fragLen, TFragment *p)
    {
        TDocumentSet &docset = DocsetArr[docsetId];
        if (spanFinish - spanStart <= fragLen - 1) {
            docset.FillFragment(Stats.UsePPM, spanStart, spanFinish - spanStart - 1, p);
        } else {
            yint offset = spanStart + rng.Uniform(spanFinish - spanStart - fragLen - 1);
            docset.FillFragment(Stats.UsePPM, offset, fragLen, p);
        }
    }

public:
    TDataset() {}

    TDataset(bool usePPM, yint vocabSize, yint docStartToken)
    {
        Stats.UsePPM = usePPM;
        Stats.VocabSize = vocabSize;
        Stats.DocStartToken = docStartToken;
        DocsetArr.reserve(10000);
    }

    const TDataStats &GetStats() const override
    {
        return Stats;
    }

    void SampleFragments(ETrainTest trt, yint rngSeed, yint fragCount, yint len, TVector<TFragment> *pFragArr) override
    {
        TXRng rng(rngSeed);
        for (yint k = 0; k < 17; ++k) {
            rng.GenRand();
        }
        const TVector<TDatasetWeightedSpan> &spanArr = (trt == TRAIN) ? TrainSpans : TestSpans;
        Y_VERIFY(!spanArr.empty());
        for (yint k = 0; k < fragCount; ++k) {
            // use gumbel max trick
            float best = -1e38f;
            const TDatasetWeightedSpan *bestSpan = &spanArr[0];
            for (yint k = 0; k < YSize(spanArr); ++k) {
                float score = spanArr[k].Weight / -log(rng.GenRandReal3());
                if (score > best) {
                    best = score;
                    bestSpan = &spanArr[k];
                }
            }
            TFragment &frag = *pFragArr->insert(pFragArr->end());
            MakeRandomFragment(rng, bestSpan->DocsetId, bestSpan->SpanStart, bestSpan->SpanFinish, len, &frag);
        }
    }

    friend class TDatasetBuilder;
};


class TDatasetBuilder : public TThrRefBase
{
    TIntrusivePtr<TDataset> Dataset;
    TVector<double> FreqArr;
    yint TotalUtf8Chars = 0;
    yint TotalTokens = 0;
    yint DocStartToken = -1;

private:
    void Init(bool usePPM, yint vocabSize, yint docStartToken)
    {
        DocStartToken = docStartToken;
        Dataset->Stats.UsePPM = usePPM;
        Dataset->Stats.VocabSize = vocabSize;
        ClearPodArray(&FreqArr, vocabSize);
        Dataset->DocsetArr.reserve(10000);
    }

    void AddParams(yint docsetId, const TDatasetParams &params, float weight)
    {
        Y_VERIFY(YSize(FreqArr) == YSize(params.FreqArr));
        for (yint x = 0; x < YSize(params.FreqArr); ++x) {
            FreqArr[x] += params.FreqArr[x];
        }
        for (const TDatasetWeightedSpan &span : params.TrainSpans) {
            Dataset->TrainSpans.push_back(TDatasetWeightedSpan(span.Weight * weight, docsetId, span.SpanStart, span.SpanFinish));
        }
        for (const TDatasetWeightedSpan &span : params.TestSpans) {
            Dataset->TestSpans.push_back(TDatasetWeightedSpan(span.Weight * weight, docsetId, span.SpanStart, span.SpanFinish));
        }
        TotalUtf8Chars += params.TotalUtf8Chars;
        TotalTokens += params.TotalTokens;
    }

public:
    TDatasetBuilder(bool usePPM, yint vocabSize, yint docStartToken)
    {
        Dataset = new TDataset(usePPM, vocabSize, docStartToken);
        ClearPodArray(&FreqArr, vocabSize);
    }

    TDatasetBuilder(bool usePPM, const TTokenizer &tokenizer)
    {
        yint vocabSize = tokenizer.GetVocabSize();
        yint docStartToken = tokenizer.HasDocStartToken() ? tokenizer.GetDocStartToken() : -1;
        Dataset = new TDataset(usePPM, vocabSize, docStartToken);
        ClearPodArray(&FreqArr, vocabSize);
    }

    void AddTokenizedDocset(const TVector<TBPEToken> &data, const TDatasetParams &params, float weight)
    {
        yint docsetId = YSize(Dataset->DocsetArr);
        TDataset::TDocumentSet &docset = *Dataset->DocsetArr.insert(Dataset->DocsetArr.end());
        docset.Text = data;
        if (Dataset->Stats.UsePPM) {
            ComputeWindowPPM(docset.Text, &docset.PPM, DocStartToken);
        }
        AddParams(docsetId, params, weight);
    }

    void AddIndexedDocset(const TString &indexFilename, const TString &ppmIndexFilename, const TDatasetParams &params, yint vocabSize, float weight)
    {
        Y_VERIFY(vocabSize == Dataset->Stats.VocabSize);
        yint docsetId = YSize(Dataset->DocsetArr);
        TDataset::TDocumentSet &docset = *Dataset->DocsetArr.insert(Dataset->DocsetArr.end());
        docset.IndexFilename = indexFilename;
        if (Dataset->Stats.UsePPM) {
            docset.PPMIndexFilename = ppmIndexFilename;
        }
        docset.BytesPerToken = params.BytesPerToken;
        AddParams(docsetId, params, weight);
    }

    TIntrusivePtr<TDataset> MakeDataset()
    {
        yint vocabSize = Dataset->Stats.VocabSize;
        Dataset->Stats.Bias.resize(vocabSize);
        for (yint c = 0; c < vocabSize; ++c) {
            Dataset->Stats.Bias[c] = log2(FreqArr[c] + 0.5);
        }
        Dataset->Stats.Compression = TotalTokens / (TotalUtf8Chars + 0.);
        Dataset->Stats.HasTest = !Dataset->TestSpans.empty();
        return Dataset.Release();
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
TIntrusivePtr<TDataset> MakeCharDataset(TTokenizer *pTokenizer, const TVector<char> &text, float testFraction, bool usePPM);
void AddDocset(TIntrusivePtr<TDatasetBuilder> pBuilder, const TTokenizer &tokenizer, const TVector<TVector<char>> &docSet, float weight, float testFraction);

void AddIndexedDocset(TDatasetBuilder *pBuilder, const TString &dir, float weight);
void IndexDocsetDir(const TString &dir, const TTokenizer &tokenizer, bool usePPM, float testFraction);

void IndexTokenizedDir(const TString &dir, yint vocabSize, yint docStartToken, bool usePPM, float testFraction, yint tokenWidth, yint headerSize);
