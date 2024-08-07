#pragma once
#include "bpe.h"
#include "ppm_window.h"
#include <gpt/rng/xrng.h>


///////////////////////////////////////////////////////////////////////////////////////////////////
struct TFragment
{
    yint Offset = 0;
    TVector<TBPEToken> Text;
    TVector<TBPEToken> PPM1;
    TVector<TBPEToken> Target;
    SAVELOAD(Offset, Text, PPM1, Target);

    yint GetLength() const { return YSize(Text); }
    void Truncate(yint len)
    {
        if (YSize(Text) > len) {
            Text.resize(len);
        }
        if (YSize(PPM1) > len) {
            Text.resize(len);
        }
        if (YSize(Target) > len) {
            Text.resize(len);
        }
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
class TFragmentGen
{
    bool UsePPM = false;
    TVector<TBPEToken> Text;
    TVector<TBPEToken> PPM;
    TWindowPPMIndex Index;
public:
    TFragmentGen(bool usePPM) : UsePPM(usePPM) {}

    void AddToken(TBPEToken token)
    {
        Text.push_back(token);
        if (UsePPM) {
            yint bestLen = 0;
            yint bestPos = 0;
            Index.IndexPos(Text, YSize(Text) - 1, &bestLen, &bestPos);
            if (bestLen > 0) {
                PPM.push_back(Text[bestPos + 1]);
            } else {
                PPM.push_back(UNDEFINED_TOKEN);
            }
        }
    }

    void FillFragment(TFragment *pFrag, yint maxLen) const
    {
        *pFrag = TFragment();
        yint start = Max<yint>(0, YSize(Text) - maxLen);
        yint fin = YSize(Text);
        for (yint t = start; t < fin; ++t) {
            pFrag->Text.push_back(Text[t]);
            if (UsePPM) {
                pFrag->PPM1.push_back(PPM[t]);
            }
        }
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
struct IDataSource : public TThrRefBase
{
    struct TDataStats
    {
        bool UsePPM = false;
        float Compression = 0;
        yint VocabSize = 0;
        yint DocStartToken = -1;
        TVector<float> Bias;
        bool HasTest = false;

        SAVELOAD(UsePPM, Compression, VocabSize, DocStartToken, Bias, HasTest);
    };

    enum ETrainTest
    {
        TRAIN,
        TEST,
    };

    virtual const TDataStats &GetStats() const = 0;
    virtual void SampleFragments(ETrainTest trt, yint rngSeed, yint fragCount, yint len, TVector<TFragment> *pFragArr) = 0;
};

