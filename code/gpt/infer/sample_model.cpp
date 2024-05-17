#include "stdafx.h"
#include "sample_model.h"
#include <gpt/data/data.h>


//static const char *PREFIX = "the future will ";
//static con char *PREFIX = "Seven plus eleven equals ";
//static const char *PREFIX = "90276 - 81721 ="; // 8555, arith test
//static const char *PREFIX = "\n176 * 871 ="; // 153296, arith test


static int SampleFromDistr(TXRng &rng, const TVector<float> &distr, float temperature)
{
    // use gumbel max trick
    float best = -1e38f;
    yint res = 0;
    for (yint k = 0; k < YSize(distr); ++k) {
        //float score = distr[k] / -log(rng.GenRandReal3());
        float score = log(distr[k]) / temperature - log(-log(rng.GenRandReal3()));
        if (score > best) {
            best = score;
            res = k;
        }
    }
    return res;
}


TString SampleFromModel(TXRng &rng, TSamplingModel &model, const TString &prefix)
{
    //const float TEMPERATURE = 0.2f;
    //const float TEMPERATURE = 0.8f;
    const float TEMPERATURE = 1;

    yint fragLen = YSize(prefix) + 1;
    yint tokenCount = 0;

    TVector<char> text;
    for (char c : prefix) {
        text.push_back(c);
    }

    TVector<TBPEToken> prompt;
    model.Tokenizer.GenWords(text, 0, YSize(text), &prompt);

    TFragmentGen fgen(model.UsePPM);
    for (TBPEToken &x : prompt) {
        fgen.AddToken(x);
    }
    // generate token or correct utf8 letter
    TString res;
    yint utf8len = 0;
    bool letterHasStarted = false;
    bool isCapitalFirstLetter = false;
    for (;;) {
        TFragment frag;
        fgen.FillFragment(&frag, model.MaxLen);

        TVector<TFragment> fragArr;
        fragArr.push_back(frag);
        MakeTest(model.Window, fragArr, model.Ctx.Get(), MAIN_DEVICE);

        TVector<TVector<float>> predArr;
        model.Ctx->ComputeFragmentPredictions(&predArr);

        for (;;) {
            int letter = SampleFromDistr(rng, predArr.back(), TEMPERATURE);
            DebugPrintf("letter %g, %s\n", letter * 1., model.Tokenizer.GetWord(letter).c_str());
            if (letter == model.Tokenizer.GetCapitalWordToken()) {
                if (letterHasStarted) {
                    continue;
                }
                isCapitalFirstLetter = true;
            } else {
                TString cc = model.Tokenizer.GetWord(letter);
                if (letterHasStarted) {
                    if (YSize(cc) > 1 || (cc[0] & 0xc0) != 0x80) {
                        // failed to sample correct utf8 char continuation
                        continue;
                    }
                } else {
                    if (YSize(cc) == 1) {
                        utf8len = Utf8CodeLength[(ui8)cc[0]];
                        if (utf8len == 255) {
                            // incorrect utf8 char start
                            continue;
                        }
                    } else {
                        utf8len = 1;
                    }
                    letterHasStarted = true;
                }
                res += cc;
            }
            fgen.AddToken(letter);
            break;
        }
        if (letterHasStarted && --utf8len == 0) {
            break;
        }
    }
    if (isCapitalFirstLetter) {
        res = UpcaseFirstLetter(res);
    }
    return res;
}
