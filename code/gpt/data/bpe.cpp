#include "stdafx.h"
#include "bpe.h"
#include <lib/random/rand_utils.h>


///////////////////////////////////////////////////////////////////////////////////////////////////
ui8 Utf8CodeLength[256] = {
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    4, 4, 4, 4, 4, 4, 4, 4, 255, 255, 255, 255, 255, 255, 255, 255,
};

//void PrintTable()
//{
//    for (yint n1 = 0; n1 < 256; n1 += 16) {
//        for (yint n2 = n1; n2 < n1 + 16; ++n2) {
//            if (n2 & 0x80) {
//                if (n2 & 0x40) {
//                    if (n2 & 0x20) {
//                        if (n2 & 0x10) {
//                            if (n2 & 8) {
//                                DebugPrintf("255, "); // invalid character, can not be part of utf8 encoded text
//                            } else {
//                                DebugPrintf("4, ");
//                            }
//                        } else {
//                            DebugPrintf("3, ");
//                        }
//                    } else {
//                        DebugPrintf("2, ");
//                    }
//                } else {
//                    DebugPrintf("255, "); // not first octet of the character (10xxxxxx octects);
//                }
//            } else {
//                DebugPrintf("1, ");
//            }
//        }
//        DebugPrintf("\n");
//    }
//}

// latin letters
//41..5a
//61..7a
static bool IsUpper1(ui8 c1)
{
    return (c1 >= 0x41) && (c1 <= 0x5a);
}
static void ToLower1(char *p1)
{
    ui8 c1 = *p1;
    if ((c1 >= 0x41) && (c1 <= 0x5a)) {
        *p1 = c1 + 0x20;
    }
}

static void ToUpper1(char *p1)
{
    ui8 c1 = *p1;
    if ((c1 >= 0x61) && (c1 <= 0x7a)) {
        *p1 = c1 - 0x20;
    }
}

// russian letters
//d090 ..d0af
//d081 
//d0b0 ..d0bf, d180 .. d18f
//d191
static bool IsUpper2(ui8 c1, ui8 c2)
{
    return (c1 == 0xd0) && ((c2 == 0x81) || ((c2 >= 0x90) && (c2 <= 0xaf)));
}
static void ToLower2(char *p1, char *p2)
{
    ui8 c1 = *p1;
    if (c1 != 0xd0) {
        return;
    }
    ui8 c2 = *p2;
    if (c2 == 0x81) {
        *p1 = (char)0xd1;
        *p2 = (char)0x91;
        return;
    }
    if (c2 >= 0x90 && c2 <= 0x9f) {
        *p2 = c2 + 0x20;
        return;
    }
    if (c2 >= 0xa0 && c2 <= 0xaf) {
        *p1 = (char)0xd1;
        *p2 = c2 - 0x20;
        return;
    }
}
static void ToUpper2(char *p1, char *p2)
{
    ui8 c1 = *p1;
    if (c1 != 0xd0) {
        return;
    }
    ui8 c2 = *p2;
    if (c1 == 0xd1 && c2 == 0x91) {
        *p1 = (char)0xd0;
        *p2 = (char)0x81;
        return;
    }
    if (c1 == 0xd0 && c2 >= 0xb0 && c2 <= 0xbf) {
        *p2 = c2 - 0x20;
        return;
    }
    if (c1 == 0xd1 && c2 >= 0x80 && c2 <= 0x8f) {
        *p1 = (char)0xd0;
        *p2 = c2 + 0x20;
        return;
    }
}


EWordCase GetWordCase(const TString &str)
{
    EWordCase res = WORD_LOWER_CASE;
    for (yint k = 0, sz = YSize(str); k < sz;) {
        ui8 c = str[k];
        yint len = Utf8CodeLength[c];
        if (k + len > sz) {
            // broken encoding
            res = WORD_MIXED_CASE;
            break;
        }
        if (len == 1 && IsUpper1(str[k])) {
            if (k == 0) {
                res = WORD_CAPITAL_START;
            } else {
                res = WORD_MIXED_CASE;
            }
        }
        if (len == 2 && IsUpper2(str[k], str[k + 1])) {
            if (k == 0) {
                res = WORD_CAPITAL_START;
            } else {
                res = WORD_MIXED_CASE;
            }
        }
        k += len;
    }
    return res;
}


TString ToLower(const TString &str)
{
    TString res = str;
    for (yint k = 0, sz = YSize(str); k < sz; ++k) {
        yint len = Utf8CodeLength[(ui8)res[k]];
        if (k + len > sz) {
            // broken encoding
            break;
        }
        if (len == 1) {
            ToLower1(&res[k]);
        }
        if (len == 2) {
            ToLower2(&res[k], &res[k + 1]);
        }
        k += len;
    }
    return res;
}


TString UpcaseFirstLetter(const TString &str)
{
    if (str.empty()) {
        return "";
    }
    TString res = str;
    yint sz = YSize(res);
    yint len = Utf8CodeLength[(ui8)res[0]];
    if (len > sz) {
        // broken encoding
        return res;
    }
    if (len == 1) {
        ToUpper1(&res[0]);
    }
    if (len == 2) {
        ToUpper2(&res[0], &res[1]);
    }
    return res;
}


///////////////////////////////////////////////////////////////////////////////////////////////////
struct TWordCount
{
    TString Word;
    yint Count = 0;
};
void CollectFrequentWords(const TVector<TVector<char>> &textArr, TVector<TString> *pRes, yint maxWordCount)
{
    TVector<TWordCount> wcArr;
    THashMap<TString, int> wordCounts;
    for (const TVector<char> &text : textArr) {
        TUtf8WordIterator it(text, 0, YSize(text));
        while (it.NextWord()) {
            const TString &word = it.GetWord();
            if (YSize(word) > 1) {
                wordCounts[it.Word] += 1;
            }
        }
    }
    for (auto it = wordCounts.begin(); it != wordCounts.end(); ++it) {
        TWordCount wc;
        wc.Word = it->first;
        wc.Count = it->second;
        wcArr.push_back(wc);
    }
    Sort(wcArr.begin(), wcArr.end(), [](const TWordCount &a, const TWordCount &b) { return a.Count > b.Count; });
    if (YSize(wcArr) > maxWordCount) {
        wcArr.resize(maxWordCount);
    }
    for (const TWordCount &wc : wcArr) {
        pRes->push_back(wc.Word);
    }
}


void CreateWordsetTokenizer(TTokenizer *pTokenizer, const TVector<TString> &words, TTokenizer::ETokenizer tk)
{
    pTokenizer->MakeByteEncoder(tk);
    for (const TString &w : words) {
        Y_ASSERT(YSize(w) > 1);
        pTokenizer->AddWord(w);
    }
}
