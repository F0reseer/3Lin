#pragma once
#include "bpe.h"


///////////////////////////////////////////////////////////////////////////////////////////////////
// sorted array of refs, match length is exact, position not always latest
class TCompactPPMIndex
{
    enum {
        // max subsequence size
        MAX_LEN = 32,
    };

    //static const i32 INVALID_TOKEN = -1;
    typedef ui32 TPos;

    struct TMatch
    {
        yint Pos = 0;
        yint Len = 0;

        TMatch() {}
        TMatch(yint pos, yint len) : Pos(pos), Len(len) {}
    };

private:
    TVector<TPos> Arr;
    yint PrevIndexPos = 0;
    TMatch PrevBestMatch;

private:
    static bool IsLower(const TVector<TBPEToken> &text, yint pos1, yint pos2, yint *pSameLen);

    // search in interval [beg;fin)
    TMatch SearchLongestMatch(const TVector<TBPEToken> &text, yint searchPos, yint beg, yint fin)
    {
        yint begSameLen = -1;
        yint finSameLen = -1;
        while (fin - beg > 1) {
            yint mid = (beg + fin) / 2;
            yint sameLen;
            if (IsLower(text, Arr[mid], searchPos, &sameLen)) {
                beg = mid;
                begSameLen = sameLen;
            } else {
                fin = mid;
                finSameLen = sameLen;
            }
        }
        if (begSameLen < 0) {
            IsLower(text, Arr[beg], searchPos, &begSameLen);
        }
        if (finSameLen == begSameLen) {
            // does not find latest match of this length, but still better then nothing
            return TMatch(Max<yint>(Arr[beg], Arr[fin]), finSameLen);
        }
        if (finSameLen > begSameLen) {
            return TMatch(Arr[fin], finSameLen);
        }
        return TMatch(Arr[beg], begSameLen);
    }

    TMatch SearchLongestMatch(const TVector<TBPEToken> &text, yint searchPos)
    {
        yint sz = YSize(Arr);
        TMatch bestMatch;
        for (yint blk = 1; blk <= sz; blk *= 2) {
            if (sz & blk) {
                yint offset = sz & ~(2 * blk - 1);
                TMatch match = SearchLongestMatch(text, searchPos, offset, offset + blk);
                if (match.Len > bestMatch.Len || (match.Len == bestMatch.Len && match.Pos > bestMatch.Pos)) {
                    bestMatch = match;
                }
            }
        }
        return bestMatch;
    }

    void Merge(const TVector<TBPEToken> &text, yint offset, yint blkSize)
    {
        TVector<TPos> newBlock;
        newBlock.yresize(blkSize * 2);
        yint p1 = offset;
        yint p2 = offset + blkSize;
        yint fin1 = p2;
        yint fin2 = p2 + blkSize;
        yint resPtr = 0;
        for (;;) {
            yint sameLen;
            if (IsLower(text, Arr[p1], Arr[p2], &sameLen)) {
                newBlock[resPtr++] = Arr[p1++];
                if (p1 == fin1) {
                    while (p2 < fin2) {
                        newBlock[resPtr++] = Arr[p2++];
                    }
                    break;
                }
            } else {
                newBlock[resPtr++] = Arr[p2++];
                if (p2 == fin2) {
                    while (p1 < fin1) {
                        newBlock[resPtr++] = Arr[p1++];
                    }
                    break;
                }
            }
        }
        for (yint i = 0; i < blkSize * 2; ++i) {
            Arr[offset + i] = newBlock[i];
        }
    }

    void Add(const TVector<TBPEToken> &text, yint k)
    {
        yint prevSize = YSize(Arr);
        Arr.push_back(k);
        yint newSize = YSize(Arr);
        //printf("%g -> %g\n", prevSize * 1., newSize * 1.);
        for (yint blk = 2; blk <= newSize; blk *= 2) {
            //if ((prevSize & blk) == blk && (newSize & blk) == 0) {
            if ((prevSize ^ newSize) & blk) {
                yint offset = prevSize & ~(blk - 1);
                //printf("  merge [%g %g] [%g %g]\n", offset * 1., offset * 1. + blk / 2 - 1, offset * 1. + blk / 2, offset * 1. + blk - 1);
                Merge(text, offset, blk / 2);
            }
        }
    }

public:
    SAVELOAD(Arr);

    TCompactPPMIndex()
    {
    }

    void Clear()
    {
        Arr.resize(0);
        PrevIndexPos = 0;
        PrevBestMatch = TMatch();
    }

    bool IsValid() const { return !Arr.empty(); }

    void IndexPos(const TVector<TBPEToken> &text, yint indexPos, yint *pBestLen, yint *pBestPos)
    {
        if (indexPos < MAX_LEN) {
            // omit first chars to avoid bound checks
            // can add fake chars to text[] start
            return;
        }
        Y_VERIFY(indexPos < 0xffffffffll);
        TMatch bestMatch;
        // try to extend previous match
        if (PrevBestMatch.Len > 0 && indexPos == PrevIndexPos + 1) {
            if (text[indexPos] == text[PrevBestMatch.Pos + 1]) {
                bestMatch = PrevBestMatch;
                bestMatch.Len += 1;
                bestMatch.Pos += 1;
            }
        }
        // search if extending failed
        if (bestMatch.Len == 0) {
            bestMatch = SearchLongestMatch(text, indexPos);
        }
        *pBestLen = bestMatch.Len;
        if (bestMatch.Len > 0) {
            *pBestPos = bestMatch.Pos;
        }
        PrevIndexPos = indexPos;
        PrevBestMatch = bestMatch;
        Add(text, indexPos);
    }
};
