#include "stdafx.h"
#include "ppm_compact.h"


///////////////////////////////////////////////////////////////////////////////////////////////////
bool TCompactPPMIndex::IsLower(const TVector<TBPEToken> &text, yint pos1, yint pos2, yint *pSameLen)
{
    // faster then sse version somehow
    for (yint k = 0; k < MAX_LEN; ++k) {
        TBPEToken c1 = text[pos1 - k];
        TBPEToken c2 = text[pos2 - k];
        if (c1 != c2) {
            *pSameLen = k;
            return c1 < c2;
        }
    }
    *pSameLen = MAX_LEN;
    return false; // unknown actually
}


