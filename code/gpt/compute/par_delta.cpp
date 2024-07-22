#include "stdafx.h"
#include "par_delta.h"
#include <immintrin.h>


///////////////////////////////////////////////////////////////////////////////////////////////////
inline float HalfToFloat(ui16 x)
{
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_set1_epi16(x)));
}

inline ui16 FloatToHalf(float x)
{
    return _mm_extract_epi16(_mm_cvtps_ph(_mm_set1_ps(x), 0), 0);
}


void TModelMatrixHalfDelta::GetAllData(TArray2D<float> *p) const
{
    yint xSize = SizeX;
    yint ySize = SizeY;
    p->SetSizes(xSize, ySize);
    p->FillZero();
    for (yint y = 0; y < ySize; ++y) {
        const TRow &row = Rows[y];
        if (row.Scale == 0) {
            continue;
        }
        const ui16 *deltaRowPtr = GetRow(y);
        for (yint x = 0; x < xSize; ++x) {
            (*p)[y][x] = HalfToFloat(deltaRowPtr[x]) * row.Scale;
        }
    }
}


inline void CopyRow(TModelMatrixHalfDelta::TRow &row, yint xSize, ui16 *dstArg, const i8 *srcArg, float srcScale)
{
    float sum2 = 0;
    if (srcScale > 0) {
        __m256 mult = _mm256_set1_ps(1.0f / 64);
        const ui64 *src = (const ui64*)srcArg;
        __m128i *dst = (__m128i *) dstArg;
        __m256 rowSum2 = _mm256_setzero_ps();
        for (yint x8 = 0; x8 < xSize / 8; ++x8) {
            // Load the 64-bit integer into a 128-bit register
            __m128i src8 = _mm_cvtsi64_si128(src[x8]);
            // Unpack the 8 int8 integers into 32-bit integers
            __m256i src32 = _mm256_cvtepi8_epi32(src8);
            // convert to float and scale
            __m256 newDstVal = _mm256_mul_ps(_mm256_cvtepi32_ps(src32), mult);
            // convert to fp16 and write
            dst[x8] = _mm256_cvtps_ph(newDstVal, 0);
            // collect rowsum
            rowSum2 = _mm256_add_ps(_mm256_mul_ps(newDstVal, newDstVal), rowSum2);
        }
        sum2 = HorizontalSum(rowSum2);
    }
    Y_ASSERT(!isnan(sum2) && finite(sum2));
    row.Scale = srcScale / 64;
    row.Sum2 = sum2 * Sqr(row.Scale);
}


inline void AddRow(TModelMatrixHalfDelta::TRow &row, yint xSize, ui16 *dstArg, const i8 *srcArg, float srcScale)
{
    Y_ASSERT(row.Scale > 0 && srcScale > 0);
    float srcRowScale = srcScale / 64;
    float newRowScale = Max<float>(srcRowScale, row.Scale);
    __m256 srcMult = _mm256_set1_ps((1.0f / 64) * srcRowScale / newRowScale);
    __m256 dstMult = _mm256_set1_ps(1 * row.Scale / newRowScale);

    const ui64 *src = (const ui64 *)srcArg;
    __m128i *dst = (__m128i *) dstArg;
    __m256 rowSum2 = _mm256_setzero_ps();
    for (yint x8 = 0; x8 < xSize / 8; ++x8) {
        __m256 dstVal = _mm256_cvtph_ps(dst[x8]);
        // Load the 64-bit integer into a 128-bit register
        __m128i src8 = _mm_cvtsi64_si128(src[x8]);
        // Unpack the 8 int8 integers into 32-bit integers
        __m256i src32 = _mm256_cvtepi8_epi32(src8);
        // convert to float and scale
        __m256 srcVal = _mm256_cvtepi32_ps(src32);
        // new val
        __m256 newDstVal = _mm256_add_ps(_mm256_mul_ps(srcVal, srcMult), _mm256_mul_ps(dstVal, dstMult));
        // convert to fp16 and write
        dst[x8] = _mm256_cvtps_ph(newDstVal, 0);
        // collect rowsum
        rowSum2 = _mm256_add_ps(_mm256_mul_ps(newDstVal, newDstVal), rowSum2);
    }
    float sum2 = HorizontalSum(rowSum2);
    Y_ASSERT(!isnan(sum2) && finite(sum2));
    row.Scale = newRowScale;
    row.Sum2 = sum2 * Sqr(row.Scale);
}


void Copy(TModelMatrixHalfDelta *p, const TModelMatrixInt8Delta &delta)
{
    yint xSize = delta.XSize;
    yint ySize = delta.YSize;
    p->Delta.resize(xSize * ySize);
    p->Rows.resize(ySize);
    for (yint y = 0; y < ySize; ++y) {
        TModelMatrixHalfDelta::TRow &row = p->Rows[y];
        const i8 *src = delta.GetRow(y);
        ui16 *dst = &p->Delta[y * xSize];
        float rowScale = delta.RowScale[y];
        CopyRow(p->Rows[y], xSize, dst, src, rowScale);
    }
}


void Add(TModelMatrixHalfDelta *p, const TModelMatrixInt8Delta &delta)
{
    yint xSize = delta.XSize;
    yint ySize = delta.YSize;
    for (yint y = 0; y < ySize; ++y) {
        TModelMatrixHalfDelta::TRow &row = p->Rows[y];
        const i8 *src = delta.GetRow(y);
        ui16 *dst = &p->Delta[y * xSize];
        float rowScale = delta.RowScale[y];
        if (row.Sum2 == 0) {
            CopyRow(p->Rows[y], xSize, dst, src, rowScale);
        } else {
            if (rowScale > 0) {
                AddRow(row, xSize, dst, src, rowScale);
            }
        }
    }
}


void Compress(TModelMatrixInt8Delta *p, const TArray2D<float> &data)
{
    yint xSize = data.GetXSize();
    yint ySize = data.GetYSize();
    Y_ASSERT(xSize == p->XSize);
    Y_ASSERT(ySize == p->YSize);
    for (yint y = 0; y < ySize; ++y) {
        float maxVal = 0;
        for (yint x = 0; x < xSize; ++x) {
            maxVal = Max<float>(maxVal, fabs(data[y][x]));
        }
        if (maxVal == 0) {
            p->RowScale[y] = 0;
        } else {
            float scale = maxVal / 127;
            float mult = 1 / scale;
            p->RowScale[y] = scale;
            ConvertArray(p->GetRow(y), &data[y][0], xSize, _mm256_set1_ps(mult));
        }
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
static void Add1(yint sz, const ui64 *a, ui64 *tail)
{
    for (yint k = 0; k < sz; ++k) {
        ui64 a1 = a[k];
        tail[k] = a1;
    }
}

static void Add2(yint sz, const ui64 *a, ui64 *tail, ui64 *res)
{
    for (yint k = 0; k < sz; ++k) {
        ui64 a1 = a[k];
        ui64 a2 = tail[k];
        res[k] = a1 & a2;
        tail[k] = a1 ^ a2;
    }
}

static void Add3(yint sz, const ui64 *a, const ui64 *b, ui64 *tail, ui64 *res)
{
    for (yint k = 0; k < sz; ++k) {
        ui64 a1 = a[k];
        ui64 a2 = b[k];
        ui64 a3 = tail[k];
        res[k] = (a1 & a2) | (a1 & a3) | (a2 & a3);
        tail[k] = a1 ^ a2 ^ a3;
    }
}


void SumBitDelta(const TModelMatrixBitDelta &a, const TModelMatrixBitDelta &b, TModelMatrixBitDeltaTail *pTail, TModelMatrixBitDelta *pRes)
{
    if (a.IsEmpty() && b.IsEmpty()) {
        // zero delta
        pRes->Clear();
        return;
    }
    if (a.HasRowDisp) {
        Y_VERIFY(b.HasRowDisp);
        yint ySize = YSize(a.DeltaRowSum2);
        yint width = YSize(a.BitDelta) / ySize;
        pRes->HasRowDisp = true;
        pRes->BitDelta.yresize(ySize * width);
        pRes->DeltaRowSum2.yresize(ySize);
        for (yint y = 0; y < ySize; ++y) {
            // sum deltas
            const ui64 *aBits = &a.BitDelta[y * width];
            const ui64 *bBits = &b.BitDelta[y * width];
            ui64 *tailBits = &pTail->BitDelta[y * width];
            ui64 *resBits = &pRes->BitDelta[y * width];
            if (pTail->HasDelta[y]) {
                Add3(width, aBits, bBits, tailBits, resBits);
            } else {
                Add1(width, aBits, tailBits);
                Add2(width, bBits, tailBits, resBits);
                pTail->HasDelta[y] = true;
            }
            // row sum2
            pRes->DeltaRowSum2[y] = (a.DeltaRowSum2[y] + b.DeltaRowSum2[y]) * 0.5f;
        }

    } else {
        Y_VERIFY(!b.HasRowDisp);
        yint sz = YSize(a.BitDelta);
        Y_VERIFY(YSize(b.BitDelta) == sz);
        Y_VERIFY(YSize(pTail->BitDelta) == sz);
        pRes->HasRowDisp = false;
        pRes->BitDelta.yresize(sz);
        Add3(sz, a.BitDelta.data(), b.BitDelta.data(), pTail->BitDelta.data(), pRes->BitDelta.data());
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
bool TModelMatrixData::AddDelta(const TModelMatrixHalfDelta &delta, float rowDispDecay, float step, float shrinkMult)
{
    yint xSize = Matr.GetXSize();
    yint ySize = Matr.GetYSize();

    if (HasRowDisp()) {
        SumWeight = SumWeight * rowDispDecay + 1;
        float m2scale = 1 / SumWeight;
        for (int y = 0; y < ySize; ++y) {
            RowDisp[y] *= rowDispDecay;
            RowSum2Cache[y] *= Sqr(shrinkMult);
            RowScale[y] *= shrinkMult;
        }
        for (yint y = 0; y < ySize; ++y) {
            const TModelMatrixHalfDelta::TRow &row = delta.Rows[y];
            float deltaSum2 = row.Sum2;
            if (deltaSum2 > 0) {
                RowDisp[y] += deltaSum2;
                // add row
                __m256 rowSum2 = _mm256_setzero_ps();
                const __m128i *deltaPtr = (const __m128i *)delta.GetRow(y);
                __m256 *matrPtr = (__m256 *) & Matr[y][0];
                __m256 scale = _mm256_set1_ps(row.Scale * step / sqrt(RowDisp[y] * m2scale));
                __m256 shrink = _mm256_set1_ps(RowScale[y]);
                for (yint x8 = 0; x8 < xSize / 8; ++x8) {
                    __m256 deltaVal = _mm256_cvtph_ps(deltaPtr[x8]);
                    __m256 val = _mm256_add_ps(_mm256_mul_ps(matrPtr[x8], shrink), _mm256_mul_ps(deltaVal, scale));
                    matrPtr[x8] = val;
                    rowSum2 = _mm256_add_ps(rowSum2, _mm256_mul_ps(val, val));
                }
                RowSum2Cache[y] = HorizontalSum(rowSum2);
                RowScale[y] = 1;
            }
        }
        Sum2 = CalcSum2Cached();

    } else {
        float sum2 = delta.CalcSum2();
        if (sum2 == 0 && shrinkMult == 1) {
            return false;
        }
        // fast add delta
        float globalScale = (sum2 > 0) ? step / sqrtf(sum2) : 0;
        __m256 shrink = _mm256_set1_ps(shrinkMult);
        __m256 newSum2 = _mm256_setzero_ps();
        for (int y = 0; y < ySize; ++y) {
            const TModelMatrixHalfDelta::TRow &row = delta.Rows[y];
            __m256 scale = _mm256_set1_ps(globalScale * row.Scale);
            __m256 rowSum2 = _mm256_setzero_ps();
            const __m128i *deltaPtr = (const __m128i *)delta.GetRow(y);
            __m256 *matrPtr = (__m256 *) & Matr[y][0];
            for (int x8 = 0; x8 < xSize / 8; ++x8) {
                __m256 deltaVal = _mm256_cvtph_ps(deltaPtr[x8]);
                __m256 val = _mm256_add_ps(_mm256_mul_ps(matrPtr[x8], shrink), _mm256_mul_ps(deltaVal, scale));
                matrPtr[x8] = val;
                rowSum2 = _mm256_add_ps(rowSum2, _mm256_mul_ps(val, val));
            }
            newSum2 = _mm256_add_ps(newSum2, rowSum2);
        }
        Sum2 = HorizontalSum(newSum2);
    }
    return true;
}


static ui64 ByteMaskToInt[256];
static struct TInitByteMaskToInt
{
    TInitByteMaskToInt()
    {
        for (yint k = 0; k < 256; ++k) {
            ui64 res = 0;
            for (yint b = 0; b < 8; ++b) {
                if (k & (1ll << b)) {
                    res |= 0xffull << (b * 8);
                }
            }
            ByteMaskToInt[k] = res;
        }
    }
} initByteMaskToInt;


inline __m256 AddBitLine(float *matrPtrArg, ui8 *bitDeltaPtr, yint xSize, __m256 allSignBits, __m256 scale, __m256 shrink)
{
    __m256 rowSum2 = _mm256_setzero_ps();
    __m256 *matrPtr = (__m256 *) matrPtrArg;
    for (int x8 = 0; x8 < xSize / 8; ++x8) {
        ui64 byteMask = ByteMaskToInt[bitDeltaPtr[x8]];
        __m256i mask = _mm256_cvtepi8_epi32(_mm_set_epi64x(0, byteMask));
        __m256 signBits = _mm256_and_ps(allSignBits, _mm256_castsi256_ps(_mm256_cvtepi8_epi32(_mm_set_epi64x(0, byteMask))));
        __m256 val = _mm256_add_ps(_mm256_mul_ps(matrPtr[x8], shrink), _mm256_xor_ps(signBits, scale));
        matrPtr[x8] = val;
        rowSum2 = _mm256_add_ps(rowSum2, _mm256_mul_ps(val, val));
    }
    return rowSum2;
}


bool TModelMatrixData::AddBitDelta(const TModelMatrixBitDelta &bitDelta, float rowDispDecay, float step, float shrinkMult)
{
    if (bitDelta.IsEmpty()) {
        return false;
    }
    yint xSize = GetXSize();
    yint ySize = GetYSize();
    __m256 allSignBits = _mm256_castsi256_ps(_mm256_set1_epi32(0x80000000));

    __m256 shrink = _mm256_set1_ps(shrinkMult);
    if (HasRowDisp()) {
        // fast add delta with separate row dispersion
        __m256 scale = _mm256_set1_ps(step / sqrtf(xSize));
        for (yint y = 0; y < ySize; ++y) {
            ui8 *bitDeltaPtr = (ui8 *)&bitDelta.BitDelta[y * xSize / 64];
            __m256 rowSum2 = AddBitLine(&Matr[y][0], bitDeltaPtr, xSize, allSignBits, scale, shrink);
            RowSum2Cache[y] = HorizontalSum(rowSum2);
        }
        // update row disp
        SumWeight = SumWeight * rowDispDecay + 1;
        for (int y = 0; y < ySize; ++y) {
            RowDisp[y] *= rowDispDecay;
            RowDisp[y] += bitDelta.DeltaRowSum2[y];
        }
        Sum2 = CalcSum2Cached();

    } else {
        // fast add delta
        __m256 scale = _mm256_set1_ps(step / sqrtf(xSize * ySize));
        __m256 newSum2 = _mm256_setzero_ps();
        for (int y = 0; y < ySize; ++y) {
            ui8 *bitDeltaPtr = (ui8 *)&bitDelta.BitDelta[y * xSize / 64];
            __m256 rowSum2 = AddBitLine(&Matr[y][0], bitDeltaPtr, xSize, allSignBits, scale, shrink);
            newSum2 = _mm256_add_ps(newSum2, rowSum2);
        }
        Sum2 = HorizontalSum(newSum2);
    }
    return true;
}


inline void CompressLine(ui8 *resPtr, const __m128i *deltaPtr, __m256 deltaScale, __m256 *deltaTailPtr, yint xSize, __m256 allSignBits, __m256 basicStep)
{
    for (yint x8 = 0; x8 < xSize / 8; ++x8) {
        __m256 deltaVal = _mm256_mul_ps(_mm256_cvtph_ps(deltaPtr[x8]), deltaScale);
        // val = tail + delta
        __m256 val = _mm256_add_ps(deltaTailPtr[x8], deltaVal);
        // signBit = val > 0
        __m256 signBit = _mm256_and_ps(allSignBits, val);
        // add = (val > 0) ? basicStep : -basicStep
        __m256 add = _mm256_or_ps(signBit, basicStep);
        // tail = val - add
        deltaTailPtr[x8] = _mm256_sub_ps(val, add);
        resPtr[x8] = _mm256_movemask_ps(signBit);
    }
}

void TModelMatrixData::CompressDelta(const TModelMatrixHalfDelta &delta, TModelMatrixBitDelta *pBitDelta, TArray2D<float> *pDeltaTail)
{
    TArray2D<float> &deltaTail = *pDeltaTail;
    yint xSize = Matr.GetXSize();
    yint ySize = Matr.GetYSize();
    Y_ASSERT((xSize % 64) == 0);

    if (HasRowDisp()) {
        __m256 allSignBits = _mm256_castsi256_ps(_mm256_set1_epi32(0x80000000));

        pBitDelta->HasRowDisp = true;
        pBitDelta->DeltaRowSum2.yresize(ySize);
        pBitDelta->BitDelta.yresize(ySize * xSize / 64);
        for (yint y = 0; y < ySize; ++y) {
            const TModelMatrixHalfDelta::TRow &row = delta.Rows[y];
            float deltaSum2 = row.Sum2;
            pBitDelta->DeltaRowSum2[y] = deltaSum2;

            // each row has separate scale
            // take into account current delta dispersion (somehow gives better results)
            float rowDispEstimate = (RowDisp[y] + deltaSum2) / (SumWeight + 1);
            __m256 basicStep = _mm256_set1_ps(sqrt(rowDispEstimate / xSize));

            const __m128i *deltaPtr = (const __m128i *)delta.GetRow(y);
            __m256 deltaScale = _mm256_set1_ps(row.Scale);
            __m256 *deltaTailPtr = (__m256 *) & deltaTail[y][0];
            ui8 *resPtr = (ui8 *)&pBitDelta->BitDelta[y * xSize / 64];
            CompressLine(resPtr, deltaPtr, deltaScale, deltaTailPtr, xSize, allSignBits, basicStep);
        }

    } else {
        float sum2 = delta.CalcSum2();
        if (sum2 == 0) {
            pBitDelta->Clear();
            return;
        }
        __m256 basicStep = _mm256_set1_ps(sqrt(sum2 / (xSize * ySize)));

        pBitDelta->HasRowDisp = false;
        pBitDelta->BitDelta.yresize(xSize * ySize / 64);

        __m256 allSignBits = _mm256_castsi256_ps(_mm256_set1_epi32(0x80000000));
        for (yint y = 0; y < ySize; ++y) {
            const TModelMatrixHalfDelta::TRow &row = delta.Rows[y];
            const __m128i *deltaPtr = (const __m128i *)delta.GetRow(y);
            __m256 deltaScale = _mm256_set1_ps(row.Scale);
            __m256 *deltaTailPtr = (__m256 *) & deltaTail[y][0];
            ui8 *resPtr = (ui8 *)&pBitDelta->BitDelta[y * xSize / 64];
            CompressLine(resPtr, deltaPtr, deltaScale, deltaTailPtr, xSize, allSignBits, basicStep);
        }
    }
}
