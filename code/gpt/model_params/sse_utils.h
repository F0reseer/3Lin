#pragma once
#include <immintrin.h>


///////////////////////////////////////////////////////////////////////////////////////////////////
// https://stackoverflow.com/questions/51778721/how-to-convert-32-bit-float-to-8-bit-signed-char-41-packing-of-int32-to-int8
inline __m256i PackFloatToInt8(const __m256 *src, __m256 mult)
{
    // _mm256_loadu_ps() not needed, we expect aligned addresses
    __m256i a = _mm256_cvtps_epi32(_mm256_mul_ps(src[0], mult));
    __m256i b = _mm256_cvtps_epi32(_mm256_mul_ps(src[1], mult));
    __m256i c = _mm256_cvtps_epi32(_mm256_mul_ps(src[2], mult));
    __m256i d = _mm256_cvtps_epi32(_mm256_mul_ps(src[3], mult));
    __m256i ab = _mm256_packs_epi32(a, b);        // 16x int16_t
    __m256i cd = _mm256_packs_epi32(c, d);
    __m256i abcd = _mm256_packs_epi16(ab, cd);   // 32x int8_t
    // packed to one vector, but in [ a_lo, b_lo, c_lo, d_lo | a_hi, b_hi, c_hi, d_hi ] order
    // if you can deal with that in-memory format (e.g. for later in-lane unpack), great, you're done

    // but if you need sequential order, then vpermd:
    __m256i lanefix = _mm256_permutevar8x32_epi32(abcd, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
    return lanefix;
}

inline void ConvertArray(i8 *dst, const float *src, yint xSize, __m256 mult)
{
    for (yint x = 0; x < xSize; x += 32) {
        *(__m256i *)(dst + x) = PackFloatToInt8((const __m256 *)(src + x), mult);
    }
}


inline void UnpackArray(float *dst, const i8 *src, yint xSize, __m256 mult)
{
    for (yint x = 0; x < xSize; x += 8) {
        __m128i src8 = _mm_cvtsi64_si128(*(const i64*)(src + x));
        __m256i src32 = _mm256_cvtepi8_epi32(src8);
        __m256 srcVal = _mm256_cvtepi32_ps(src32);
        *(__m256 *)(dst + x) = _mm256_mul_ps(srcVal, mult);
    }
}

inline void AddPackedArray(float *dst, const i8 *src, yint xSize, __m256 mult)
{
    for (yint x = 0; x < xSize; x += 8) {
        __m128i src8 = _mm_cvtsi64_si128(*(const i64 *)(src + x));
        __m256i src32 = _mm256_cvtepi8_epi32(src8);
        __m256 srcVal = _mm256_cvtepi32_ps(src32);
        __m256 *dstPtr = (__m256 *)(dst + x);
        *dstPtr = _mm256_add_ps(*dstPtr, _mm256_mul_ps(srcVal, mult));
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// fp16
static void ConvertToFp16(ui16 *dst, const float *src, yint xSize, __m256 mult)
{
    Y_ASSERT((xSize & 7) == 0);
    for (yint x = 0; x < xSize; x += 8) {
        // Load 8 floats from the input vector into a 256-bit register
        __m256 val = _mm256_mul_ps(_mm256_load_ps(src + x), mult);
        // Convert the 8 floats to 8 fp16 values and store them in a 128-bit register
        __m128i res = _mm256_cvtps_ph(val, 0);
        *(__m128i *)(dst + x) = res;
    }
}

inline void UnpackFp16Array(float *dst, const ui16 *src, yint xSize, __m256 mult)
{
    Y_ASSERT((xSize & 7) == 0);
    for (yint x = 0; x < xSize; x += 8) {
        __m256 srcVal = _mm256_cvtph_ps(*(__m128i *)(src + x));
        *(__m256 *)(dst + x) = _mm256_mul_ps(srcVal, mult);
    }
}

inline void AddPackedFp16Array(float *dst, const ui16 *src, yint xSize, __m256 mult)
{
    Y_ASSERT((xSize & 7) == 0);
    for (yint x = 0; x < xSize; x += 8) {
        __m256 srcVal = _mm256_cvtph_ps(*(__m128i *)(src + x));
        __m256 *dstPtr = (__m256 *)(dst + x);
        *dstPtr = _mm256_add_ps(*dstPtr, _mm256_mul_ps(srcVal, mult));
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// x = ( x7, x6, x5, x4, x3, x2, x1, x0 )
inline float HorizontalSum(__m256 x)
{
    // hiQuad = ( x7, x6, x5, x4 )
    const __m128 hiQuad = _mm256_extractf128_ps(x, 1);
    // loQuad = ( x3, x2, x1, x0 )
    const __m128 loQuad = _mm256_castps256_ps128(x);
    // sumQuad = ( x3 + x7, x2 + x6, x1 + x5, x0 + x4 )
    const __m128 sumQuad = _mm_add_ps(loQuad, hiQuad);
    // loDual = ( -, -, x1 + x5, x0 + x4 )
    const __m128 loDual = sumQuad;
    // hiDual = ( -, -, x3 + x7, x2 + x6 )
    const __m128 hiDual = _mm_movehl_ps(sumQuad, sumQuad);
    // sumDual = ( -, -, x1 + x3 + x5 + x7, x0 + x2 + x4 + x6 )
    const __m128 sumDual = _mm_add_ps(loDual, hiDual);
    // lo = ( -, -, -, x0 + x2 + x4 + x6 )
    const __m128 lo = sumDual;
    // hi = ( -, -, -, x1 + x3 + x5 + x7 )
    const __m128 hi = _mm_shuffle_ps(sumDual, sumDual, 0x1);
    // sum = ( -, -, -, x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7 )
    const __m128 sum = _mm_add_ss(lo, hi);
    return _mm_cvtss_f32(sum);
}


static __m256 CalcRowSum2(const float *src, yint xSize)
{
    Y_ASSERT((xSize & 7) == 0);
    __m256 rowSum2 = _mm256_setzero_ps();
    for (yint x = 0; x < xSize; x += 8) {
        __m256 val = _mm256_load_ps(src + x);
        rowSum2 = _mm256_add_ps(rowSum2, _mm256_mul_ps(val, val));
    }
    return rowSum2;
}

template <class TMatrix>
static float CalcMatrixSum2(const TMatrix &matr)
{
    yint xSize = matr.GetXSize();
    yint ySize = matr.GetYSize();
    __m256 sum2 = _mm256_setzero_ps();
    for (yint y = 0; y < ySize; ++y) {
        __m256 rowSum2 = CalcRowSum2(matr.GetRow(y), xSize);
        sum2 = _mm256_add_ps(sum2, rowSum2);
    }
    return HorizontalSum(sum2);
}


