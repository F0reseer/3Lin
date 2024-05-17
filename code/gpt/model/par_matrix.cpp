#include "stdafx.h"
#include "par_matrix.h"
#include <immintrin.h>


const float ROW_DISP_DECAY = 0.99f;


namespace NCuda
{

///////////////////////////////////////////////////////////////////////////////////////////////////
TModelMatrixScale::TModelMatrixScale(yint sz)
{
    MatrixScaleHost.AllocateHost(sz);
}

void TModelMatrixScale::SetScale(yint index, float val)
{
    Y_ASSERT(index >= 0 && index < MatrixScaleHost.GetSize());
    MatrixScaleHost.GetHostPtr()[index] = val;
}

float TModelMatrixScale::GetScale(yint index)
{
    Y_ASSERT(index >= 0 && index < MatrixScaleHost.GetSize());
    return MatrixScaleHost.GetHostPtr()[index];
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


static float CalcMatrixSum2(const TArray2D<float> &matr)
{
    yint xSize = matr.GetXSize();
    yint ySize = matr.GetYSize();
    __m256 sum2 = _mm256_setzero_ps();
    for (yint y = 0; y < ySize; ++y) {
        const __m256 *src = (const __m256*) & matr[y][0];
        __m256 rowSum2 = _mm256_setzero_ps();
        for (yint x8 = 0; x8 < xSize / 8; ++x8) {
            __m256 val = src[x8];
            rowSum2 = _mm256_add_ps(rowSum2, _mm256_mul_ps(val, val));
        }
        sum2 = _mm256_add_ps(sum2, rowSum2);
    }
    return HorizontalSum(sum2);
}


float CalcMatrixSum2(const TCuda2DArray<float> &matr)
{
    yint xSize = matr.GetXSize();
    yint ySize = matr.GetYSize();
    __m256 sum2 = _mm256_setzero_ps();
    TMemoryBlob mem = matr.GetHostMem();
    for (yint y = 0; y < ySize; ++y) {
        const __m256 *src = (const __m256 *) mem.GetElementAddress<float>(0, y);
        __m256 rowSum2 = _mm256_setzero_ps();
        for (yint x8 = 0; x8 < xSize / 8; ++x8) {
            __m256 val = src[x8];
            rowSum2 = _mm256_add_ps(rowSum2, _mm256_mul_ps(val, val));
        }
        sum2 = _mm256_add_ps(sum2, rowSum2);
    }
    return HorizontalSum(sum2);
}


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


static i8 QBitRecode2bit[256];
static struct TInitBitTable
{
    TInitBitTable()
    {
        for (yint a = 0; a < 256; ++a) {
            i8 x = a;
            if (x < -24) {
                QBitRecode2bit[a] = -36;
            } else if (x < 0) {
                QBitRecode2bit[a] = -12;
            } else if (x <= 24) {
                QBitRecode2bit[a] = 12;
            } else {
                QBitRecode2bit[a] = 36;
            }
        }
    }
} initBitTable;

static void ConvertToFastMatrixFloat(i8 *dst, const float *src, __m256 mult, int xSize, EModelMatrixQuant quant)
{
    for (yint x = 0; x < xSize; x += 32) {
        *(__m256i *)(dst + x) = PackFloatToInt8((const __m256 *)(src + x), mult);
    }
    // simulate quantization
    if (quant == MM_QUANT_158BIT) {
        // 1.58 bit
        for (yint x = 0; x < xSize; ++x) {
            //dst[x] = (src[x] > 0) ? 32 : -32;
            if (dst[x] < -15) {
                dst[x] = -32;
            } else if (dst[x] > 15) {
                dst[x] = 32;
            } else {
                dst[x] = 0;
            }
        }
    } else if (quant == MM_QUANT_2BIT) {
        for (yint x = 0; x < xSize; ++x) {
            dst[x] = QBitRecode2bit[(ui8)dst[x]]; // can be speed up with SSE
        }
    }
}


static void ConvertToFastMatrixFloat(half *dst, const float *src, __m256 mult, int xSize, EModelMatrixQuant quant)
{
    for (yint x = 0; x < xSize; x += 8) {
        // Load 8 floats from the input vector into a 256-bit register
        __m256 val = _mm256_mul_ps(_mm256_load_ps(src + x), mult);
        // Convert the 8 floats to 8 fp16 values and store them in a 128-bit register
        __m128i res = _mm256_cvtps_ph(val, 0);
        *(__m128i *)(dst + x) = res;
    }
    Y_VERIFY(quant == MM_QUANT_NONE);
}


inline void CopyRow(const TMemoryBlob &dstMem, const TMemoryBlob &srcMem, yint xSize, yint y)
{
    __m256 *dstPtr = (__m256 *) dstMem.GetElementAddress<float>(0, y);
    const __m256 *srcPtr = (__m256 *) srcMem.GetElementAddress<float>(0, y);
    for (yint x8 = 0; x8 < xSize / 8; ++x8) {
        dstPtr[x8] = srcPtr[x8];
    }
}

inline void AddRow(const TMemoryBlob &dstMem, const TMemoryBlob &srcMem, yint xSize, yint y)
{
    __m256 *dstPtr = (__m256 *) dstMem.GetElementAddress<float>(0, y);
    const __m256 *srcPtr = (__m256 *) srcMem.GetElementAddress<float>(0, y);
    for (yint x8 = 0; x8 < xSize / 8; ++x8) {
        dstPtr[x8] = _mm256_add_ps(dstPtr[x8], srcPtr[x8]);
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
void TModelMatrix::Allocate(yint deviceCount, TIntrusivePtr<TModelMatrixScale> pScale,
    float discrScale, yint xSize, yint ySize,
    EModelMatrixUseRowDisp useRowDisp, EModelMatrixQuant quant, EModelMatrixStaleGradient staleGrad)
{
    DiscrScale = discrScale;
    Matr.SetSizes(xSize, ySize);
    if (useRowDisp == MM_DISP_ROW) {
        SumNonzeroRows.resize(ySize);
        for (yint y = 0; y < ySize; ++y) {
            SumNonzeroRows[y] = y;
        }
    } else {
        Matr.StripRowDisp();
    }
    SumDelta.AllocateHost(xSize, ySize);
    SumDelta.ClearHostMem();
    FastHost.AllocateHost(xSize, ySize);
    MatrixScale = pScale;
    MatrixScaleIndex = pScale->GetIndex();
    Quantization = quant;
    StaleGradientAllowed = (staleGrad == MM_STALE_GRADIENT);
    DeviceArr.resize(deviceCount);
    for (yint deviceId = 0; deviceId < YSize(DeviceArr); ++deviceId) {
        DeviceArr[deviceId] = new TDeviceData;
        DeviceArr[deviceId]->NonzeroRows = SumNonzeroRows;
    }
    for (yint deviceId = 1; deviceId < YSize(DeviceArr); ++deviceId) {
        // we use SumDelta to store delta for device 0
        TDeviceData &dev = *DeviceArr[deviceId];
        dev.Delta.AllocateHost(xSize, ySize);
        dev.Delta.ClearHostMem();
    }
}


void TModelMatrix::AttachOp(int *opPointer, const TVector<TCudaPOD<int>> &cudaLaunchFlagArr)
{
    OpPointer = opPointer;
    Y_ASSERT(YSize(cudaLaunchFlagArr) == YSize(DeviceArr));
    for (yint deviceId = 0; deviceId < YSize(DeviceArr); ++deviceId) {
        TDeviceData &dev = *DeviceArr[deviceId];
        Y_ASSERT(dev.CudaLaunchFlag.GetOwner() == 0);
        dev.CudaLaunchFlag = cudaLaunchFlagArr[deviceId];
    }
}


void TModelMatrix::SumDeviceDeltas()
{
    yint xSize = Matr.GetXSize();
    yint ySize = Matr.GetYSize();
    if (Matr.HasRowDisp()) {
        TMemoryBlob deltaMem = SumDelta.GetHostMem();
        for (yint deviceId = 1; deviceId < YSize(DeviceArr); ++deviceId) {
            const TDeviceData &dev = *DeviceArr[deviceId];
            TMemoryBlob devDeltaMem = dev.Delta.GetHostMem();
            TVector<ui32> newNonzero;
            yint n1 = YSize(SumNonzeroRows);
            yint n2 = YSize(dev.NonzeroRows);
            newNonzero.resize(n1 + n2);
            yint i1 = 0;
            yint i2 = 0;
            yint dst = 0;
            while (i1 < n1 && i2 < n2) {
                yint y1 = SumNonzeroRows[i1];
                yint y2 = dev.NonzeroRows[i2];
                if (y1 < y2) {
                    newNonzero[dst++] = y1;
                    ++i1;
                } else if (y1 == y2) {
                    AddRow(deltaMem, devDeltaMem, xSize, y1);
                    newNonzero[dst++] = y1;
                    ++i1;
                    ++i2;
                } else {
                    CopyRow(deltaMem, devDeltaMem, xSize, y2);
                    newNonzero[dst++] = y2;
                    ++i2;
                }
            }
            while (i1 < n1) {
                yint y1 = SumNonzeroRows[i1];
                newNonzero[dst++] = y1;
                ++i1;
            }
            while (i2 < n2) {
                yint y2 = dev.NonzeroRows[i2];
                CopyRow(deltaMem, devDeltaMem, xSize, y2);
                newNonzero[dst++] = y2;
                ++i2;
            }
            newNonzero.resize(dst);
            newNonzero.swap(SumNonzeroRows);
        }

    } else {
        TMemoryBlob deltaMem = SumDelta.GetHostMem();
        for (yint deviceId = 1; deviceId < YSize(DeviceArr); ++deviceId) {
            const TDeviceData &dev = *DeviceArr[deviceId];
            TMemoryBlob devDeltaMem = dev.Delta.GetHostMem();
            for (int y = 0; y < ySize; ++y) {
                AddRow(deltaMem, devDeltaMem, xSize, y);
            }
        }
    }
}


void TModelMatrix::Convert()
{
    yint xSize = Matr.GetXSize();
    yint ySize = Matr.GetYSize();
    float sko = sqrt(Sum2 / (xSize * ySize));
    float discrScale = sko * DiscrScale;
    __m256 mult = _mm256_set1_ps((sko == 0) ? 0 : (1 / discrScale));

    MatrixScale->SetScale(MatrixScaleIndex, discrScale);

    TMemoryBlob fastMem = FastHost.GetHostMem();
    for (yint y = 0; y < ySize; ++y) {
        TFastMatrixFloat *dst = fastMem.GetElementAddress<TFastMatrixFloat>(0, y);
        const float *src = &Matr[y][0];
        ConvertToFastMatrixFloat(dst, src, mult, xSize, Quantization);
    }
}


void TModelMatrix::CacheRowSum2()
{
    yint xSize = Matr.GetXSize();
    yint ySize = Matr.GetYSize();
    RowSum2Cache.resize(ySize);
    for (int y = 0; y < ySize; ++y) {
        __m256 *matrPtr = (__m256 *) & Matr[y][0];
        __m256 rowSum2 = _mm256_setzero_ps();
        for (yint x8 = 0; x8 < xSize / 8; ++x8) {
            __m256 val = matrPtr[x8];
            rowSum2 = _mm256_add_ps(rowSum2, _mm256_mul_ps(val, val));
        }
        RowSum2Cache[y] = HorizontalSum(rowSum2);
    }
}


float TModelMatrix::CalcSum2Cached()
{
    yint ySize = Matr.GetYSize();
    float newSum2 = 0;
    for (int y = 0; y < ySize; ++y) {
        newSum2 += RowSum2Cache[y];
    }
    return newSum2;
}


void TModelMatrix::AddDelta(float step)
{
    Y_ASSERT(*OpPointer == OP_ADD_DELTA);
    yint xSize = Matr.GetXSize();
    yint ySize = Matr.GetYSize();

    if (Matr.HasRowDisp()) {
        Matr.SumWeight = Matr.SumWeight * ROW_DISP_DECAY + 1;
        float m2scale = 1 / Matr.SumWeight;
        TMemoryBlob deltaMem = SumDelta.GetHostMem();
        for (int y = 0; y < ySize; ++y) {
            Matr.RowDisp[y] *= ROW_DISP_DECAY;
        }
        for (yint y : SumNonzeroRows) {
            const __m256 *deltaPtr = (const __m256 *)deltaMem.GetElementAddress<float>(0, y);

            // compute row disp
            __m256 deltaRowSum2 = _mm256_setzero_ps();
            for (yint x8 = 0; x8 < xSize / 8; ++x8) {
                __m256 val = deltaPtr[x8];
                deltaRowSum2 = _mm256_add_ps(deltaRowSum2, _mm256_mul_ps(val, val));
            }
            float deltaSum2 = HorizontalSum(deltaRowSum2);
            if (deltaSum2 > 0) {
                Matr.RowDisp[y] += deltaSum2;

                // add row
                __m256 rowSum2 = _mm256_setzero_ps();
                __m256 *matrPtr = (__m256 *) & Matr[y][0];
                __m256 scale = _mm256_set1_ps(step / sqrt(Matr.RowDisp[y] * m2scale));
                for (yint x8 = 0; x8 < xSize / 8; ++x8) {
                    __m256 val = _mm256_add_ps(matrPtr[x8], _mm256_mul_ps(deltaPtr[x8], scale));
                    matrPtr[x8] = val;
                    rowSum2 = _mm256_add_ps(rowSum2, _mm256_mul_ps(val, val));
                }
                RowSum2Cache[y] = HorizontalSum(rowSum2);
            }
        }
        Sum2 = CalcSum2Cached();

    } else {
        float sum2 = CalcMatrixSum2(SumDelta);
        if (sum2 == 0) {
            SetOp(OP_NONE);
            return;
        }
        // fast add delta
        __m256 scale = _mm256_set1_ps(step / sqrtf(sum2));
        __m256 newSum2 = _mm256_setzero_ps();
        TMemoryBlob deltaMem = SumDelta.GetHostMem();
        for (int y = 0; y < ySize; ++y) {
            __m256 rowSum2 = _mm256_setzero_ps();
            __m256 *matrPtr = (__m256*) &Matr[y][0];
            const __m256 *deltaPtr = (__m256*) deltaMem.GetElementAddress<float>(0, y);
            for (int x8 = 0; x8 < xSize / 8; ++x8) {
                __m256 val = _mm256_add_ps(matrPtr[x8], _mm256_mul_ps(deltaPtr[x8], scale));
                matrPtr[x8] = val;
                rowSum2 = _mm256_add_ps(rowSum2, _mm256_mul_ps(val, val));
            }
            newSum2 = _mm256_add_ps(newSum2, rowSum2);
        }
        Sum2 = HorizontalSum(newSum2);
    }

    Convert();
    SetOp(OP_NONE);
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


inline __m256 AddBitLine(float *matrPtrArg, ui8 *bitDeltaPtr, yint xSize, __m256 allSignBits, __m256 scale)
{
    __m256 rowSum2 = _mm256_setzero_ps();
    __m256 *matrPtr = (__m256 *) matrPtrArg;
    for (int x8 = 0; x8 < xSize / 8; ++x8) {
        ui64 byteMask = ByteMaskToInt[bitDeltaPtr[x8]];
        __m256i mask = _mm256_cvtepi8_epi32(_mm_set_epi64x(0, byteMask));
        __m256 signBits = _mm256_and_ps(allSignBits, _mm256_castsi256_ps(_mm256_cvtepi8_epi32(_mm_set_epi64x(0, byteMask))));
        __m256 val = _mm256_add_ps(matrPtr[x8], _mm256_xor_ps(signBits, scale));
        matrPtr[x8] = val;
        rowSum2 = _mm256_add_ps(rowSum2, _mm256_mul_ps(val, val));
    }
    return rowSum2;
}

void TModelMatrix::AddBitDelta(float step)
{
    Y_ASSERT(*OpPointer == OP_ADD_BIT_DELTA);
    if (BitDelta.IsEmpty()) {
        SetOp(OP_NONE);
        return;
    }
    yint xSize = Matr.GetXSize();
    yint ySize = Matr.GetYSize();
    __m256 allSignBits = _mm256_castsi256_ps(_mm256_set1_epi32(0x80000000));

    if (Matr.HasRowDisp()) {
        // fast add delta with separate row dispersion
        __m256 scale = _mm256_set1_ps(step / sqrtf(xSize));
        for (yint y = 0; y < ySize; ++y) {
            ui8 *bitDeltaPtr = (ui8 *)&BitDelta.BitDelta[y * xSize / 64];
            __m256 rowSum2 = AddBitLine(&Matr[y][0], bitDeltaPtr, xSize, allSignBits, scale);
            RowSum2Cache[y] = HorizontalSum(rowSum2);
        }
        // update row disp
        Matr.SumWeight = Matr.SumWeight * ROW_DISP_DECAY + 1;
        for (int y = 0; y < ySize; ++y) {
            Matr.RowDisp[y] *= ROW_DISP_DECAY;
            Matr.RowDisp[y] += BitDelta.DeltaRowSum2[y];
        }
        Sum2 = CalcSum2Cached();

    } else {
        // fast add delta
        __m256 scale = _mm256_set1_ps(step / sqrtf(xSize * ySize));
        __m256 newSum2 = _mm256_setzero_ps();
        for (int y = 0; y < ySize; ++y) {
            ui8 *bitDeltaPtr = (ui8*) &BitDelta.BitDelta[y * xSize / 64];
            __m256 rowSum2 = AddBitLine(&Matr[y][0], bitDeltaPtr, xSize, allSignBits, scale);
            newSum2 = _mm256_add_ps(newSum2, rowSum2);
        }
        Sum2 = HorizontalSum(newSum2);
    }
    Convert();
    SetOp(OP_NONE);
}


inline void CompressLine(ui8 *resPtr, const __m256 *deltaPtr, __m256 *deltaTailPtr, yint xSize, __m256 allSignBits, __m256 basicStep)
{
    for (yint x8 = 0; x8 < xSize / 8; ++x8) {
        // val = tail + delta
        __m256 val = _mm256_add_ps(deltaTailPtr[x8], deltaPtr[x8]);
        // signBit = val > 0
        __m256 signBit = _mm256_and_ps(allSignBits, val);
        // add = (val > 0) ? basicStep : -basicStep
        __m256 add = _mm256_or_ps(signBit, basicStep);
        // tail = val - add
        deltaTailPtr[x8] = _mm256_sub_ps(val, add);
        resPtr[x8] = _mm256_movemask_ps(signBit);
    }
}

void TModelMatrix::CompressDelta(TModelMatrixDelta *pBitDelta, TArray2D<float> *pDeltaTail)
{
    TMemoryBlob deltaMem = SumDelta.GetHostMem();
    TArray2D<float> &deltaTail = *pDeltaTail;
    yint xSize = Matr.GetXSize();
    yint ySize = Matr.GetYSize();
    Y_ASSERT((xSize % 64) == 0);

    if (Matr.HasRowDisp()) {
        __m256 allSignBits = _mm256_castsi256_ps(_mm256_set1_epi32(0x80000000));

        pBitDelta->HasRowDisp = true;
        pBitDelta->DeltaRowSum2.yresize(ySize);
        pBitDelta->BitDelta.yresize(ySize * xSize / 64);
        for (yint y = 0; y < ySize; ++y) {
            __m256 *deltaPtr = (__m256 *) deltaMem.GetElementAddress<float>(0, y);

            __m256 deltaRowSum2 = _mm256_setzero_ps();
            for (yint x8 = 0; x8 < xSize / 8; ++x8) {
                __m256 val = deltaPtr[x8];
                deltaRowSum2 = _mm256_add_ps(deltaRowSum2, _mm256_mul_ps(val, val));
            }
            float deltaSum2 = HorizontalSum(deltaRowSum2);
            pBitDelta->DeltaRowSum2[y] = deltaSum2;

            // each row has separate scale
            // take into account current delta dispersion (somehow gives better results)
            float rowDispEstimate = (Matr.RowDisp[y] + deltaSum2) / (Matr.SumWeight + 1);
            __m256 basicStep = _mm256_set1_ps(sqrt(rowDispEstimate / xSize));

            __m256 *deltaTailPtr = (__m256 *) & deltaTail[y][0];
            ui8 *resPtr = (ui8 *)&pBitDelta->BitDelta[y * xSize / 64];
            CompressLine(resPtr, deltaPtr, deltaTailPtr, xSize, allSignBits, basicStep);
        }

    } else {
        float sum2 = CalcMatrixSum2(SumDelta);
        if (sum2 == 0) {
            pBitDelta->Clear();
            return;
        }
        __m256 basicStep = _mm256_set1_ps(sqrt(sum2 / (xSize * ySize)));

        pBitDelta->HasRowDisp = false;
        pBitDelta->BitDelta.yresize(xSize * ySize / 64);

        __m256 allSignBits = _mm256_castsi256_ps(_mm256_set1_epi32(0x80000000));
        for (yint y = 0; y < ySize; ++y) {
            __m256 *deltaPtr = (__m256 *) deltaMem.GetElementAddress<float>(0, y);
            __m256 *deltaTailPtr = (__m256 *) & deltaTail[y][0];
            ui8 *resPtr = (ui8*) & pBitDelta->BitDelta[y * xSize / 64];
            CompressLine(resPtr, deltaPtr, deltaTailPtr, xSize, allSignBits, basicStep);
        }
    }
}


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


void TModelMatrix::SumBitDelta(const TModelMatrixDelta &a, const TModelMatrixDelta &b, TModelMatrixBitTail *pTail, TModelMatrixDelta *pRes)
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


void TModelMatrix::GetData(TArray2D<float> *p) const
{
    *p = Matr.Matr;
}

void TModelMatrix::GetData(TModelMatrixRowDisp *p) const
{
    Y_ASSERT(Matr.HasRowDisp());
    *p = Matr;
}

void TModelMatrix::GetDataFast(TArray2D<float> *p) const
{
    yint xSize = Matr.GetXSize();
    yint ySize = Matr.GetYSize();
    THost2DPtr<TFastMatrixFloat> src = FastHost.GetHostPtr();
    float scale = MatrixScale->GetScale(MatrixScaleIndex);
    p->SetSizes(xSize, ySize);
    for (yint y = 0; y < ySize; ++y) {
        for (yint x = 0; x < xSize; ++x) {
            (*p)[y][x] = float(src[y][x]) * scale;
        }
    }
}

void TModelMatrix::GetDeltaData(TArray2D<float> *p) const
{
    SumDelta.GetAllData(p);
}


void TModelMatrix::SetData(const TArray2D<float> &data)
{
    Y_VERIFY(!Matr.HasRowDisp());
    Y_VERIFY(data.GetXSize() == Matr.GetXSize() && data.GetYSize() == Matr.GetYSize());
    Matr.Matr = data;
    Sum2 = CalcMatrixSum2(Matr.Matr);
    Convert();
}


void TModelMatrix::SetData(const TModelMatrixRowDisp &data)
{
    Y_VERIFY(Matr.HasRowDisp());
    Y_VERIFY(data.GetXSize() == Matr.GetXSize() && data.GetYSize() == Matr.GetYSize());
    Matr = data;
    CacheRowSum2();
    Sum2 = CalcSum2Cached();
    Convert();
}


void TModelMatrix::ApplyDelta(const TArray2D<float> &data)
{
    SumDelta.PutHost(data);
    SetOp(OP_NEW_DELTA);
}


///////////////////////////////////////////////////////////////////////////////////////////////////
bool TCPUMatrixAdd::PerformAllOps(yint workerId)
{
    volatile int *matrixOpArr = MatrixOpArr.data();
    bool res = true;
    yint deviceCount = YSize(DeviceArr);
    for (yint k : WorkerArr[workerId]->WorkerMatrices) {
        volatile int &op = matrixOpArr[k];
        for (yint deviceId = 0; deviceId < deviceCount; ++deviceId) {
            TDeviceData &dev = *DeviceArr[deviceId];
            volatile int *cudaAddDeltaFlag = dev.CudaAddDeltaFlag.GetHostPtr();
            int newCudaFlag = cudaAddDeltaFlag[k];
            if (newCudaFlag != dev.PrevCudaAddDeltaFlag[k]) {
                // avoid modifying cudaAddDeltaFlag from cpu & gpu concurrently
                dev.PrevCudaAddDeltaFlag[k] = newCudaFlag;
                if (deviceCount > 1) {
                    if (++MatrixReadyDeviceCount[k] == deviceCount) {
                        MatrixReadyDeviceCount[k] = 0;
                        MatrixArr[k]->SumDeviceDeltas();
                    } else {
                        continue;
                    }
                }
                Y_VERIFY(op == TModelMatrix::OP_NONE);
                op = TModelMatrix::OP_NEW_DELTA;
            }
        }
        if (op == TModelMatrix::OP_NEW_DELTA) {
            if (DeltaHookArr[k].Get()) {
                DeltaHookArr[k]->OnDelta();
            } else {
                op = TModelMatrix::OP_ADD_DELTA;
            }
        }
        if (op == TModelMatrix::OP_ADD_DELTA) {
            MatrixArr[k]->AddDelta(Step);
        } else if (op == TModelMatrix::OP_ADD_BIT_DELTA) {
            MatrixArr[k]->AddBitDelta(Step);
        }
        res &= (op == TModelMatrix::OP_NONE);
    }
    return res;
}


void TCPUMatrixAdd::WorkerThread()
{
    yint workerId = WorkerCount.fetch_add(1);
    TWorkerData *data = WorkerArr[workerId].Get();
    while (!Exit) {
        TVector<TJob> jobArr;
        if (data->JobQueue.DequeueAll(&jobArr)) {
            // the only job is to wait all ops completion
            while (!PerformAllOps(workerId)) {
                _mm_pause();
            }
            JobCount.fetch_add(-YSize(jobArr));
        } else {
            PerformAllOps(workerId);
            _mm_pause();
        }
    }
}


TCPUMatrixAdd::~TCPUMatrixAdd()
{
    Exit = true;
}


TCPUMatrixAdd::TCPUMatrixAdd(yint deviceCount, yint maxDeltaMatrices, IMMDeltaHookGen *deltaHookGen) : DeltaHookGen(deltaHookGen), WorkerCount(0), JobCount(0)
{
    MaxDeltaMatrices = maxDeltaMatrices;
    MatrixArr.resize(maxDeltaMatrices);
    DeltaHookArr.resize(maxDeltaMatrices);
    DeviceArr.resize(deviceCount);
    for (yint deviceId = 0; deviceId < deviceCount; ++deviceId) {
        DeviceArr[deviceId] = new TDeviceData;
        TDeviceData &dev = *DeviceArr[deviceId];
        dev.CudaAddDeltaFlag.AllocateHost(maxDeltaMatrices);
        dev.CudaAddDeltaFlag.ClearHostMem();
        ClearPodArray(&dev.PrevCudaAddDeltaFlag, maxDeltaMatrices);
    }
    ClearPodArray(&MatrixOpArr, maxDeltaMatrices);
    ClearPodArray(&MatrixReadyDeviceCount, maxDeltaMatrices);
    yint workerCount = BASE_WORKER_COUNT + deviceCount * PER_GPU_WORKER_COUNT;
    WorkerArr.resize(workerCount);
    for (yint workerId = 0; workerId < workerCount; ++workerId) {
        WorkerArr[workerId] = new TWorkerData();
    }
}


void TCPUMatrixAdd::AddMatrix(TIntrusivePtr<TModelMatrix> p)
{
    yint idx = MatrixCount++;
    Y_VERIFY(idx < YSize(MatrixArr));
    MatrixArr[idx] = p;
    TVector<TCudaPOD<int>> cudaLaunchFlagArr;
    for (yint deviceId = 0; deviceId < YSize(DeviceArr); ++deviceId) {
        cudaLaunchFlagArr.push_back(DeviceArr[deviceId]->CudaAddDeltaFlag.GetElement(idx));
    }
    p->AttachOp(&MatrixOpArr[idx], cudaLaunchFlagArr);
    if (DeltaHookGen.Get()) {
        DeltaHookArr[idx] = DeltaHookGen->CreateDeltaHook(idx, p);
    }
}


struct TCPUMatrixWeight
{
    int Index = 0;
    float Weight = 0;
};

void TCPUMatrixAdd::LaunchWorkers()
{
    // load balance, assign matrix to the least loaded worker
    TVector<TCPUMatrixWeight> mwArr;
    mwArr.resize(MatrixCount);
    for (yint k = 0; k < MatrixCount; ++k) {
        TModelMatrix *p = MatrixArr[k].Get();
        mwArr[k].Index = k;
        mwArr[k].Weight = p->GetXSize() * p->GetYSize();
    }
    Sort(mwArr.begin(), mwArr.end(), [](const TCPUMatrixWeight &a, const TCPUMatrixWeight &b) { return a.Weight > b.Weight; });
    for (const TCPUMatrixWeight &mw : mwArr) {
        float minParamCount = 1e38f;
        TWorkerData *minWorker = WorkerArr[0].Get();
        for (yint workerId = 0; workerId < YSize(WorkerArr); ++workerId) {
            TWorkerData *p = WorkerArr[workerId].Get();
            if (p->ParamCount < minParamCount) {
                minParamCount = p->ParamCount;
                minWorker = p;
            }
        }
        minWorker->ParamCount += mw.Weight;
        minWorker->WorkerMatrices.push_back(mw.Index);
    }
    // launch workers
    for (TIntrusivePtr<TWorkerData> &w : WorkerArr) {
        w->Thr.Create(this);
    }
}


void TCPUMatrixAdd::StartIteration(float step)
{
    Y_ASSERT(JobCount.load() == 0);
    Step = step;
    if (DeltaHookGen.Get()) {
        DeltaHookGen->OnIterationStart();
    }
}

void TCPUMatrixAdd::Wait()
{
    // add perform all pending ops command
    TJob job;
    yint workerCount = WorkerCount.load();
    for (yint workerId = 0; workerId < workerCount; ++workerId) {
        JobCount.fetch_add(1);
        WorkerArr[workerId]->JobQueue.Enqueue(job);
    }

    // wait completion
    for (;;) {
        if (JobCount.load() == 0) {
            break;
        }
        _mm_pause();
    }
}


void TCPUMatrixAdd::ResetIterCount()
{
    for (yint deviceId = 0; deviceId < YSize(DeviceArr); ++deviceId) {
        TDeviceData &dev = *DeviceArr[deviceId];
        ClearPodArray(&dev.PrevCudaAddDeltaFlag, YSize(dev.PrevCudaAddDeltaFlag));
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
TIntrusivePtr<TModelMatrix> CreateModelMatrix(TIntrusivePtr<TCPUMatrixAdd> cpuAdd, TIntrusivePtr<TModelMatrixScale> pScale,
    float discrScale, yint xSize, yint ySize,
    EModelMatrixUseRowDisp useRowDisp, EModelMatrixQuant quant, EModelMatrixStaleGradient staleGrad)
{
    TIntrusivePtr<TModelMatrix> res = new TModelMatrix();
    res->Allocate(cpuAdd->GetDeviceCount(), pScale, discrScale, xSize, ySize, useRowDisp, quant, staleGrad);
    cpuAdd->AddMatrix(res);
    return res;
}

}
