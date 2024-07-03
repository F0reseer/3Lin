#pragma once
#include <gpt/model_params/model_matrix.h>
#include <gpt/model_params/sse_utils.h>
#include <immintrin.h>


///////////////////////////////////////////////////////////////////////////////////////////////////
struct TModelMatrixInt8Delta
{
    i8 *Data = 0;
    yint Stride = 0;
    float *RowScale = 0;
    yint XSize = 0;
    yint YSize = 0;

    TModelMatrixInt8Delta(void *data, yint strideInBytes, float *rowScale, yint xSize, yint ySize) : Data((i8 *)data), Stride(strideInBytes), RowScale(rowScale), XSize(xSize), YSize(ySize) {}
    const i8 *GetRow(yint y) const { return (Data + y * Stride); }
    i8 *GetRow(yint y) { return (Data + y * Stride); }
};

struct TModelMatrixHalfDelta
{
    struct TRow
    {
        float Scale;
        float Sum2; // sum2 of scaled values
    };
    yint SizeX = 0;
    yint SizeY = 0;
    TVector<ui16> Delta;
    TVector<TRow> Rows;

public:
    void Init(yint xSize, yint ySize)
    {
        SizeX = xSize;
        SizeY = ySize;
        ClearPodArray(&Delta, xSize * ySize);
        ClearPodArray(&Rows, ySize);
    }
    float CalcSum2() const
    {
        float sum2 = 0;
        for (const TRow &row : Rows) {
            sum2 += row.Sum2;
        }
        return sum2;
    }
    const ui16 *GetRow(yint y) const { return &Delta[y * SizeX]; }
    ui16 *GetRow(yint y) { return &Delta[y * SizeX]; }
    void GetAllData(TArray2D<float> *p) const;
};

void Copy(TModelMatrixHalfDelta *p, const TModelMatrixInt8Delta &delta);
void Add(TModelMatrixHalfDelta *p, const TModelMatrixInt8Delta &delta);
void Compress(TModelMatrixInt8Delta *p, const TArray2D<float> &data);


///////////////////////////////////////////////////////////////////////////////////////////////////
struct TModelMatrixBitDelta
{
    bool HasRowDisp = false;
    TVector<float> DeltaRowSum2;
    TVector<ui64> BitDelta;
    SAVELOAD(HasRowDisp, DeltaRowSum2, BitDelta);

    bool IsEmpty() const
    {
        return BitDelta.empty();
    }
    void Clear()
    {
        DeltaRowSum2.resize(0);
        BitDelta.resize(0);
    }
    void Swap(TModelMatrixBitDelta *p)
    {
        DoSwap(HasRowDisp, p->HasRowDisp);
        DeltaRowSum2.swap(p->DeltaRowSum2);
        BitDelta.swap(p->BitDelta);
    }
};


struct TModelMatrixBitDeltaTail
{
    TVector<ui64> BitDelta;
    TVector<bool> HasDelta;
    yint Width = 0;

    void Init(yint xSize, yint ySize, bool hasRowDisp)
    {
        Y_VERIFY((xSize % 64) == 0);
        if (hasRowDisp) {
            ClearPodArray(&HasDelta, ySize);
        }
        ClearPodArray(&BitDelta, ySize * xSize / 64);
        Width = xSize / 64;
    }
};


void SumBitDelta(const TModelMatrixBitDelta &a, const TModelMatrixBitDelta &b, TModelMatrixBitDeltaTail *pTail, TModelMatrixBitDelta *pRes);


///////////////////////////////////////////////////////////////////////////////////////////////////
enum EModelMatrixUseRowDisp
{
    MM_DISP_ROW,
    MM_DISP_MATRIX,
};

class TModelMatrixData
{
    TArray2D<float> Matr;
    float SumWeight = 0;
    TVector<float> RowDisp;
    TVector<float> RowScale;
    float Sum2 = 0;
    TVector<float> RowSum2Cache;
public:
    SAVELOAD(Matr, SumWeight, RowDisp, RowScale);

private:
    void CacheRowSum2()
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

    float CalcSum2Cached()
    {
        yint ySize = Matr.GetYSize();
        float newSum2 = 0;
        for (int y = 0; y < ySize; ++y) {
            newSum2 += RowSum2Cache[y];
        }
        return newSum2;
    }

    void ApplyRowScale()
    {
        yint xSize = Matr.GetXSize();
        yint ySize = Matr.GetYSize();
        for (yint y = 0; y < ySize; ++y) {
            float mult = RowScale[y];
            if (mult != 1) {
                for (yint x = 0; x < xSize; ++x) {
                    Matr[y][x] *= mult;
                }
                RowScale[y] = 1;
            }
        }
    }

    void OnDataUpdate()
    {
        if (HasRowDisp()) {
            RowScale.resize(0);
            RowScale.resize(Matr.GetYSize(), 1.0f);
            CacheRowSum2();
            Sum2 = CalcSum2Cached();
        } else {
            Sum2 = CalcMatrixSum2(Matr);
        }
    }

public:
    yint GetXSize() const { return Matr.GetXSize(); }
    yint GetYSize() const { return Matr.GetYSize(); }
    bool HasRowDisp() const
    {
        return !RowDisp.empty();
    }

    // direct data access
    float GetRowScale(yint y) const { return RowScale[y]; }
    float *GetRow(yint y) { return &Matr[y][0]; }
    float GetSum2() const { return Sum2; }

    // Set/Get ops
    void GetData(TArray2D<float> *p)
    {
        Y_VERIFY(!HasRowDisp());
        *p = Matr;
    }
    void GetData(TModelMatrixRowDisp *p)
    {
        Y_VERIFY(HasRowDisp());
        ApplyRowScale();
        p->SetMatrix(Matr, RowDisp, SumWeight);
    }
    void Init(yint xSize, yint ySize, EModelMatrixUseRowDisp useRowDisp)
    {
        *this = TModelMatrixData();
        Matr.SetSizes(xSize, ySize);
        Matr.FillZero();
        if (useRowDisp == MM_DISP_ROW) {
            SumWeight = 0;
            ClearPodArray(&RowDisp, ySize);
        }
        OnDataUpdate();
    }
    void SetData(const TArray2D<float> &data)
    {
        Y_VERIFY(!HasRowDisp());
        Y_VERIFY(data.GetXSize() == Matr.GetXSize() && data.GetYSize() == Matr.GetYSize());
        Matr = data;
        OnDataUpdate();
    }
    void SetData(const TModelMatrixRowDisp &data)
    {
        Y_VERIFY(HasRowDisp());
        Y_VERIFY(data.GetXSize() == Matr.GetXSize() && data.GetYSize() == Matr.GetYSize());
        Matr = data.GetMatrix();
        RowDisp = data.GetRowDisp();
        SumWeight = data.GetSumWeight();
        OnDataUpdate();
    }

    // delta ops
    bool AddDelta(const TModelMatrixHalfDelta &delta, float rowDispDecay, float step, float shrinkMult);
    bool AddBitDelta(const TModelMatrixBitDelta &bitDelta, float rowDispDecay, float step, float shrinkMult);
    void CompressDelta(const TModelMatrixHalfDelta &delta, TModelMatrixBitDelta *pBitDelta, TArray2D<float> *pDeltaTail);
};
