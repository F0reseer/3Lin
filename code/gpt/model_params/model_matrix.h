#pragma once


class TModelMatrixRowDisp
{
    TArray2D<float> Matr;
    float SumWeight = 0;
    TVector<float> RowDisp;
public:
    SAVELOAD(Matr, SumWeight, RowDisp);

public:
    yint GetXSize() const { return Matr.GetXSize(); }
    yint GetYSize() const { return Matr.GetYSize(); }
    const TVector<float> &GetRowDisp() const { return RowDisp; }
    float GetSumWeight() const { return SumWeight; }
    TArray2D<float> &GetMatrix() { return Matr; }
    const TArray2D<float> &GetMatrix() const { return Matr; }

    // set ops
    void SetMatrix(const TArray2D<float> &data)
    {
        Matr = data;
        SumWeight = 0;
        ClearPodArray(&RowDisp, data.GetYSize());
    }
    void SetMatrix(const TArray2D<float> &data, const TVector<float> &rowDisp, float sumWeight)
    {
        Matr = data;
        SumWeight = sumWeight;
        RowDisp = rowDisp;
    }

    // row disp ops
    void SetRowDisp(const TVector<float> &rowDisp, float sumWeight)
    {
        SumWeight = sumWeight;
        RowDisp = rowDisp;
    }
    void ScaleRowDisp(float scale)
    {
        for (float &x : RowDisp) {
            x *= scale;
        }
    }
    void AddRowDisp(const TVector<float> &rowDisp, float sumWeight, float scale)
    {
        if (sumWeight == 0 || scale == 0) {
            return;
        }
        if (sumWeight > SumWeight) {
            if (SumWeight > 0) {
                for (yint k = 0; k < YSize(rowDisp); ++k) {
                    RowDisp[k] *= sumWeight / SumWeight;
                }
            }
            SumWeight = sumWeight;
        }
        for (yint k = 0; k < YSize(rowDisp); ++k) {
            RowDisp[k] += rowDisp[k] * SumWeight / sumWeight * scale;
        }
    }
};
