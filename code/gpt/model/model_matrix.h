#pragma once


struct TModelMatrixRowDisp
{
    TArray2D<float> Matr;
    float SumWeight = 0;
    TVector<float> RowDisp;
    SAVELOAD(Matr, SumWeight, RowDisp);

    yint GetXSize() const { return Matr.GetXSize(); }
    yint GetYSize() const { return Matr.GetYSize(); }
    void SetSizes(yint xSize, yint ySize)
    {
        Matr.SetSizes(xSize, ySize);
        SumWeight = 0;
        ClearPodArray(&RowDisp, ySize);
    }
    bool HasRowDisp() const
    {
        return !RowDisp.empty();
    }
    auto operator[](yint y) const
    {
        return Matr[y];
    }
    void ResetDisp()
    {
        SumWeight = 0;
        ClearPodArray(&RowDisp, YSize(RowDisp));
    }
    void StripRowDisp()
    {
        SumWeight = 0;
        RowDisp.clear();
    }
};

