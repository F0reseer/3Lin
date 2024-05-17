#pragma once
#include "par_matrix.h"
#include <util/thread.h>

namespace NCuda
{
///////////////////////////////////////////////////////////////////////////////////////////////////
// CPU matrix add
class TGraph;

class TCudaModelMatrixScale : public TThrRefBase
{
    TIntrusivePtr<TModelMatrixScale> MatrixScale;
    TCudaVector<float> MatrixScaleDevice;
    TCudaVector<int> CopyDeltaLock; // allow single concurrent delta copy over pcie
public:
    TCudaModelMatrixScale(TIntrusivePtr<TModelMatrixScale> pScale, TStream &stream);
    TIntrusivePtr<TModelMatrixScale> GetMatrixScale()
    {
        return MatrixScale;
    }
    void CopyToDevice(TIntrusivePtr<TGraph> c);
    TCudaPOD<float> GetElement(yint idx) const
    {
        return MatrixScaleDevice.GetElement(idx);
    }
    TCudaPOD<int> GetCopyLock() const { return CopyDeltaLock.GetElement(0); }
};


enum EModelMatrixMemory
{
    MM_MEM_HOST,
    MM_MEM_DEVICE,
};

class TCudaModelMatrix : public TThrRefBase
{
    yint DeviceId = 0;
    EModelMatrixMemory Mem = MM_MEM_DEVICE;
    TIntrusivePtr<TModelMatrix> Matrix;
    TIntrusivePtr<TCudaModelMatrixScale> CudaMatrixScale;
    TCuda2DArray<TFastMatrixFloat> FastDevice;

public:
    TCudaModelMatrix(yint deviceId, TIntrusivePtr<TCudaModelMatrixScale> pCudaMatrixScale, TIntrusivePtr<TModelMatrix> pMatrix, EModelMatrixMemory mmm);
    void CopyDeltaToHostAndApply(TIntrusivePtr<TGraph> c, TCuda2DArray<float> &delta, TCudaVector<int> &iterCounter);
    void CopyToDevice(TIntrusivePtr<TGraph> c);
    void ApplyHostDelta(TIntrusivePtr<TGraph> c, TCudaVector<int> &iterCounter);
    void SetNonzeroRows(const TVector<ui32> &nonzeroRows)
    {
        Matrix->SetNonzeroRows(DeviceId, nonzeroRows);
    }
    //
    TCuda2DArray<TFastMatrixFloat> &GetFast()
    {
        if (Mem == MM_MEM_HOST) {
            return Matrix->GetFastHost();
        } else {
            return FastDevice;
        }
    }
    TCuda2DArray<float> &GetDeltaHost() { return Matrix->GetDelta(DeviceId); }
    TCudaPOD<float> GetScale() const { return CudaMatrixScale->GetElement(Matrix->GetMatrixScaleIndex()); }
};

void IncrementIterCounter(TIntrusivePtr<TGraph> c, TCudaVector<int> &iterCounter);

}
