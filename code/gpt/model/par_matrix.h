#pragma once
#include <lib/cuda/cuda_arrays.h>
#include "model_matrix.h"
#include <util/thread.h>


namespace NCuda
{
    class TModelMatrix;
}
///////////////////////////////////////////////////////////////////////////////////////////////////
struct IMMDeltaHook : public TThrRefBase
{
    virtual void OnDelta() = 0;
};

struct IMMDeltaHookGen : public TThrRefBase
{
    virtual IMMDeltaHook *CreateDeltaHook(yint idx, TIntrusivePtr<NCuda::TModelMatrix> p) = 0;
    virtual void OnIterationStart() = 0;
};


namespace NCuda
{
///////////////////////////////////////////////////////////////////////////////////////////////////
typedef i8 TFastMatrixFloat;
//typedef half TFastMatrixFloat;

///////////////////////////////////////////////////////////////////////////////////////////////////
class TModelMatrixScale : public TThrRefBase
{
    TCudaVector<float> MatrixScaleHost;
    int IndexCount = 0;
public:
    TModelMatrixScale(yint sz);
    yint GetSize() { return MatrixScaleHost.GetSize(); }
    void SetScale(yint index, float val);
    float GetScale(yint index);
    int GetIndex()
    {
        Y_VERIFY(IndexCount < MatrixScaleHost.GetSize());
        return IndexCount++;
    }
    TCudaVector<float> &GetMatrixScaleHost()
    {
        return MatrixScaleHost;
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
float CalcMatrixSum2(const TCuda2DArray<float> &matr);


///////////////////////////////////////////////////////////////////////////////////////////////////
struct TModelMatrixDelta
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
    void Swap(TModelMatrixDelta *p)
    {
        DoSwap(HasRowDisp, p->HasRowDisp);
        DeltaRowSum2.swap(p->DeltaRowSum2);
        BitDelta.swap(p->BitDelta);
    }
};


struct TModelMatrixBitTail
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


///////////////////////////////////////////////////////////////////////////////////////////////////
// simulation model params quantization
enum EModelMatrixQuant
{
    MM_QUANT_NONE,
    MM_QUANT_158BIT, // -1 / 0 / 1
    MM_QUANT_2BIT,
};

enum EModelMatrixUseRowDisp
{
    MM_DISP_ROW,
    MM_DISP_MATRIX
};

enum EModelMatrixStaleGradient
{
    MM_SYNC_GRADIENT,
    MM_STALE_GRADIENT,
};

class TModelMatrix : public TThrRefBase
{
    struct TDeviceData : public TThrRefBase
    {
        TVector<ui32> NonzeroRows;
        TCuda2DArray<float> Delta;
        TCudaPOD<int> CudaLaunchFlag;

        TDeviceData() : CudaLaunchFlag(0, 0) {}
    };
private:
    TVector<TIntrusivePtr<TDeviceData>> DeviceArr;
    float DiscrScale = 0;
    TModelMatrixRowDisp Matr;
    TVector<float> RowSum2Cache;
    TCuda2DArray<TFastMatrixFloat> FastHost;
    TCuda2DArray<float> SumDelta;
    TVector<ui32> SumNonzeroRows;
    TModelMatrixDelta BitDelta;
    TIntrusivePtr<TModelMatrixScale> MatrixScale;
    yint MatrixScaleIndex = 0;
    bool StaleGradientAllowed = false;
    float Sum2 = 0;
    EModelMatrixQuant Quantization;
    volatile int *OpPointer = nullptr;

    void Convert();
    void CacheRowSum2();
    float CalcSum2Cached();
public:
    enum {
        OP_NONE,
        OP_NEW_DELTA,
        OP_WAIT,
        OP_ADD_DELTA,
        OP_ADD_BIT_DELTA,
    };

    void Allocate(yint deviceCount, TIntrusivePtr<TModelMatrixScale> pScale,
        float discrScale, yint xSize, yint ySize,
        EModelMatrixUseRowDisp useRowDisp, EModelMatrixQuant quant, EModelMatrixStaleGradient staleGrad);
    yint GetXSize() const { return Matr.GetXSize(); }
    yint GetYSize() const { return Matr.GetYSize(); }
    void AttachOp(int *opPointer, const TVector<TCudaPOD<int>> &cudaLaunchFlagArr);
    int GetOp() const { return *OpPointer; }
    void SetOp(int op) { *OpPointer = op; }
    TIntrusivePtr<TModelMatrixScale> GetMatrixScale() { return MatrixScale; }
    yint GetMatrixScaleIndex() const { return MatrixScaleIndex; }
    //
    void SumDeviceDeltas();
    void AddDelta(float step);
    void AddBitDelta(float step);
    void CompressDelta(TModelMatrixDelta *pBitDelta, TArray2D<float> *pDeltaTail);
    static void SumBitDelta(const TModelMatrixDelta &a, const TModelMatrixDelta &b, TModelMatrixBitTail *pTail, TModelMatrixDelta *pRes);
    void SetNonzeroRows(yint deviceId, const TVector<ui32> &nonzeroRows)
    {
        if (deviceId == 0) {
            SumNonzeroRows = nonzeroRows;
        } else {
            DeviceArr[deviceId]->NonzeroRows = nonzeroRows;
        }
    }
    bool HasRowDisp() const { return Matr.HasRowDisp(); }
    bool CanUseStaleGradient() const { return StaleGradientAllowed; }
    //
    const TArray2D<float> &GetData() const { return Matr.Matr; }
    void GetData(TArray2D<float> *p) const;
    void GetData(TModelMatrixRowDisp *p) const;
    void GetDataFast(TArray2D<float> *p) const;
    void GetDeltaData(TArray2D<float> *p) const;
    void SetData(const TArray2D<float> &data);
    void SetData(const TModelMatrixRowDisp &data);
    void ApplyDelta(const TArray2D<float> &data);
    TCuda2DArray<TFastMatrixFloat> &GetFastHost() { return FastHost; }
    TModelMatrixDelta &GetBitDelta() { return BitDelta; }
    TCuda2DArray<float> &GetDelta(yint deviceId) { return deviceId == 0 ? SumDelta : DeviceArr[deviceId]->Delta; }
    TCudaPOD<int> GetLaunchFlag(yint deviceId) const { return DeviceArr[deviceId]->CudaLaunchFlag; }
    // allow hook operations on sum delta
    TCuda2DArray<float> &GetSumDelta() { return SumDelta; }
    const TVector<ui32> &GetSumNonzeroRows() const { return SumNonzeroRows; }
    void SetSumNonzeroRows(const TVector<ui32> &nonzeroRows) { SumNonzeroRows = nonzeroRows; }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
class TCPUMatrixAdd : public TThrRefBase
{
    enum {
#ifdef _MSC_VER
        BASE_WORKER_COUNT = 2,
        PER_GPU_WORKER_COUNT = 2,
#else
        BASE_WORKER_COUNT = 8,
        PER_GPU_WORKER_COUNT = 4,
#endif
    };

    struct TJob
    {
        int Op = 0;
    };

    struct TDeviceData : public TThrRefBase
    {
        TCudaVector<int> CudaAddDeltaFlag;
        TVector<int> PrevCudaAddDeltaFlag;
    };

    struct TWorkerData : public TThrRefBase
    {
        TSingleConsumerJobQueue<TJob> JobQueue;
        TThread Thr;
        TVector<int> WorkerMatrices;
        float ParamCount = 0;
    };

private:
    yint MaxDeltaMatrices = 0;
    TVector<TIntrusivePtr<TDeviceData>> DeviceArr;
    TVector<int> MatrixOpArr;
    TVector<int> MatrixReadyDeviceCount;
    TVector<TIntrusivePtr<TModelMatrix>> MatrixArr;
    TIntrusivePtr<IMMDeltaHookGen> DeltaHookGen;
    TVector<TIntrusivePtr<IMMDeltaHook>> DeltaHookArr;
    TVector<TIntrusivePtr<TWorkerData>> WorkerArr;
    std::atomic<yint> WorkerCount;
    std::atomic<yint> JobCount;
    yint MatrixCount = 0;
    volatile bool Exit = false;
    volatile float Step = 0;

    bool PerformAllOps(yint workerId);
    ~TCPUMatrixAdd();

public:
    TCPUMatrixAdd(yint deviceCount, yint maxDeltaMatrices, IMMDeltaHookGen *deltaHookGen);
    void AddMatrix(TIntrusivePtr<TModelMatrix> p);
    void LaunchWorkers();
    void StartIteration(float step); // assume no pending ops at this moment
    void Wait();
    void ResetIterCount();
    yint GetDeviceCount() const { return YSize(DeviceArr); }

public:
    void WorkerThread();
};


TIntrusivePtr<TModelMatrix> CreateModelMatrix(TIntrusivePtr<TCPUMatrixAdd> cpuAdd, TIntrusivePtr<TModelMatrixScale> pScale,
    float discrScale, yint xSize, yint ySize,
    EModelMatrixUseRowDisp useRowDisp, EModelMatrixQuant quant, EModelMatrixStaleGradient staleGrad);

}
