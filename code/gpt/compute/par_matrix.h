#pragma once
#include <lib/cuda/cuda_arrays.h>
#include "par_delta.h"
#include <gpt/train_config/train_step.h>
#include <util/thread.h>


namespace NCuda
{
    class TModelMatrix;
}
///////////////////////////////////////////////////////////////////////////////////////////////////
enum EAddToModel
{
    GRADIENT_ACCUMULATE,
    GRADIENT_APPLY,
};

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
// simulation model params quantization
enum EModelMatrixQuant
{
    MM_QUANT_NONE,
    MM_QUANT_158BIT, // -1 / 0 / 1
    MM_QUANT_2BIT,
    MM_QUANT_4BIT,
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
        TCuda2DArray<i8> Delta;
        TCudaVector<float> RowScale;
        TCudaPOD<int> CudaLaunchFlag;

        TDeviceData() : CudaLaunchFlag(0, 0) {}
        TModelMatrixInt8Delta GetDelta()
        {
            TMemoryBlob blob = Delta.GetHostMem();
            return TModelMatrixInt8Delta(blob.Ptr, blob.Stride, RowScale.GetHostPtr(), Delta.GetXSize(), Delta.GetYSize());
        }
    };
private:
    TVector<TIntrusivePtr<TDeviceData>> DeviceArr;
    float DiscrScale = 0;
    TModelMatrixData Matr;
    TCuda2DArray<TFastMatrixFloat> FastHost;
    TModelMatrixHalfDelta SumDelta;
    bool HasDelta = false;
    TModelMatrixBitDelta BitDelta;
    TIntrusivePtr<TModelMatrixScale> MatrixScale;
    yint MatrixScaleIndex = 0;
    bool StaleGradientAllowed = false;
    EModelMatrixQuant Quantization;
    volatile int *OpPointer = nullptr;

    void Convert();

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
    void AddToSumDelta(const TModelMatrixInt8Delta &delta);
    void AddDeviceToSumDelta(yint deviceId);
    //
    void AddDelta(const TTrainingStep &step);
    void AddBitDelta(const TTrainingStep &step);
    void ExtractDelta(TModelMatrixBitDelta *pBitDelta, TArray2D<float> *pDeltaTail);
    bool HasRowDisp() const { return Matr.HasRowDisp(); }
    bool CanUseStaleGradient() const { return StaleGradientAllowed; }
    //
    void GetFastFloatData(TArray2D<float> *p) const;
    void GetData(TArray2D<float> *p);
    void GetData(TModelMatrixRowDisp *p);
    void SetData(const TArray2D<float> &data);
    void SetData(const TModelMatrixRowDisp &data);
    void GetDeltaData(TArray2D<float> *p) const;
    void GetDeltaData(TModelMatrixRowDisp *p) const;
    void ApplyDelta(const TArray2D<float> &data);
    TCuda2DArray<TFastMatrixFloat> &GetFastHost() { return FastHost; }
    TModelMatrixBitDelta &GetBitDelta() { return BitDelta; }
    TCuda2DArray<i8> &GetDelta(yint deviceId) { return DeviceArr[deviceId]->Delta; }
    TCudaVector<float> &GetRowScale(yint deviceId) { return DeviceArr[deviceId]->RowScale; }
    TCudaPOD<int> GetLaunchFlag(yint deviceId) const { return DeviceArr[deviceId]->CudaLaunchFlag; }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
class TCPUMatrixAdd : public TThrRefBase
{
    enum {
        BASE_WORKER_COUNT = 2,
        PER_GPU_WORKER_COUNT = 2,
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
    TTrainingStep Step;
    EAddToModel AddToModel = GRADIENT_APPLY;

    bool PerformAllOps(yint workerId);
    ~TCPUMatrixAdd();

public:
    TCPUMatrixAdd(yint deviceCount, yint maxDeltaMatrices, IMMDeltaHookGen *deltaHookGen);
    void AddMatrix(TIntrusivePtr<TModelMatrix> p);
    void LaunchWorkers();
    void StartIteration(const TTrainingStep &step, EAddToModel addToModel); // assume no pending ops at this moment
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
