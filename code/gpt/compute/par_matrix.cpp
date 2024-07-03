#include "stdafx.h"
#include "par_matrix.h"
#include <gpt/model_params/sse_utils.h>
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
static i8 QBitRecode2bit[256];
static i8 QBitRecode4bit[256];
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
        for (yint a = 0; a < 256; ++a) {
            i8 x = a;
            yint xx = x;
            yint bitVal = ClampVal<yint>((xx + 4 + 9 * 8) / 9, 0, 15); // 4.88512
            QBitRecode4bit[a] = bitVal * 9 + 4 - 9 * 8;
        }
    }
} initBitTable;

static void ConvertToFastMatrixFloat(i8 *dst, const float *src, __m256 mult, int xSize, EModelMatrixQuant quant)
{
    ConvertArray(dst, src, xSize, mult);

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
    } else if (quant == MM_QUANT_4BIT) {
        for (yint x = 0; x < xSize; ++x) {
            dst[x] = QBitRecode4bit[(ui8)dst[x]]; // can be speed up with SSE
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


///////////////////////////////////////////////////////////////////////////////////////////////////
void TModelMatrix::Allocate(yint deviceCount, TIntrusivePtr<TModelMatrixScale> pScale,
    float discrScale, yint xSize, yint ySize,
    EModelMatrixUseRowDisp useRowDisp, EModelMatrixQuant quant, EModelMatrixStaleGradient staleGrad)
{
    DiscrScale = discrScale;
    Matr.Init(xSize, ySize, useRowDisp);
    SumDelta.Init(xSize, ySize);
    FastHost.AllocateHost(xSize, ySize);
    MatrixScale = pScale;
    MatrixScaleIndex = pScale->GetIndex();
    Quantization = quant;
    StaleGradientAllowed = (staleGrad == MM_STALE_GRADIENT);
    DeviceArr.resize(deviceCount);
    for (yint deviceId = 0; deviceId < YSize(DeviceArr); ++deviceId) {
        DeviceArr[deviceId] = new TDeviceData;
        TDeviceData &dev = *DeviceArr[deviceId];
        dev.Delta.AllocateHost(xSize, ySize);
        dev.Delta.ClearHostMem();
        dev.RowScale.AllocateHost(ySize);
        dev.RowScale.ClearHostMem();
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


void TModelMatrix::AddToSumDelta(const TModelMatrixInt8Delta &delta)
{
    if (HasDelta) {
        Add(&SumDelta, delta);
    } else {
        Copy(&SumDelta, delta);
    }
    HasDelta = true;
}


void TModelMatrix::AddDeviceToSumDelta(yint deviceId)
{
    AddToSumDelta(DeviceArr[deviceId]->GetDelta());
}


void TModelMatrix::Convert()
{
    yint xSize = Matr.GetXSize();
    yint ySize = Matr.GetYSize();

    float sko = sqrt(Matr.GetSum2() / (xSize * ySize));
    float discrScale = sko * DiscrScale;
    __m256 mult = _mm256_set1_ps((sko == 0) ? 0 : (1 / discrScale));

    MatrixScale->SetScale(MatrixScaleIndex, discrScale);

    TMemoryBlob fastMem = FastHost.GetHostMem();
    bool hasRowDisp = Matr.HasRowDisp();
    for (yint y = 0; y < ySize; ++y) {
        __m256 rowMult = mult;
        if (hasRowDisp) { // always predicted correctly
            rowMult = _mm256_mul_ps(mult, _mm256_set1_ps(Matr.GetRowScale(y)));
        }
        TFastMatrixFloat *dst = fastMem.GetElementAddress<TFastMatrixFloat>(0, y);
        const float *src = Matr.GetRow(y);
        ConvertToFastMatrixFloat(dst, src, rowMult, xSize, Quantization);
    }
}


void TModelMatrix::AddDelta(const TTrainingStep &step)
{
    Y_ASSERT(HasDelta);
    Y_ASSERT(*OpPointer == OP_ADD_DELTA);
    if (!Matr.AddDelta(SumDelta, ROW_DISP_DECAY, step.Rate, step.GetShrinkMult())) {
        SetOp(OP_NONE);
        return;
    }
    Convert();
    SetOp(OP_NONE);
    HasDelta = false;
}


void TModelMatrix::AddBitDelta(const TTrainingStep &step)
{
    Y_ASSERT(*OpPointer == OP_ADD_BIT_DELTA);
    if (!Matr.AddBitDelta(BitDelta, ROW_DISP_DECAY, step.Rate, step.GetShrinkMult())) {
        SetOp(OP_NONE);
        return;
    }
    Convert();
    SetOp(OP_NONE);
}


void TModelMatrix::ExtractDelta(TModelMatrixBitDelta *pBitDelta, TArray2D<float> *pDeltaTail)
{
    Matr.CompressDelta(SumDelta, pBitDelta, pDeltaTail);
    HasDelta = false;
}


void TModelMatrix::GetFastFloatData(TArray2D<float> *p) const
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

void TModelMatrix::GetData(TArray2D<float> *p)
{
    Matr.GetData(p);
}

void TModelMatrix::GetData(TModelMatrixRowDisp *p)
{
    Matr.GetData(p);
}

void TModelMatrix::SetData(const TArray2D<float> &data)
{
    Matr.SetData(data);
    Convert();
}

void TModelMatrix::SetData(const TModelMatrixRowDisp &data)
{
    Matr.SetData(data);
    Convert();
}

void TModelMatrix::GetDeltaData(TArray2D<float> *p) const
{
    Y_VERIFY(!Matr.HasRowDisp());
    SumDelta.GetAllData(p);
}

void TModelMatrix::GetDeltaData(TModelMatrixRowDisp *p) const
{
    Y_VERIFY(Matr.HasRowDisp());
    TArray2D<float> grad;
    SumDelta.GetAllData(&grad);
    p->SetMatrix(grad);
}


void TModelMatrix::ApplyDelta(const TArray2D<float> &data)
{
    yint xSize = GetXSize();
    yint ySize = GetYSize();
    TVector<i8> arr;
    TVector<float> rowScale;
    ClearPodArray(&arr, xSize * ySize);
    ClearPodArray(&rowScale, ySize);
    TModelMatrixInt8Delta modelDelta(arr.data(), xSize, rowScale.data(), xSize, ySize);
    Compress(&modelDelta, data);
    AddToSumDelta(modelDelta);
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
                // sum delta from this device
                MatrixArr[k]->AddDeviceToSumDelta(deviceId);
                // count if all devices are ready
                if (deviceCount > 1) {
                    if (++MatrixReadyDeviceCount[k] < deviceCount) {
                        continue;
                    }
                    MatrixReadyDeviceCount[k] = 0;
                }
                Y_VERIFY(op == TModelMatrix::OP_NONE);
                if (AddToModel == GRADIENT_ACCUMULATE) {
                    continue;
                }
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


void TCPUMatrixAdd::StartIteration(const TTrainingStep &step, EAddToModel addToModel)
{
    Y_ASSERT(JobCount.load() == 0);
    Step = step;
    AddToModel = addToModel;
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
