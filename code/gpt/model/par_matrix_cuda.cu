#include "stdafx.h"
#define KERNEL_UNIT "par_matrix_cuda/"
#include "par_matrix_cuda.cuh"
#include <lib/cuda/cuda_graph.cuh>
#include <lib/cuda/cuda_matmul.cuh>


namespace NCuda
{

///////////////////////////////////////////////////////////////////////////////////////////////////
TCudaModelMatrixScale::TCudaModelMatrixScale(TIntrusivePtr<TModelMatrixScale> pScale, TStream &stream) : MatrixScale(pScale)
{
    MatrixScaleDevice.AllocateCuda(pScale->GetSize());
    CopyDeltaLock.AllocateCuda(1);
    CopyDeltaLock.ClearDeviceMem(stream);
}

void TCudaModelMatrixScale::CopyToDevice(TIntrusivePtr<TGraph> c)
{
    c->KernelCopy(&MatrixScaleDevice, MatrixScale->GetMatrixScaleHost());
}



///////////////////////////////////////////////////////////////////////////////////////////////////
//
TCudaModelMatrix::TCudaModelMatrix(yint deviceId, TIntrusivePtr<TCudaModelMatrixScale> pCudaMatrixScale, TIntrusivePtr<TModelMatrix> pMatrix, EModelMatrixMemory mmm)
    : DeviceId(deviceId), Matrix(pMatrix), CudaMatrixScale(pCudaMatrixScale), Mem(mmm)
{
    Y_ASSERT(Matrix->GetMatrixScale() == CudaMatrixScale->GetMatrixScale());
    if (Mem == MM_MEM_DEVICE) {
        yint xSize = pMatrix->GetData().GetXSize();
        yint ySize = pMatrix->GetData().GetYSize();
        Y_ASSERT((xSize % MM_TILE) == 0);
        yint roundYSize = DivCeil(ySize, MM_TILE) * MM_TILE;
        FastDevice.AllocateCuda(xSize, roundYSize);
    }
}


const int COPY_DELTA_BLOCK = 8;
__global__ void CopyDelta(TCuda2DPtr<float> srcArr, int ySize, int *iterCounter, TCuda2DPtr<float> dstArr, int *launchOpPtr, int *copyLock)
{
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        // allow no more then 2 concurrent transfers
        for (;;) {
            if (atomicCAS(copyLock, 0, 1) == 0) {
                break;
            }
            if (atomicCAS(copyLock, 1, 2) == 1) {
                break;
            }
        }
    }
    __syncthreads();
    int thrOffset = threadIdx.y * WARP_SIZE + threadIdx.x;
    int4 *src = (int4 *)&srcArr[0][0];
    int4 *dst = (int4 *)&dstArr[0][0];
    int len = ySize * srcArr.GetStrideInBytes() / sizeof(*src);
    for (int blkOffset = 0; blkOffset < len; blkOffset += WARP_SIZE * COPY_DELTA_BLOCK) {
        int offset = blkOffset + thrOffset;
        if (offset < len) {
            dst[offset] = src[offset];
        }
        // slows down copying but significantly speeds up graph execution (which also utilise pcie bus?)
        __syncthreads();
    }
    __threadfence_system(); // neccessary!
    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        atomicAdd(copyLock, -1);
        *launchOpPtr = *iterCounter;
    }
    __threadfence_system(); // neccessary?
}
KERNEL_BLOCK_SIZE(CopyDelta, WARP_SIZE, COPY_DELTA_BLOCK);


__global__ void LaunchOpKernel(TCuda2DPtr<float> delta, int *iterCounter, int *launchOpPtr)
{
    (void)delta; // needed for dependency
    __threadfence_system(); // neccessary?
    if (threadIdx.x == 0) {
        *launchOpPtr = *iterCounter;
    }
    __threadfence_system(); // neccessary?
}


void TCudaModelMatrix::CopyDeltaToHostAndApply(TIntrusivePtr<TGraph> c, TCuda2DArray<float> &delta, TCudaVector<int> &iterCounter)
{
    // copy first rows, delta might have more rows due to size rounding
    TCuda2DArray<float> &deltaHost = GetDeltaHost();
    Y_ASSERT(deltaHost.GetHostMem().Stride == delta.GetDeviceMem().Stride);
    int ySize = deltaHost.GetYSize();
    TCudaPOD<int> launchFlag = Matrix->GetLaunchFlag(DeviceId);
    TCudaPOD<int> copyLock = CudaMatrixScale->GetCopyLock();
    CudaCall(c, CopyDelta)(delta, ySize, iterCounter).Write(&deltaHost, &launchFlag).AtomicWrite(&copyLock);
}


void TCudaModelMatrix::ApplyHostDelta(TIntrusivePtr<TGraph> c, TCudaVector<int> &iterCounter)
{
    TCuda2DArray<float> &deltaHost = GetDeltaHost(); // add dependency
    TCudaPOD<int> launchFlag = Matrix->GetLaunchFlag(DeviceId);
    CudaCall(c, LaunchOpKernel)(deltaHost, iterCounter).Write(&launchFlag);
}


void TCudaModelMatrix::CopyToDevice(TIntrusivePtr<TGraph> c)
{
    if (Mem == MM_MEM_HOST) {
        return;
    }
    TCuda2DArray<TFastMatrixFloat> &fastHost = Matrix->GetFastHost();
    // copy over PCIE bypassing CPU completely
    c->KernelCopy(&FastDevice, fastHost, fastHost.GetYSize());
    // on Windows under WDDM sometimes hangs due to some obscure buffering
    //c->CopyToDevice(&FastDevice, fastHost);
}


///////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void IncrementIterCounterKernel(int *iterCounter)
{
    if (threadIdx.x == 0) {
        *iterCounter += 1;
    }
}

void IncrementIterCounter(TIntrusivePtr<TGraph> c, TCudaVector<int> &iterCounter)
{
    CudaCall(c, IncrementIterCounterKernel).Write(&iterCounter);
}

}
