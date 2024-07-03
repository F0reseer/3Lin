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
        yint xSize = pMatrix->GetXSize();
        yint ySize = pMatrix->GetYSize();
        Y_ASSERT((xSize % MM_TILE) == 0);
        yint roundYSize = DivCeil(ySize, MM_TILE) * MM_TILE;
        FastDevice.AllocateCuda(xSize, roundYSize);
    }
}


const int COPY_DELTA_BLOCK = 8;
__global__ void CopyDelta(TCuda2DPtr<float> srcArr, int xSize, int ySize, int *iterCounter, TCuda2DPtr<i8> dstArr, float *dstRowScale, int *launchOpPtr)
{
    int h = threadIdx.x;
    int warpId = threadIdx.y;
    for (int yBase = 0; yBase < ySize; yBase += COPY_DELTA_BLOCK) {
        int y = yBase + warpId;
        if (y >= ySize) {
            break;
        }
        // compute row scale
        float maxVal = 0;
        for (int xBase = 0; xBase < xSize; xBase += WARP_SIZE) {
            int x = xBase + h;
            float val = srcArr[y][x];
            maxVal = fmaxf(maxVal, fabsf(val));
        }
        maxVal = WarpMax(maxVal);
        float scale = (maxVal > 0) ? maxVal / 127 : 0;
        float mult = (maxVal > 0) ? 1 / scale : 0;

        // copy row
        for (int xBase = 0; xBase < xSize; xBase += WARP_SIZE * 16) {
            int x = xBase + h * 16;
            if (x >= xSize) {
                break;
            }
            // read
            int4 *srcPtr = (int4 *)&srcArr[y][x];
            union {
                int4 srcBlock[4];
                float src[16];
            };
            srcBlock[0] = srcPtr[0];
            srcBlock[1] = srcPtr[1];
            srcBlock[2] = srcPtr[2];
            srcBlock[3] = srcPtr[3];

            // convert
            union {
                int4 resBlock;
                i8 res[16];
            };
            for (int xx = 0; xx < 16; ++xx) {
                float val = srcArr[y][x + xx];
                res[xx] = CvtToI8(val * mult);
            }

            // write
            *(int4 *)&dstArr[y][x] = resBlock;
        }

        if (h == 0) {
            dstRowScale[y] = scale;
        }
    }
    __threadfence_system(); // neccessary!
    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        *launchOpPtr = *iterCounter;
    }
    __threadfence_system(); // neccessary?
}
KERNEL_BLOCK_SIZE(CopyDelta, WARP_SIZE, COPY_DELTA_BLOCK);


__global__ void ClearHostMemKernel(int len, float *p)
{
    float4 zero = make_float4(0, 0, 0, 0);
    float4 *dst = (float4 *)p;
    for (int base = 0; base < len / sizeof(dst[0]); base += WARP_SIZE) {
        int x = base + threadIdx.x;
        if (x < len) {
            dst[x] = zero;
        }
    }
    __threadfence_system(); // neccessary?
}


__global__ void LaunchOpKernel(TCuda2DPtr<float> delta, float *rowScale, int *iterCounter, int *launchOpPtr)
{
    (void)delta; // needed for dependency
    (void)rowScale; // needed for dependency
    __threadfence_system(); // neccessary?
    if (threadIdx.x == 0) {
        *launchOpPtr = *iterCounter;
    }
    __threadfence_system(); // neccessary?
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


void TCudaModelMatrix::ClearRowScale(TIntrusivePtr<TGraph> c)
{
    TCudaVector<float> &rs = GetDeltaRowScale();
    int ySize = rs.GetSize();
    CudaCall(c, ClearHostMemKernel)(ySize).Write(&rs);
}


void TCudaModelMatrix::CopyDeltaToHostAndApply(TIntrusivePtr<TGraph> c, TCuda2DArray<float> &delta, TCudaVector<int> &iterCounter)
{
    // copy first rows, delta might have more rows due to size rounding
    TCuda2DArray<i8> &deltaHost = GetDelta();
    TCudaVector<float> &deltaRowScaleHost = GetDeltaRowScale();
    int xSize = deltaHost.GetXSize();
    int ySize = deltaHost.GetYSize();
    TCudaPOD<int> launchFlag = Matrix->GetLaunchFlag(DeviceId);
    CudaCall(c, CopyDelta)(delta, xSize, ySize, iterCounter).Write(&deltaHost, &deltaRowScaleHost, &launchFlag);
}


void TCudaModelMatrix::ApplyHostDelta(TIntrusivePtr<TGraph> c, TCudaVector<int> &iterCounter)
{
    TCuda2DArray<i8> &deltaHost = GetDelta(); // add dependency
    TCudaVector<float> &deltaRowScaleHost = GetDeltaRowScale();
    TCudaPOD<int> launchFlag = Matrix->GetLaunchFlag(DeviceId);
    CudaCall(c, LaunchOpKernel)(deltaHost, deltaRowScaleHost, iterCounter).Write(&launchFlag);
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
