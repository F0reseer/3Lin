#pragma once
#include "cuda_util.cuh"
#include "cuda_graph.cuh"

namespace NCuda
{
///////////////////////////////////////////////////////////////////////////////////////////////////
// vector utils

// per thread buffer size
#define WCOUNT (STATE_DIM / WARP_SIZE)

inline __device__ float GetStateLength(int stateDim)
{
    return sqrt(1.f * stateDim);
}


template <int STATE_DIM>
inline __device__ float DotProduct(const float *a, const float *b)
{
    float dp = 0;
    for (int k = 0; k < WCOUNT; ++k) {
        dp += a[k] * b[k];
    }
    return WarpSum(dp);
}


// passing h-th element of the vector, returns h-th element of the vector
// src[] - not normalized source vector, grad[WCOUNT] - array gradient, returns gradient of pre normalization vector
template <int STATE_DIM>
inline __device__ void StateNormalizeBackprop(float *src, float *grad, float *dst)
{
    float sum2 = 0;
    float dp = 0;
    for (int k = 0; k < WCOUNT; ++k) {
        float val = src[k];
        sum2 += val * val;
        dp += val * float(grad[k]);
    }
    sum2 = WarpSum(sum2);
    if (sum2 == 0) {
        for (int k = 0; k < WCOUNT; ++k) {
            dst[k] = 0;
        }
    } else {
        dp = WarpSum(dp);

        float sigma = dp / sum2;
        float scale = sqrtf(1.0f / sum2) * GetStateLength(STATE_DIM);
        for (int k = 0; k < WCOUNT; ++k) {
            dst[k] = scale * (float(grad[k]) - float(src[k]) * sigma);
        }
    }
}


// dst is thread local
template <int STATE_DIM, class TDst>
inline __device__ void LoadZeroVec(TDst *dst)
{
    for (int k = 0; k < WCOUNT; ++k) {
        dst[k] = 0;
    }
}

// src is global, dst is thread local
template <int STATE_DIM, class TDst, class TSrc>
inline __device__ void LoadVec(TDst *dst, TSrc *src)
{
    int h = threadIdx.x;
    for (int w = 0; w < STATE_DIM; w += WARP_SIZE) {
        int d = w + h;
        dst[w / WARP_SIZE] = src[d];
    }
}

// src is global, dst & srcLocal is thread local
template <int STATE_DIM, class TSrc>
inline __device__ void LoadVecAdd(float *dst, float *srcLocal, TSrc *src)
{
    int h = threadIdx.x;
    for (int w = 0; w < STATE_DIM; w += WARP_SIZE) {
        int d = w + h;
        dst[w / WARP_SIZE] = srcLocal[w / WARP_SIZE] + ((float)src[d]);
    }
}

// src & dst are local
template <int STATE_DIM, class TDst, class TSrc>
inline __device__ void AddVec(TDst *dst, TSrc *src)
{
    for (int k = 0; k < WCOUNT; ++k) {
        dst[k] += src[k];
    }
}

// src & dst are local, can scale inplace
template <int STATE_DIM, class TDst, class TSrc, class TScale>
inline __device__ void ScaleVec(TDst *dst, TSrc *src, TScale scale)
{
    for (int k = 0; k < WCOUNT; ++k) {
        dst[k] = TScale(src[k]) * scale;
    }
}

// src & dst are local
template <int STATE_DIM, class TDst, class TSrc, class TScale>
inline __device__ void AddScaledVec(TDst *dst, TSrc *src, TScale scale)
{
    for (int k = 0; k < WCOUNT; ++k) {
        dst[k] += TScale(src[k]) * scale;
    }
}

// local vec
template <int STATE_DIM>
inline __device__ float CalcLinf(float *v)
{
    float res = 0;
    for (int k = 0; k < WCOUNT; ++k) {
        res = fmaxf(res, fabsf(v[k]));
    }
    return WarpMax(res);
}

// local vec
template <int STATE_DIM>
inline __device__ float CalcSum2(float *v)
{
    float sum2 = 0;
    for (int k = 0; k < WCOUNT; ++k) {
        float val = v[k];
        sum2 += val * val;
    }
    return WarpSum(sum2);
}

// src & dst are local, can normalize inplace
template <int STATE_DIM>
inline __device__ void NormalizeVec(float *dst, float *src)
{
    float sum2 = CalcSum2<STATE_DIM>(src);
    float scale = (sum2 == 0) ? 0 : sqrtf(1.0f / sum2) * GetStateLength(STATE_DIM);
    ScaleVec<STATE_DIM>(dst, src, scale);
}

// src is thread local, dst is global
template <int STATE_DIM, class TDst, class TSrc>
inline __device__ void StoreVec(TDst *dst, TSrc *src)
{
    int h = threadIdx.x;
    for (int w = 0; w < STATE_DIM; w += WARP_SIZE) {
        int d = w + h;
        dst[d] = src[w / WARP_SIZE];
    }
}

template <int STATE_DIM, class TDst>
inline __device__ void StoreZeroVec(TDst *dst)
{
    int h = threadIdx.x;
    for (int w = 0; w < STATE_DIM; w += WARP_SIZE) {
        int d = w + h;
        dst[d] = 0;
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// block size for some state vector processing kernels
constexpr int SAMPLE_BLOCK = 8;

template <int STATE_DIM, class TSrc, class TDst>
__global__ void CopyVecs(int len, TCuda2DPtr<TSrc> src, TCuda2DPtr<TDst> dst)
{
    int t = blockIdx.x * SAMPLE_BLOCK + threadIdx.y;
    if (t < len) {
        float v[WCOUNT];
        LoadVec<STATE_DIM>(v, src[t]);
        StoreVec<STATE_DIM>(dst[t], v);
    } else {
        StoreZeroVec<STATE_DIM>(dst[t]);
    }
}
KERNEL_BLOCK_SIZE(CopyVecs, WARP_SIZE, SAMPLE_BLOCK);


template <int STATE_DIM, class T>
__global__ void NormalizeVecs(int len, TCuda2DPtr<T> src, TCuda2DPtr<T> dst)
{
    int t = blockIdx.x * SAMPLE_BLOCK + threadIdx.y;

    // normalize
    if (t < len) {
        float v[WCOUNT];
        float vNorm[WCOUNT];
        LoadVec<STATE_DIM>(v, src[t]);
        NormalizeVec<STATE_DIM>(vNorm, v);
        StoreVec<STATE_DIM>(dst[t], vNorm);
    } else {
        StoreZeroVec<STATE_DIM>(dst[t]);
    }
}
KERNEL_BLOCK_SIZE(NormalizeVecs, WARP_SIZE, SAMPLE_BLOCK);


template <int STATE_DIM, class T>
__global__ void BackpropNormalizeVecs(int len, TCuda2DPtr<T> src, TCuda2DPtr<half> grad, TCuda2DPtr<half> dst)
{
    int t = blockIdx.x * SAMPLE_BLOCK + threadIdx.y;

    // normalize
    if (t < len) {
        float v[WCOUNT];
        LoadVec<STATE_DIM>(v, src[t]);
        float vGrad[WCOUNT];
        LoadVec<STATE_DIM>(vGrad, grad[t]);
        float stateGrad[WCOUNT];
        StateNormalizeBackprop<STATE_DIM>(v, vGrad, stateGrad);
        StoreVec<STATE_DIM>(dst[t], stateGrad);
    } else {
        StoreZeroVec<STATE_DIM>(dst[t]);
    }
}
KERNEL_BLOCK_SIZE(BackpropNormalizeVecs, WARP_SIZE, SAMPLE_BLOCK);



template <int STATE_DIM, class T>
__global__ void SumVecsKernel1(int len, TCuda2DPtr<T> src, TCuda2DPtr<T> dst)
{
    int t = blockIdx.x * SAMPLE_BLOCK + threadIdx.y;
    if (t < len) {
        float v[WCOUNT];
        LoadVec<STATE_DIM>(v, src[t]);
        LoadVecAdd<STATE_DIM>(v, v, dst[t]);
        StoreVec<STATE_DIM>(dst[t], v);
    }
}
KERNEL_BLOCK_SIZE(SumVecsKernel1, WARP_SIZE, SAMPLE_BLOCK);

template <int STATE_DIM, class T>
__global__ void SumVecsKernel2(int len, TCuda2DPtr<T> src1, TCuda2DPtr<T> src2, TCuda2DPtr<T> dst)
{
    int t = blockIdx.x * SAMPLE_BLOCK + threadIdx.y;
    if (t < len) {
        float v[WCOUNT];
        LoadVec<STATE_DIM>(v, src1[t]);
        LoadVecAdd<STATE_DIM>(v, v, src2[t]);
        LoadVecAdd<STATE_DIM>(v, v, dst[t]);
        StoreVec<STATE_DIM>(dst[t], v);
    }
}
KERNEL_BLOCK_SIZE(SumVecsKernel2, WARP_SIZE, SAMPLE_BLOCK);


///////////////////////////////////////////////////////////////////////////////////////////////////
// debug kernels
//template <int STATE_DIM>
//__global__ void TestNan(int stepId, int id, int len, TCuda2DPtr<float> vec)
//{
//    int t = blockIdx.x * SAMPLE_BLOCK + threadIdx.y;
//    if (t < len) {
//        float v[WCOUNT];
//        LoadVec<STATE_DIM>(v, vec[t]);
//        for (int k = 0; k < WCOUNT; ++k) {
//            if (isnan(v[k]) || !isfinite(v[k])) {
//                printf("TestNan(%g / %g), t = %g, %g\n", stepId * 1., id * 1., t * 1., v[k]);
//                return;
//            }
//        }
//    }
//}
//KERNEL_BLOCK_SIZE(TestNan, WARP_SIZE, SAMPLE_BLOCK);
//
//
//template <int STATE_DIM>
//__global__ void TestNanHalf(int stepId, int id, int len, TCuda2DPtr<half> vec)
//{
//    int t = blockIdx.x * SAMPLE_BLOCK + threadIdx.y;
//    if (t < len) {
//        float v[WCOUNT];
//        LoadVec<STATE_DIM>(v, vec[t]);
//        for (int k = 0; k < WCOUNT; ++k) {
//            if (isnan(v[k]) || !isfinite(v[k])) {
//                printf("TestNanHalf(%g / %g), t = %g, %g\n", stepId * 1., id * 1., t * 1., v[k]);
//                return;
//            }
//        }
//    }
//}
//KERNEL_BLOCK_SIZE(TestNanHalf, WARP_SIZE, SAMPLE_BLOCK);

template <int STATE_DIM, class T>
__global__ void VecsCheckSum(int len, TCuda2DPtr<T> vecs)
{
    int chkSum = 0;
    for (int base = 0; base < len; base += SAMPLE_BLOCK) {
        int t = base + threadIdx.y;
        if (t < len) {
            float v[WCOUNT];
            LoadVec<STATE_DIM>(v, vecs[t]);
            for (int k = 0; k < WCOUNT; ++k) {
                chkSum += __float_as_int(v[k]);
            }
        }
    }
    __shared__ int blkSum[SAMPLE_BLOCK];
    chkSum = WarpIntSum(chkSum);
    if (threadIdx.x == 0) {
        blkSum[threadIdx.y] = chkSum;
    }
    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        int res = 0;
        for (int k = 0; k < SAMPLE_BLOCK; ++k) {
            res += blkSum[k];
        }
        printf("vecs %p, chksum %d\n", &vecs[0][0], res);
    }
}
KERNEL_BLOCK_SIZE(VecsCheckSum, WARP_SIZE, SAMPLE_BLOCK);


template <class T>
__global__ void PrintValue(T *p)
{
    if (threadIdx.x == 0) {
        printf("Value = %g\n", float(*p));
    }
}

template <int STATE_DIM, class T>
__global__ void PrintVec(int t, TCuda2DPtr<T> vecs)
{
    float v[WCOUNT];
    LoadVec<STATE_DIM>(v, vecs[t]);
    for (int k = 0; k < WCOUNT; ++k) {
        printf("gpu vec[%g] = %g\n", k * WARP_SIZE + threadIdx.x + 0., v[k]);
    }
}

}
