#pragma once
#include "cuda_util.cuh"
#include "cuda_graph.cuh"

namespace NCuda
{

inline __device__ float GetStateLength(int stateDim)
{
    return sqrt(1.f * stateDim);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// vector utils
// one vector per warp (small dim vectors)
template <int WSZ, class TSrc>
__device__ void LoadWarpVec(float *res, const TSrc *src)
{
    int h = threadIdx.x;
    for (int k = 0; k < WSZ; ++k) {
        int d = k * WARP_SIZE + h;
        res[k] = float(src[d]);
    }
}

template <int WSZ>
__device__ void LoadZeroWarpVec(float *res)
{
    for (int k = 0; k < WSZ; ++k) {
        res[k] = 0;
    }
}

// src & dst are local, can scale inplace
template <int WSZ>
inline __device__ void ScaleWarpVec(float *dst, float scale)
{
    for (int k = 0; k < WSZ; ++k) {
        dst[k] *= scale;
    }
}

// src[WSZ] - not normalized source vector, grad[WSZ] - array gradient, returns gradient of pre normalization vector
template <int WSZ>
inline __device__ void StateNormalizeBackpropWarpVec(float *src, float *grad, float *dst)
{
    constexpr int STATE_DIM = WSZ * WARP_SIZE;
    float sum2 = 0;
    float dp = 0;
    for (int k = 0; k < WSZ; ++k) {
        float val = src[k];
        sum2 += val * val;
        dp += val * float(grad[k]);
    }
    sum2 = WarpSum(sum2);
    if (sum2 == 0) {
        for (int k = 0; k < WSZ; ++k) {
            dst[k] = 0;
        }
    } else {
        dp = WarpSum(dp);

        float sigma = dp / sum2;
        float scale = sqrtf(1.0f / sum2) * GetStateLength(STATE_DIM);
        for (int k = 0; k < WSZ; ++k) {
            dst[k] = scale * (float(grad[k]) - float(src[k]) * sigma);
        }
    }
}


template <int WSZ, class TDst>
__device__ void StoreWarpVec(TDst *dst, float *src)
{
    int h = threadIdx.x;
    for (int k = 0; k < WSZ; ++k) {
        int d = k * WARP_SIZE + h;
        dst[d] = TDst(src[k]);
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// vectors are processed one vector per block
constexpr int VEC_BLOCK = 8;

// per thread buffer size
#define WCOUNT (STATE_DIM / WARP_SIZE / VEC_BLOCK)


inline __device__ float VecBlockSum(float x)
{
    __shared__ float val[VEC_BLOCK];
    __syncthreads();
    int h = threadIdx.x;
    int warpId = threadIdx.y;
    float sum = WarpSum(x);
    if (h == 0) {
        val[warpId] = sum;
    }
    __syncthreads();
    CUDA_ASSERT(VEC_BLOCK == 8);
    sum = val[h & 7];
    sum += __shfl_xor_sync(0xffffffff, sum, 4);
    sum += __shfl_xor_sync(0xffffffff, sum, 2);
    sum += __shfl_xor_sync(0xffffffff, sum, 1);
    return sum;
}


inline __device__ int VecBlockIntSum(int x)
{
    __shared__ int val[VEC_BLOCK];
    __syncthreads();
    int h = threadIdx.x;
    int warpId = threadIdx.y;
    int sum = WarpIntSum(x);
    if (h == 0) {
        val[warpId] = sum;
    }
    __syncthreads();
    CUDA_ASSERT(VEC_BLOCK == 8);
    sum = val[h & 7];
    sum += __shfl_xor_sync(0xffffffff, sum, 4);
    sum += __shfl_xor_sync(0xffffffff, sum, 2);
    sum += __shfl_xor_sync(0xffffffff, sum, 1);
    return sum;
}


inline __device__ float VecBlockMax(float x)
{
    __shared__ float val[VEC_BLOCK];
    __syncthreads();
    int h = threadIdx.x;
    int warpId = threadIdx.y;
    float res = WarpMax(x);
    if (h == 0) {
        val[warpId] = res;
    }
    __syncthreads();
    CUDA_ASSERT(VEC_BLOCK == 8);
    res = val[h & 7];
    res = fmaxf(res, __shfl_xor_sync(0xffffffff, res, 4));
    res = fmaxf(res, __shfl_xor_sync(0xffffffff, res, 2));
    res = fmaxf(res, __shfl_xor_sync(0xffffffff, res, 1));
    return res;
}


inline __device__ bool IsMainVecBlockThread()
{
    return threadIdx.x == 0 && threadIdx.y == 0;
}


///////////////////////////////////////////////////////////////////////////////////////////////////

template <int STATE_DIM>
inline __device__ float DotProduct(const float *a, const float *b)
{
    float dp = 0;
    for (int k = 0; k < WCOUNT; ++k) {
        dp += a[k] * b[k];
    }
    return VecBlockSum(dp);
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
    sum2 = VecBlockSum(sum2);
    if (sum2 == 0) {
        for (int k = 0; k < WCOUNT; ++k) {
            dst[k] = 0;
        }
    } else {
        dp = VecBlockSum(dp);

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
    int hh = threadIdx.y;
    for (int k = 0; k < WCOUNT; ++k) {
        int d = k * WARP_SIZE * VEC_BLOCK + hh * WARP_SIZE + h;
        dst[k] = src[d];
    }
}

// src is global, dst & srcLocal is thread local
template <int STATE_DIM, class TSrc>
inline __device__ void LoadVecAdd(float *dst, float *srcLocal, TSrc *src)
{
    int h = threadIdx.x;
    int hh = threadIdx.y;
    for (int k = 0; k < WCOUNT; ++k) {
        int d = k * WARP_SIZE * VEC_BLOCK + hh * WARP_SIZE + h;
        dst[k] = srcLocal[k] + ((float)src[d]);
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
    return VecBlockMax(res);
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
    return VecBlockSum(sum2);
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
    int hh = threadIdx.y;
    for (int k = 0; k < WCOUNT; ++k) {
        int d = k * WARP_SIZE * VEC_BLOCK + hh * WARP_SIZE + h;
        dst[d] = src[k];
    }
}

// src is thread local, dst is global
template <int STATE_DIM, class TSrc>
inline __device__ void StoreVecInt8(i8 *dst, TSrc *src)
{
    int h = threadIdx.x;
    int hh = threadIdx.y;
    for (int k = 0; k < WCOUNT; ++k) {
        int d = k * WARP_SIZE * VEC_BLOCK + hh * WARP_SIZE + h;
        dst[d] = CvtToI8(src[k]);
    }
}

template <int STATE_DIM, class TDst>
inline __device__ void StoreZeroVec(TDst *dst)
{
    int h = threadIdx.x;
    int hh = threadIdx.y;
    for (int k = 0; k < WCOUNT; ++k) {
        int d = k * WARP_SIZE * VEC_BLOCK + hh * WARP_SIZE + h;
        dst[d] = 0;
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
template <int STATE_DIM, class TSrc, class TDst>
__global__ void CopyVecs(TCuda2DPtr<TSrc> src, TCuda2DPtr<TDst> dst)
{
    int t = blockIdx.x;
    float v[WCOUNT];
    LoadVec<STATE_DIM>(v, src[t]);
    StoreVec<STATE_DIM>(dst[t], v);
}
KERNEL_BLOCK_SIZE(CopyVecs, WARP_SIZE, VEC_BLOCK);


template <int STATE_DIM, class T>
__global__ void SumVecsKernel1(TCuda2DPtr<T> src, TCuda2DPtr<T> dst)
{
    int t = blockIdx.x;
    float v[WCOUNT];
    LoadVec<STATE_DIM>(v, src[t]);
    LoadVecAdd<STATE_DIM>(v, v, dst[t]);
    StoreVec<STATE_DIM>(dst[t], v);
}
KERNEL_BLOCK_SIZE(SumVecsKernel1, WARP_SIZE, VEC_BLOCK);


template <int STATE_DIM, class T>
__global__ void SumVecsKernel2(TCuda2DPtr<T> src1, TCuda2DPtr<T> src2, TCuda2DPtr<T> dst)
{
    int t = blockIdx.x;
    float v[WCOUNT];
    LoadVec<STATE_DIM>(v, src1[t]);
    LoadVecAdd<STATE_DIM>(v, v, src2[t]);
    LoadVecAdd<STATE_DIM>(v, v, dst[t]);
    StoreVec<STATE_DIM>(dst[t], v);
}
KERNEL_BLOCK_SIZE(SumVecsKernel2, WARP_SIZE, VEC_BLOCK);


///////////////////////////////////////////////////////////////////////////////////////////////////
// debug kernels
template <int STATE_DIM>
__global__ void TestNan(int stepId, int id, TCuda2DPtr<float> vec)
{
    int h = threadIdx.x;
    int t = blockIdx.x;
    for (int k = 0; k < STATE_DIM / WARP_SIZE; ++k) {
        int d = k * WARP_SIZE + h;
        float val = vec[t][d];
        if (isnan(val) || !isfinite(val)) {
            printf("TestNanHalf(%g / %g), t = %g, %g\n", stepId * 1., id * 1., t * 1., val);
            return;
        }
    }
}


template <int STATE_DIM>
__global__ void TestNanHalf(int stepId, int id, TCuda2DPtr<half> vec)
{
    int h = threadIdx.x;
    int t = blockIdx.x;
    for (int k = 0; k < STATE_DIM / WARP_SIZE; ++k) {
        int d = k * WARP_SIZE + h;
        float val = vec[t][d];
        if (isnan(val) || !isfinite(val)) {
            printf("TestNanHalf(%g / %g), t = %g, %g\n", stepId * 1., id * 1., t * 1., val);
            return;
        }
    }
}


template <int STATE_DIM, class T>
__global__ void VecsCheckSum(int len, TCuda2DPtr<T> vecs)
{
    int h = threadIdx.x;
    int chkSum = 0;
    for (int t = 0; t < len; ++t) {
        for (int k = 0; k < STATE_DIM / WARP_SIZE; ++k) {
            int d = k * WARP_SIZE + threadIdx.x;
            float val = vecs[t][d];
            chkSum += __float_as_int(val);
        }
    }
    chkSum = WarpIntSum(chkSum);
    if (h == 0) {
        printf("vecs %p, chksum %d\n", &vecs[0][0], chkSum);
    }
}


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
    int h = threadIdx.x;
    for (int k = 0; k < STATE_DIM / WARP_SIZE; ++k) {
        int d = k * WARP_SIZE + h;
        float val = vecs[t][d];
        printf("gpu vec[%g] = %g\n", d * 1., val);
    }
}

}
