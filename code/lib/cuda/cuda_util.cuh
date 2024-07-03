#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "cuda_arrays.h"


namespace NCuda
{
const int WARP_SIZE = 32;


///////////////////////////////////////////////////////////////////////////////////////////////////
namespace staticassert
{
    template <bool x> struct CheckStruct;
    template <> struct CheckStruct<true> { int X; };// enum { value = 1 };
    template<int x> struct test {};
};

#define CUDA_STATIC_ASSERT( B )  typedef staticassert::test<sizeof(staticassert::CheckStruct< (bool)(B) >) > static_assert_chk_ ## __LINE__


///////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ i8 CvtToI8(float x)
{
    int32_t res;
    asm volatile("cvt.rni.sat.s8.f32 %0, %1;" : "=r"(res) : "f"(x));
    return res;
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// warp sum
// https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
inline __device__ float WarpSum(float val)
{
    Y_ASSERT(WARP_SIZE == 32);
    float sum = val;
    sum += __shfl_xor_sync(0xffffffff, sum, 16);
    sum += __shfl_xor_sync(0xffffffff, sum, 8);
    sum += __shfl_xor_sync(0xffffffff, sum, 4);
    sum += __shfl_xor_sync(0xffffffff, sum, 2);
    sum += __shfl_xor_sync(0xffffffff, sum, 1);
    return sum;
}

inline __device__ int WarpIntSum(int val)
{
    Y_ASSERT(WARP_SIZE == 32);
    int sum = val;
    sum += __shfl_xor_sync(0xffffffff, sum, 16);
    sum += __shfl_xor_sync(0xffffffff, sum, 8);
    sum += __shfl_xor_sync(0xffffffff, sum, 4);
    sum += __shfl_xor_sync(0xffffffff, sum, 2);
    sum += __shfl_xor_sync(0xffffffff, sum, 1);
    return sum;
}

inline __device__ float HalfWarpSum(float val)
{
    Y_ASSERT(WARP_SIZE == 32);
    float sum = val;
    sum += __shfl_xor_sync(0xffffffff, sum, 8);
    sum += __shfl_xor_sync(0xffffffff, sum, 4);
    sum += __shfl_xor_sync(0xffffffff, sum, 2);
    sum += __shfl_xor_sync(0xffffffff, sum, 1);
    return sum;
}

inline __device__ float WarpMax(float val)
{
    Y_ASSERT(WARP_SIZE == 32);
    float res = val;
    res = fmaxf(res, __shfl_xor_sync(0xffffffff, res, 16));
    res = fmaxf(res, __shfl_xor_sync(0xffffffff, res, 8));
    res = fmaxf(res, __shfl_xor_sync(0xffffffff, res, 4));
    res = fmaxf(res, __shfl_xor_sync(0xffffffff, res, 2));
    res = fmaxf(res, __shfl_xor_sync(0xffffffff, res, 1));
    return res;
}

inline __device__ float HalfWarpMax(float val)
{
    Y_ASSERT(WARP_SIZE == 32);
    float res = val;
    res = fmaxf(res, __shfl_xor_sync(0xffffffff, res, 8));
    res = fmaxf(res, __shfl_xor_sync(0xffffffff, res, 4));
    res = fmaxf(res, __shfl_xor_sync(0xffffffff, res, 2));
    res = fmaxf(res, __shfl_xor_sync(0xffffffff, res, 1));
    return res;
}

inline __device__ int WarpMinInt(int val)
{
    Y_ASSERT(WARP_SIZE == 32);
    int res = val;
    res = min(res, __shfl_xor_sync(0xffffffff, res, 16));
    res = min(res, __shfl_xor_sync(0xffffffff, res, 8));
    res = min(res, __shfl_xor_sync(0xffffffff, res, 4));
    res = min(res, __shfl_xor_sync(0xffffffff, res, 2));
    res = min(res, __shfl_xor_sync(0xffffffff, res, 1));
    return res;
}

inline __device__ int WarpMaxInt(int val)
{
    Y_ASSERT(WARP_SIZE == 32);
    int res = val;
    res = max(res, __shfl_xor_sync(0xffffffff, res, 16));
    res = max(res, __shfl_xor_sync(0xffffffff, res, 8));
    res = max(res, __shfl_xor_sync(0xffffffff, res, 4));
    res = max(res, __shfl_xor_sync(0xffffffff, res, 2));
    res = max(res, __shfl_xor_sync(0xffffffff, res, 1));
    return res;
}


inline __device__ void atomicAddExact(float *pDst, float val)
{
    int *p = (int*)pDst;
    for (;;) {
        int assumed = *p;// assumed = old;
        if (atomicCAS(p, assumed, __float_as_int(val + __int_as_float(assumed))) == assumed) {
            return;
        }
    }
}

}
