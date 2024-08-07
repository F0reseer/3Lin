#include "stdafx.h"
#define KERNEL_UNIT "cuda_sort/"
#include "cuda_sort.cuh"


namespace NCuda
{

inline __device__ void Order(TSortNode *a, TSortNode *b)
{
    if (a->Score > b->Score) {
        TSortNode tmp = *a;
        *a = *b;
        *b = tmp;
    }
}

inline __device__ void SortPass(TSortNode *nodes, int nodeCount, int sz, int bit, int bitValue)
{
    int thrIdx = threadIdx.x + threadIdx.y * WARP_SIZE;
    for (int i = thrIdx; i < nodeCount - bit; i += WARP_SIZE * FLOAT_SORT_WARP_COUNT) {
        int opp = i + bit;
        if ((i & bit) == bitValue && ((i ^ opp) & ~(sz - 1)) == 0) {
            Order(nodes + i, nodes + opp);
        }
    }
    __syncthreads();
}

__global__ void SortFloatsKernel(float *valArr, int nodeCount, TSortNode *nodes)
{
    int thrIdx = threadIdx.x + threadIdx.y * WARP_SIZE;
    for (int i = thrIdx; i < nodeCount; i += WARP_SIZE * FLOAT_SORT_WARP_COUNT) {
        TSortNode &dst = nodes[i];
        dst.NodeId = i;
        dst.Score = valArr[i];
    }
    __syncthreads();
    // Batcher odd even merge sort (sorting network, to understand better search for picture)
    for (int bit = 1; bit < nodeCount; bit *= 2) {
        int sz = bit * 2; // size of lists to sort on this iteration
        SortPass(nodes, nodeCount, sz, bit, 0);
        for (int sub = bit / 2; sub > 0; sub /= 2) {
            SortPass(nodes, nodeCount, sz, sub, sub);
        }
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void HistSortFloatsKernel(float *valArr, int nodeCount, TSortNode *nodes)
{
    constexpr int N_BITS = 13;
    constexpr int SZ = 1 << N_BITS;
    __shared__ int hist[SZ];

    int thrIdx = threadIdx.x + threadIdx.y * WARP_SIZE;
    // clear histogram
    for (int i = thrIdx; i < SZ; i += WARP_SIZE * FLOAT_SORT_WARP_COUNT) {
        hist[i] = 0;
    }
    // collect histogram
    for (int i = thrIdx; i < nodeCount; i += WARP_SIZE * FLOAT_SORT_WARP_COUNT) {
        CUDA_ASSERT(valArr[i] >= 0 && "implementation for positive floats");
        float val = valArr[i];
        int bin = __float_as_int(val) >> (31 - N_BITS);
        atomicAdd(&hist[bin], 1);
    }
    __syncthreads();
    // sum histogram
    for (int bit = 1; bit < nodeCount; bit *= 2) {
        for (int i = thrIdx; i < SZ; i += WARP_SIZE * FLOAT_SORT_WARP_COUNT) {
            if (i & bit) {
                hist[i] += hist[(i ^ bit) | (bit - 1)];
            }
        }
        __syncthreads();
    }
    // output result
    for (int i = thrIdx; i < nodeCount; i += WARP_SIZE * FLOAT_SORT_WARP_COUNT) {
        float val = valArr[i];
        int bin = __float_as_int(val) >> (31 - N_BITS);
        int ptr = atomicAdd(&hist[bin], -1) - 1;
        TSortNode &dst = nodes[ptr];
        dst.NodeId = i;
        dst.Score = valArr[i];
    }
}
}
