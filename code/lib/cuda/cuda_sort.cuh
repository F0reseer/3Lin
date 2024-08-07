#pragma once
#include "cuda_util.cuh"
#include "cuda_graph.cuh"


namespace NCuda
{
struct TSortNode
{
    int NodeId;
    float Score;
};

constexpr int FLOAT_SORT_WARP_COUNT = 16;


///////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void SortFloatsKernel(float *valArr, int nodeCount, TSortNode *nodes);
KERNEL_BLOCK_SIZE(SortFloatsKernel, WARP_SIZE, FLOAT_SORT_WARP_COUNT);

template <class TXSize>
void SortFloats(TIntrusivePtr<TGraph> c, TCudaVector<float> &valArr, TXSize &&len, TCudaVector<TSortNode> *pDst)
{
    CudaCall(c, SortFloatsKernel)(valArr, len).Write(pDst);
}


///////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void HistSortFloatsKernel(float *valArr, int nodeCount, TSortNode *nodes);
KERNEL_BLOCK_SIZE(HistSortFloatsKernel, WARP_SIZE, FLOAT_SORT_WARP_COUNT);

template <class TXSize>
void SortFloatsApprox(TIntrusivePtr<TGraph> c, TCudaVector<float> &valArr, TXSize &&len, TCudaVector<TSortNode> *pDst)
{
    CudaCall(c, HistSortFloatsKernel)(valArr, len).Write(pDst);
}
}
