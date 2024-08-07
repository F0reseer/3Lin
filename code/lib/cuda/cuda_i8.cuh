#pragma once
#include "cuda_util.cuh"
#include "cuda_graph.cuh"
#include "cuda_mma.cuh"
#include "cuda_matmul.cuh"
#include "cuda_sort.cuh"


namespace NCuda
{
constexpr int MM_TILE_LARGE = 128;


///////////////////////////////////////////////////////////////////////////////////////////////////
template <class TRes, class TStore>
__global__ void MatrixMultI8(TCuda2DPtr<i8> normState8, TCuda2DPtr<i8> matr, int yLargeTiles, TCuda2DPtr<TRes> res, float storeScaleVal, float *storeScalePtr)
{
    CUDA_STATIC_ASSERT(MM_TILE == 64);
    TTileCoord tc;

    // blockIdx.x - hidden dim
    // blockIdx.y - time

    int warpId = threadIdx.y;

    int aStride = normState8.GetStrideInBytes();
    int bStride = matr.GetStrideInBytes();
    i8 *aPtr = normState8[blockIdx.y * MM_TILE];
    i8 *bPtr = matr[blockIdx.x * MM_TILE];

    TRegTile<int> sum[2][2];
    for (int ty = 0; ty < 2; ++ty) {
        for (int tx = 0; tx < 2; ++tx) {
            sum[ty][tx].Clear();
        }
    }

    int warpX = (warpId & 1) * 2;
    int warpY = warpId & 2;

    __shared__ T8SMemI8Tile aFrag[4];
    __shared__ T8SMemI8Tile bFrag[4];
    for (int yTile = 0; yTile < yLargeTiles; ++yTile) {
        __syncthreads();
        for (int k = 0; k < 4; ++k) {
            Copy8Tile(&aFrag[k], warpId, TCuda2DPtr<i8>(aPtr + k * TILE * aStride, aStride, I8_TILE_GROUP_SIZE, TILE));
            Copy8Tile(&bFrag[k], warpId, TCuda2DPtr<i8>(bPtr + k * TILE * bStride, bStride, I8_TILE_GROUP_SIZE, TILE));
        }
        __syncthreads();

        for (int k = 0; k < 8; ++k) {
            TRegTile<i8> a[2];
            TRegTile<i8> b[2];
            a[0] = TMmaRowMajor::FragA(aFrag[warpY + 0], k);
            a[1] = TMmaRowMajor::FragA(aFrag[warpY + 1], k);
            b[0] = TMmaColMajor::FragB(bFrag[warpX + 0], k);
            b[1] = TMmaColMajor::FragB(bFrag[warpX + 1], k);
            for (int ty = 0; ty < 2; ++ty) {
                for (int tx = 0; tx < 2; ++tx) {
                    MMA(&sum[ty][tx], a[ty], b[tx]);
                }
            }
        }
        aPtr += I8_TILE_GROUP_SIZE;
        bPtr += I8_TILE_GROUP_SIZE;
    }
    for (int ty = 0; ty < 2; ++ty) {
        for (int tx = 0; tx < 2; ++tx) {
            int blkX = blockIdx.x * (MM_TILE / TILE) + warpX + tx;
            int blkY = blockIdx.y * (MM_TILE / TILE) + warpY + ty;
            TStore::Store(storeScaleVal, storeScalePtr, tc, sum[ty][tx], res.Fragment(blkX * TILE, blkY * TILE));
        }
    }
}
KERNEL_BLOCK_SIZE(MatrixMultI8, WARP_SIZE, 4);


// XY,ZY->XZ
template <class T1, class T2, class TStoreType, class TRes, class TXSize, class TYSize, class TZSize>
void I8MatMulXYoZYeXZ(TIntrusivePtr<TGraph> c,
    const T1 &aMatr, const T2 &bMatr, TRes *pResMatr,
    TXSize &&xTiles, TYSize &&yLargeTiles, TZSize &&zTiles, TStoreType &&storeFunc)
{
    CudaCall(c, MatrixMultI8<typename TRes::TElem, TStoreType>)
        .Grid(zTiles, xTiles)
        (aMatr, bMatr, yLargeTiles)
        .Write(pResMatr)
        (storeFunc.GetScale(), storeFunc.GetScalePtr());
}


///////////////////////////////////////////////////////////////////////////////////////////////////
template <class TRes, class TStore>
__global__ void MatrixMultI8large(TCuda2DPtr<i8> normState8, float *nsLargeTileScale, TCuda2DPtr<i8> matr, int yLargeTiles, TCuda2DPtr<TRes> res, float storeScaleVal, float *storeScalePtr)
{
    CUDA_STATIC_ASSERT(MM_TILE_LARGE == 128);
    TTileCoord tc;

    // blockIdx.x - hidden dim
    // blockIdx.y - time

    int warpId = threadIdx.y & 3;
    int bBlockId = threadIdx.y / 4;

    int aStride = normState8.GetStrideInBytes();
    int bStride = matr.GetStrideInBytes();
    i8 *aPtr = normState8[blockIdx.y * MM_TILE_LARGE];
    i8 *bPtr = matr[blockIdx.x * MM_TILE_LARGE];

    TRegTile<int> sum[2][2][2];
    for (int aBlockId = 0; aBlockId < 2; ++aBlockId) {
        for (int ty = 0; ty < 2; ++ty) {
            for (int tx = 0; tx < 2; ++tx) {
                sum[aBlockId][ty][tx].Clear();
            }
        }
    }
    float prevTileScale = 0;

    int warpX = (warpId & 1);
    int warpY = warpId >> 1;

    __shared__ T8SMemI8Tile aFrag[2][4];
    __shared__ T8SMemI8Tile bFrag[2][4];
    for (int yTile = 0; yTile < yLargeTiles; ++yTile) {
        __syncthreads();
        if (nsLargeTileScale) {
            float scale = nsLargeTileScale[yTile];
            if (scale == 0) {
                continue;
            }
            float mult = prevTileScale / scale;
            if (mult != 1) {
                for (int aBlockId = 0; aBlockId < 2; ++aBlockId) {
                    for (int ty = 0; ty < 2; ++ty) {
                        for (int tx = 0; tx < 2; ++tx) {
                            sum[aBlockId][ty][tx].Scale(mult);
                        }
                    }
                }
            }
            prevTileScale = scale;
        }
        if (bBlockId == 0) {
            for (int blk = 0; blk < 2; ++blk) {
                for (int k = 0; k < 4; ++k) {
                    Copy8Tile(&aFrag[blk][k], warpId, TCuda2DPtr<i8>(aPtr + (blk * 4 + k) * TILE * aStride, aStride, I8_TILE_GROUP_SIZE, TILE));
                }
            }
        } else {
            for (int blk = 0; blk < 2; ++blk) {
                for (int k = 0; k < 4; ++k) {
                    Copy8Tile(&bFrag[blk][k], warpId, TCuda2DPtr<i8>(bPtr + (blk * 4 + k) * TILE * bStride, bStride, I8_TILE_GROUP_SIZE, TILE));
                }
            }
        }
        __syncthreads();

        for (int k = 0; k < 8; ++k) {
            TRegTile<i8> b[2];
            b[0] = TMmaColMajor::FragB(bFrag[bBlockId][warpX * 2 + 0], k);
            b[1] = TMmaColMajor::FragB(bFrag[bBlockId][warpX * 2 + 1], k);
            for (int aBlockId = 0; aBlockId < 2; ++aBlockId) {
                for (int ty = 0; ty < 2; ++ty) {
                    TRegTile<i8> a;
                    a = TMmaRowMajor::FragA(aFrag[aBlockId][warpY * 2 + ty], k);
                    for (int tx = 0; tx < 2; ++tx) {
                        MMA(&sum[aBlockId][ty][tx], a, b[tx]);
                    }
                }
            }
        }
        aPtr += I8_TILE_GROUP_SIZE;
        bPtr += I8_TILE_GROUP_SIZE;
    }
    float storeScaleMult = storeScaleVal;
    if (nsLargeTileScale) {
        storeScaleMult *= prevTileScale;
    }
    for (int aBlockId = 0; aBlockId < 2; ++aBlockId) {
        for (int ty = 0; ty < 2; ++ty) {
            for (int tx = 0; tx < 2; ++tx) {
                int blkX = blockIdx.x * (MM_TILE_LARGE / TILE) + bBlockId * 4 + warpX * 2 + tx;
                int blkY = blockIdx.y * (MM_TILE_LARGE / TILE) + aBlockId * 4 + warpY * 2 + ty;
                TStore::Store(storeScaleMult, storeScalePtr, tc, sum[aBlockId][ty][tx], res.Fragment(blkX * TILE, blkY * TILE));
            }
        }
    }
}
KERNEL_BLOCK_SIZE(MatrixMultI8large, WARP_SIZE, 8);


// XY,ZY->XZ
template <class T1, class T2, class TStoreType, class TRes, class TXSize, class TYSize, class TZSize>
void I8MatMulXYoZYeXZlarge(TIntrusivePtr<TGraph> c,
    const T1 &aMatr, const T2 &bMatr, TRes *pResMatr,
    TXSize &&xLargeTiles, TYSize &&yLargeTiles, TZSize &&zLargeTiles, TStoreType &&storeFunc)
{
    CudaCall(c, MatrixMultI8large<typename TRes::TElem, TStoreType>)
        .Grid(zLargeTiles, xLargeTiles)
        (aMatr, nullptr, bMatr, yLargeTiles)
        .Write(pResMatr)
        (storeFunc.GetScale(), storeFunc.GetScalePtr());
}

// XY,ZY->XZ
template <class T1, class T2, class TStoreType, class TRes, class TXSize, class TYSize, class TZSize>
void I8MatMulXYoZYeXZlarge(TIntrusivePtr<TGraph> c,
    const T1 &aMatr, const TCudaVector<float> &aLargeTileScale, const T2 &bMatr, TRes *pResMatr,
    TXSize &&xLargeTiles, TYSize &&yLargeTiles, TZSize &&zLargeTiles, TStoreType &&storeFunc)
{
    CudaCall(c, MatrixMultI8large<typename TRes::TElem, TStoreType>)
        .Grid(zLargeTiles, xLargeTiles)
        (aMatr, aLargeTileScale, bMatr, yLargeTiles)
        .Write(pResMatr)
        (storeFunc.GetScale(), storeFunc.GetScalePtr());
}


///////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void TransposeI8Matrix(TCuda2DPtr<i8> src, TCuda2DPtr<i8> dst);

template <class TXSize, class TYSize>
void Transpose(TIntrusivePtr<TGraph> c, const TCuda2DArray<i8> &src, TXSize &&xLargeTiles, TYSize &&yLargeTiles, TCuda2DArray<i8> *pDst)
{
    CudaCall(c, TransposeI8Matrix).Grid(xLargeTiles, yLargeTiles)(src).Write(pDst);
}


__global__ void ShuffleScaleTransposeKernel(TCuda2DPtr<i8> src, TSortNode *sortNode, int rowCount, TCuda2DPtr<i8> dst, float *largeTileScale);

template <class TLen, class TXSize, class TYSize>
void ShuffleScaleTranspose(TIntrusivePtr<TGraph> c,
    const TCuda2DArray<i8> &src, const TCudaVector<TSortNode> &sorted, TLen &&rowCount, TXSize &&xLargeTiles, TYSize &&yLargeTiles,
    TCuda2DArray<i8> *pDst, TCudaVector<float> *pLargeTileScale)
{
    CudaCall(c, ShuffleScaleTransposeKernel).Grid(xLargeTiles, yLargeTiles)(src, sorted, rowCount).Write(pDst, pLargeTileScale);
}

}
