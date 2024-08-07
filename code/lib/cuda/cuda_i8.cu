#include "stdafx.h"
#define KERNEL_UNIT "cuda_i8/"
#include "cuda_i8.cuh"


namespace NCuda
{
__global__ void TransposeI8Matrix(TCuda2DPtr<i8> src, TCuda2DPtr<i8> dst)
{
    CUDA_STATIC_ASSERT(MM_TILE_LARGE == 128);
    int xBlock = blockIdx.x * MM_TILE_LARGE;
    int yBlock = blockIdx.y * MM_TILE_LARGE;

    __shared__ i8 buf[128][128];

    int h = threadIdx.x;
    constexpr int yStep = WARP_SIZE / 8;
    int yOffset = h / 8;
    int xOffset = 16 * (h & 7);
    for (int yBase = 0; yBase < 128; yBase += yStep) {
        int4 *pSrc = (int4 *)&src[yBlock + yBase + yOffset][xBlock + xOffset];
        int4 *pDst = (int4 *)&buf[yBase + yOffset][xOffset];
        *pDst = *pSrc;
    }
    __syncthreads();

    for (int yBase = 0; yBase < 128; yBase += yStep) {
        union {
            int4 column;
            i8 columnBytes[16];
        };
        for (int k = 0; k < 16; ++k) {
            columnBytes[k] = buf[xOffset + k][yBase + yOffset];
        }
        int4 *pDst = (int4 *)&dst[xBlock + yBase + yOffset][yBlock + xOffset];
        *pDst = column;
    }
}


__global__ void ShuffleScaleTransposeKernel(TCuda2DPtr<i8> src, TSortNode *sortNode, int rowCount, TCuda2DPtr<i8> dst, float *largeTileScale)
{
    CUDA_STATIC_ASSERT(MM_TILE_LARGE == 128);
    int h = threadIdx.x;
    int xBlock = blockIdx.x * MM_TILE_LARGE;
    int yBlock = blockIdx.y * MM_TILE_LARGE;

    float maxScale = 0;
    for (int k = h; k < 128; k += WARP_SIZE) {
        int t = yBlock + k;
        if (t < rowCount) {
            maxScale = fmaxf(maxScale, sortNode[t].Score);
        }
    }
    maxScale = WarpMax(maxScale);
    if (maxScale == 0) {
        maxScale = 1;
    }
    // round maxScale to avoid precision loss in I8MatMulXYoZYeXZlarge() due to long chain of sum multiplication by close to 1 numbers
    maxScale = __int_as_float(__float_as_int(maxScale) & 0xfff00000);

    __shared__ i8 buf[128][128];
    constexpr int yStep = WARP_SIZE / 8;
    int yOffset = h / 8;
    int xOffset = 16 * (h & 7);
    for (int yBase = 0; yBase < 128; yBase += yStep) {
        int y = yBase + yOffset;
        int t = yBlock + y;
        int4 *pDst = (int4 *)&buf[y][xOffset];
        if (t < rowCount) {
            int nodeId = sortNode[t].NodeId;
            float mult = sortNode[t].Score / maxScale;
            union {
                int4 row;
                i8 rowBytes[16];
            };
            row = *(int4 *)&src[nodeId][xBlock + xOffset];
            for (int k = 0; k < 16; ++k) {
                rowBytes[k] = CvtToI8(rowBytes[k] * mult);
            }
            *pDst = row;
        } else {
            *pDst = make_int4(0, 0, 0, 0);
        }
    }
    __syncthreads();

    for (int yBase = 0; yBase < 128; yBase += yStep) {
        int y = yBase + yOffset;
        union {
            int4 column;
            i8 columnBytes[16];
        };
        for (int k = 0; k < 16; ++k) {
            int x = xOffset + k;
            columnBytes[k] = buf[x][y];
        }
        int4 *pDst = (int4 *)&dst[xBlock + y][yBlock + xOffset];
        *pDst = column;
    }
    if (h == 0 && xBlock == 0) {
        largeTileScale[blockIdx.y] = maxScale;
    }
}

}
