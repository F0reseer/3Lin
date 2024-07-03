#pragma once

///////////////////////////////////////////////////////////////////////////////////////////////////
//
constexpr float MATRIX_MULT_I8_SCALE = 1.f / 16384;

template <int STATE_DIM, class TDst>
__global__ void ConvertHalfToVecFloat(TCuda2DPtr<half> src, float staticScale, float *pSrcScale, TCuda2DPtr<TDst> dst8, float *scaleArr)
{
    int h = threadIdx.x;
    int t = blockIdx.x;
    // each vector has its own scale
    constexpr int WSZ = STATE_DIM / WARP_SIZE;
    float v[WSZ];
    float sum2 = 0;
    for (int k = 0; k < WSZ; ++k) {
        int d = k * WARP_SIZE + h;
        float val = src[t][d];
        v[k] = val;
        sum2 += val * val;
    }
    sum2 = WarpSum(sum2);
    if (sum2 > 0) {
        float sko = sqrtf(sum2 / STATE_DIM);
        float discrScale = sko * VEC_SCALE;
        for (int k = 0; k < WSZ; ++k) {
            TVecFloat res;
            CvtToVecFloat(&res, v[k] / discrScale);
            int d = k * WARP_SIZE + h;
            dst8[t][d] = res;
        }
        if (scaleArr && h == 0) {
            if (pSrcScale) {
                scaleArr[t] = discrScale * *pSrcScale * staticScale;
            } else {
                scaleArr[t] = discrScale * staticScale;
            }
        }
    } else {
        for (int k = 0; k < WSZ; ++k) {
            int d = k * WARP_SIZE + h;
            dst8[t][d] = 0;
        }
        if (scaleArr && h == 0) {
            scaleArr[t] = 0;
        }
    }
}


template <int STATE_DIM, class TElemType>
__global__ void BackpropNormalizeVecs8(TCuda2DPtr<TElemType> srcNorm8, float *srcScale, TCuda2DPtr<half> grad, TCuda2DPtr<half> dst)
{
    int t = blockIdx.x;
    constexpr int WSZ = STATE_DIM / WARP_SIZE;

    // normalize
    float v[WSZ];
    LoadWarpVec<WSZ>(v, srcNorm8[t]);
    ScaleWarpVec<WSZ>(v, VEC_SCALE * srcScale[t]);
    float vGrad[WSZ];
    LoadWarpVec<WSZ>(vGrad, grad[t]);
    float stateGrad[WSZ];
    StateNormalizeBackpropWarpVec<WSZ>(v, vGrad, stateGrad);
    StoreWarpVec<WSZ>(dst[t], stateGrad);
}


///////////////////////////////////////////////////////////////////////////////////////////////////
struct TStoreI8Scaled
{
    float Scale;

    TStoreI8Scaled(float scale) : Scale(scale) {}
    float GetScale() { return Scale; }
    float *GetScalePtr() { return nullptr; }
    template <class TRes>
    __device__ static void Store(float storeScaleVal, float *storeScalePtr, const TTileCoord &tc, const TRegTile<int> &tile, TCuda2DPtr<TRes> resPtr)
    {
        (void)storeScalePtr;
        tile.StoreScaled(tc, resPtr, storeScaleVal);
    }
};

struct TStoreI8AddScaled
{
    TCudaPOD<float> Scale;
    float ScaleVal;

    TStoreI8AddScaled(const TCudaPOD<float> &scale) : Scale(scale), ScaleVal(1.0f) {}
    TStoreI8AddScaled(const TCudaPOD<float> &scale, float scaleVal) : Scale(scale), ScaleVal(scaleVal) {}
    float GetScale() { return ScaleVal; }
    TCudaPOD<float> GetScalePtr() { return Scale; }
    template <class TRes>
    __device__ static void Store(float storeScaleVal, float *storeScalePtr, const TTileCoord &tc, const TRegTile<int> &tile, TCuda2DPtr<TRes> resPtr)
    {
        tile.StoreAddScaled(tc, resPtr, *storeScalePtr * storeScaleVal);
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
template <int STATE_DIM, class T1, class T2, class TRes, class TStore>
__global__ void MatrixMultI8(TCuda2DPtr<T1> normState8, TCuda2DPtr<T2> matr, TCuda2DPtr<TRes> res, float storeScaleVal, float *storeScalePtr)
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
    for (int d = 0; d < STATE_DIM; d += I8_TILE_GROUP_SIZE) {
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
template <int Y_DIM, class T1, class T2, class TStoreType, class TRes, class TXSize, class TZSize>
void I8MatMulXYoZYeXZ(TIntrusivePtr<TGraph> c,
    const T1 &aMatr, const T2 &bMatr, TRes *pResMatr,
    TXSize &&xTiles, TZSize &&zTiles, TStoreType &&storeFunc)
{
    CudaCall(c, MatrixMultI8<Y_DIM, typename T1::TElem, typename T2::TElem, typename TRes::TElem, TStoreType>)
        .Grid(zTiles, xTiles)
        (aMatr, bMatr)
        .Write(pResMatr)
        (storeFunc.GetScale(), storeFunc.GetScalePtr());
}


///////////////////////////////////////////////////////////////////////////////////////////////////
template <int STATE_DIM, class T1, class T2, class TRes, class TStore>
__global__ void MatrixMultI8large(TCuda2DPtr<T1> normState8, TCuda2DPtr<T2> matr, TCuda2DPtr<TRes> res, float storeScaleVal, float *storeScalePtr)
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

    int warpX = (warpId & 1);
    int warpY = warpId >> 1;

    __shared__ T8SMemI8Tile aFrag[2][4];
    __shared__ T8SMemI8Tile bFrag[2][4];
    for (int d = 0; d < STATE_DIM; d += I8_TILE_GROUP_SIZE) {
        __syncthreads();
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
    for (int aBlockId = 0; aBlockId < 2; ++aBlockId) {
        for (int ty = 0; ty < 2; ++ty) {
            for (int tx = 0; tx < 2; ++tx) {
                int blkX = blockIdx.x * (MM_TILE_LARGE / TILE) + bBlockId * 4 + warpX * 2 + tx;
                int blkY = blockIdx.y * (MM_TILE_LARGE / TILE) + aBlockId * 4 + warpY * 2 + ty;
                TStore::Store(storeScaleVal, storeScalePtr, tc, sum[aBlockId][ty][tx], res.Fragment(blkX * TILE, blkY * TILE));
            }
        }
    }
}
KERNEL_BLOCK_SIZE(MatrixMultI8large, WARP_SIZE, 8);


// XY,ZY->XZ
template <int Y_DIM, class T1, class T2, class TStoreType, class TRes, class TXSize, class TZSize>
void I8MatMulXYoZYeXZlarge(TIntrusivePtr<TGraph> c,
    const T1 &aMatr, const T2 &bMatr, TRes *pResMatr,
    TXSize &&xTiles, TZSize &&zTiles, TStoreType &&storeFunc)
{
    CudaCall(c, MatrixMultI8large<Y_DIM, typename T1::TElem, typename T2::TElem, typename TRes::TElem, TStoreType>)
        .Grid(zTiles, xTiles)
        (aMatr, bMatr)
        .Write(pResMatr)
        (storeFunc.GetScale(), storeFunc.GetScalePtr());
}

