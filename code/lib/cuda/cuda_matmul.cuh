#pragma once
#include "cuda_util.cuh"
#include "cuda_graph.cuh"
#include "cuda_mma.cuh"


namespace NCuda
{
constexpr int MM_TILE = 64;
constexpr int MM_TILE_LARGE = 128;


// performs res = a @ b (einsum YX,XZ -> YZ)
template <class ARotate, class BRotate, class ResRotate, class T1, class T2, class TRes, class TStore>
__global__ void MatMulKernel(
    TCuda2DPtr<T1> aData, TCuda2DPtr<T2> bData,
    TCuda2DPtr<TRes> resData, float storeScaleVal, float *storeScalePtr,
    int dotLen
)
{
    int warpId = threadIdx.y;

    TTileCoord tc;
    int aStride = aData.GetStrideInBytes();
    int aXStep = ARotate::GetXStep(aStride, (T1*)0);
    int aYStep = ARotate::GetYStep(aStride, (T1 *)0);

    int bStride = bData.GetStrideInBytes();
    int bXStep = BRotate::GetXStep(bStride, (T2 *)0);
    int bYStep = BRotate::GetYStep(bStride, (T2 *)0);

    int resStride = resData.GetStrideInBytes();
    int resXStep = ResRotate::GetXStep(resStride, (TRes*)0);
    int resYStep = ResRotate::GetYStep(resStride, (TRes *)0);

    ui8 *aPtr = aData.GetRawData() + (blockIdx.y * (MM_TILE / TILE)) * aYStep;
    ui8 *bPtr = bData.GetRawData() + (blockIdx.x * (MM_TILE / TILE)) * bXStep;

    TRegTile<float> sum[2][2]; // do not loose precision and use float anyway
    for (int ty = 0; ty < 2; ++ty) {
        for (int tx = 0; tx < 2; ++tx) {
            sum[ty][tx].Clear();
        }
    }

    int warpX = (warpId & 1) * 2;
    int warpY = warpId & 2;

    constexpr int STRIPE_TILES = 4;
    __shared__ T4x4SMemHalfTile aFrag;
    __shared__ T4x4SMemHalfTile bFrag;
    for (int t = 0; t < dotLen; ++t) {
        __syncthreads();
        Copy4x4Tile(&aFrag, warpId, TCuda2DPtr<T1>(aPtr, aStride, MM_TILE, MM_TILE));
        Copy4x4Tile(&bFrag, warpId, TCuda2DPtr<T2>(bPtr, bStride, MM_TILE, MM_TILE));
        __syncthreads();

        for (int k = 0; k < STRIPE_TILES; ++k) {
            TRegTile<half> a[2];
            TRegTile<half> b[2];
            a[0] = ARotate::Rot::FragA(aFrag, k, warpY);
            a[1] = ARotate::Rot::FragA(aFrag, k, warpY + 1);
            b[0] = BRotate::Rot::FragB(bFrag, warpX, k);
            b[1] = BRotate::Rot::FragB(bFrag, warpX + 1, k);
            for (int ty = 0; ty < 2; ++ty) {
                for (int tx = 0; tx < 2; ++tx) {
                    MMA(&sum[ty][tx], a[ty], b[tx]);
                }
            }
        }
        aPtr += aXStep * STRIPE_TILES;
        bPtr += bYStep * STRIPE_TILES;
    }

    for (int ty = 0; ty < 2; ++ty) {
        for (int tx = 0; tx < 2; ++tx) {
            int blkX = blockIdx.x * (MM_TILE / TILE) + warpX + tx;
            int blkY = blockIdx.y * (MM_TILE / TILE) + warpY + ty;
            ui8 *resPtr = resData.GetRawData() + blkX * resXStep + blkY * resYStep;
            TStore::Store(storeScaleVal, storeScalePtr, tc, &sum[ty][tx], TCuda2DPtr<TRes>(resPtr, resStride, TILE, TILE), ResRotate::Rot::StoreRot());
        }
    }
}
constexpr int MM_KERNEL_WARPS = 4;


// store result or add result
struct TStore
{
    float GetScale() { return 1.0f; }
    float *GetScalePtr() { return nullptr; }
    template <class TTile, class T, class TResRotate>
    __device__ static void Store(float storeScaleVal, float *storeScalePtr, const TTileCoord &tc, TTile *tile, TCuda2DPtr<T> resPtr, TResRotate resRotate)
    {
        (void)storeScaleVal;
        (void)storeScalePtr;
        tile->Store(tc, resPtr, resRotate);
    }
};

struct TStoreAdd
{
    float GetScale() { return 1.0f; }
    float *GetScalePtr() { return nullptr; }
    template <class TTile, class T, class TResRotate>
    __device__ static void Store(float storeScaleVal, float *storeScalePtr, const TTileCoord &tc, TTile *tile, TCuda2DPtr<T> resPtr, TResRotate resRotate)
    {
        (void)storeScaleVal;
        (void)storeScalePtr;
        tile->StoreAdd(tc, resPtr, resRotate);
    }
};

struct TStoreScaled
{
    TCudaPOD<float> Scale;

    TStoreScaled(const TCudaPOD<float> &scale) : Scale(scale) {}
    float GetScale() { return 1.0f; }
    TCudaPOD<float> GetScalePtr() { return Scale; }
    template <class TTile, class T, class TResRotate>
    __device__ static void Store(float storeScaleVal, float *storeScalePtr, const TTileCoord &tc, TTile *tile, TCuda2DPtr<T> resPtr, TResRotate resRotate)
    {
        (void)storeScaleVal;
        tile->Scale(*storeScalePtr);
        tile->Store(tc, resPtr, resRotate);
    }
};

struct TStoreAddScaled
{
    TCudaPOD<float> Scale;
    float ScaleVal;

    TStoreAddScaled(const TCudaPOD<float> &scale) : Scale(scale), ScaleVal(1.0f) {}
    TStoreAddScaled(const TCudaPOD<float> &scale, float scaleVal) : Scale(scale), ScaleVal(scaleVal) {}
    float GetScale() { return ScaleVal; }
    TCudaPOD<float> GetScalePtr() { return Scale; }
    template <class TTile, class T, class TResRotate>
    __device__ static void Store(float storeScaleVal, float *storeScalePtr, const TTileCoord &tc, TTile *tile, TCuda2DPtr<T> resPtr, TResRotate resRotate)
    {
        tile->Scale(*storeScalePtr * storeScaleVal);
        tile->StoreAdd(tc, resPtr, resRotate);
    }
};


// 
struct TMatMulDirect
{
    template <class T>
    static __device__ int GetXStep(int stride, T *p) { (void)stride; (void)p;  return TILE * sizeof(T); }
    template <class T>
    static __device__ int GetYStep(int stride, T *p) { (void)p; return TILE * stride; }
    typedef TMmaRowMajor Rot;
};

struct TMatMulTranspose
{
    template <class T>
    static __device__ int GetXStep(int stride, T *p) { (void)p; return TILE * stride; }
    template <class T>
    static __device__ int GetYStep(int stride, T *p) { (void)stride; (void)p; return TILE * sizeof(T); }
    typedef TMmaColMajor Rot;
};


// XY,ZY->XZ
template <class T1, class T2, class TStoreType, class TRes, class TXSize, class TYSize, class TZSize>
void MatMulXYoZYeXZ(TIntrusivePtr<TGraph> c,
    const T1 &aMatr, const T2 &bMatr, TRes *pResMatr,
    TXSize &&xTiles, TYSize &&yTiles, TZSize &&zTiles, TStoreType &&storeFunc)
{
    typedef TMatMulDirect ARot;
    typedef TMatMulTranspose BRot;
    typedef TMatMulDirect ResRot;
    CudaCall(c, MatMulKernel<ARot, BRot, ResRot, typename T1::TElem, typename T2::TElem, typename TRes::TElem, TStoreType>)
        .Grid(zTiles, xTiles).Block(WARP_SIZE, MM_KERNEL_WARPS)
        (aMatr, bMatr)
        .Write(pResMatr)(storeFunc.GetScale(), storeFunc.GetScalePtr(), yTiles);
}


// XY,YZ->XZ
template <class T1, class T2, class TStoreType, class TRes, class TXSize, class TYSize, class TZSize>
void MatMulXYoYZeXZ(TIntrusivePtr<TGraph> c,
    const T1 &aMatr, const T2 &bMatr, TRes *pResMatr,
    TXSize &&xTiles, TYSize &&yTiles, TZSize &&zTiles, TStoreType &&storeFunc)
{
    typedef TMatMulDirect ARot;
    typedef TMatMulDirect BRot;
    typedef TMatMulDirect ResRot;
    CudaCall(c, MatMulKernel<ARot, BRot, ResRot, typename T1::TElem, typename T2::TElem, typename TRes::TElem, TStoreType>)
        .Grid(zTiles, xTiles).Block(WARP_SIZE, MM_KERNEL_WARPS)
        (aMatr, bMatr)
        .Write(pResMatr)(storeFunc.GetScale(), storeFunc.GetScalePtr(), yTiles);
}


// XY,XZ->YZ
template <class T1, class T2, class TStoreType, class TRes, class TXSize, class TYSize, class TZSize>
void MatMulXYoXZeYZ(TIntrusivePtr<TGraph> c,
    const T1 &aMatr, const T2 &bMatr, TRes *pResMatr,
    TXSize &&xTiles, TYSize &&yTiles, TZSize &&zTiles, TStoreType &&storeFunc)
{
    typedef TMatMulTranspose ARot;
    typedef TMatMulDirect BRot;
    typedef TMatMulDirect ResRot;
    CudaCall(c, MatMulKernel<ARot, BRot, ResRot, typename T1::TElem, typename T2::TElem, typename TRes::TElem, TStoreType>)
        .Grid(zTiles, yTiles).Block(WARP_SIZE, MM_KERNEL_WARPS)
        (aMatr, bMatr)
        .Write(pResMatr)(storeFunc.GetScale(), storeFunc.GetScalePtr(), xTiles);
}

}
