#include "stdafx.h"
#define KERNEL_UNIT "cuda_matmul/"
#include "cuda_matmul.cuh"
#include "cuda_graph.cuh"
#include <lib/random/mersenne.h>
#include <lib/hp_timer/hp_timer.h>


using namespace NCuda;

//__global__ void TestLoadTile(TCuda2DPtr<half> data, TCuda2DPtr<half> dst)
//{
//    TTileCoord tc;
//    __shared__ T4x4SMemHalfTile frag;
//    int warpId = threadIdx.y;
//
//    Copy4x4Tile(&frag, warpId, data[0], data.GetStride());
//    __syncthreads();
//
//    if (warpId == 0) {
//        for (int x = 0; x < 4; ++x) {
//            for (int y = 0; y < 4; ++y) {
//                TRegTile<half> tt;
//                //LoadTile(&tt, frag, x, y);
//                LoadTileTransposed(&tt, frag, x, y);
//                tt.Store(tc, dst[y * 16] + x * 16, dst.GetStride());
//            }
//        }
//    }
//}


constexpr int FAST_MM_TILE = 128;

// performs res = a @ b (einsum YX,XZ -> YZ)
template <class ARotate, class BRotate, class ResRotate, class T1, class T2, class TRes, class TStore>
__global__ void FastMatMulKernel(
    TCuda2DPtr<T1> aData, TCuda2DPtr<T2> bData,
    TCuda2DPtr<TRes> resData, float storeScaleVal, float *storeScalePtr,
    int dotLen
)
{
    int warpId = threadIdx.y & 3;
    int bBlockId = threadIdx.y / 4;

    TTileCoord tc;
    int aStride = aData.GetStrideInBytes();
    int aXStep = ARotate::GetXStep(aStride, (T1 *)0);
    int aYStep = ARotate::GetYStep(aStride, (T1 *)0);

    int bStride = bData.GetStrideInBytes();
    int bXStep = BRotate::GetXStep(bStride, (T2 *)0);
    int bYStep = BRotate::GetYStep(bStride, (T2 *)0);

    int resStride = resData.GetStrideInBytes();
    int resXStep = ResRotate::GetXStep(resStride, (TRes *)0);
    int resYStep = ResRotate::GetYStep(resStride, (TRes *)0);

    ui8 *aPtr = aData.GetRawData() + (blockIdx.y * (FAST_MM_TILE / TILE)) * aYStep;
    ui8 *bPtr = bData.GetRawData() + (blockIdx.x * (FAST_MM_TILE / TILE)) * bXStep;

    int warpX = (warpId & 1) * 2;
    int warpY = warpId & 2;

    constexpr int SUPERBLOCK = 64;
    constexpr int SUPERBLOCK_TILES = SUPERBLOCK / TILE;

    __shared__ T4x4SMemHalfTile aFrag[2];
    __shared__ T4x4SMemHalfTile bFrag[2];

    // fast accums
    TRegTile<half> sum[2][2][2];
    for (int aBlockId = 0; aBlockId < 2; ++aBlockId) {
        for (int ty = 0; ty < 2; ++ty) {
            for (int tx = 0; tx < 2; ++tx) {
                sum[aBlockId][ty][tx].Clear();
            }
        }
    }
    for (int t = 0; t < dotLen * (FAST_MM_TILE / SUPERBLOCK); ++t) {
        __syncthreads();
        if (bBlockId == 0) {
            Copy4x4Tile(&aFrag[0], warpId, TCuda2DPtr<T1>(aPtr, aStride, SUPERBLOCK, SUPERBLOCK));
            Copy4x4Tile(&aFrag[1], warpId, TCuda2DPtr<T1>(aPtr + aYStep * SUPERBLOCK_TILES, aStride, SUPERBLOCK, SUPERBLOCK));
        } else {
            Copy4x4Tile(&bFrag[0], warpId, TCuda2DPtr<T2>(bPtr, bStride, SUPERBLOCK, SUPERBLOCK));
            Copy4x4Tile(&bFrag[1], warpId, TCuda2DPtr<T2>(bPtr + bXStep * SUPERBLOCK_TILES, bStride, SUPERBLOCK, SUPERBLOCK));
        }
        __syncthreads();

        for (int k = 0; k < SUPERBLOCK_TILES; ++k) {
            TRegTile<half> b[2];
            b[0] = BRotate::Rot::FragB(bFrag[bBlockId], warpX, k);
            b[1] = BRotate::Rot::FragB(bFrag[bBlockId], warpX + 1, k);
            for (int aBlockId = 0; aBlockId < 2; ++aBlockId) {
                for (int ty = 0; ty < 2; ++ty) {
                    TRegTile<half> a;
                    a = ARotate::Rot::FragA(aFrag[aBlockId], k, warpY + ty);
                    for (int tx = 0; tx < 2; ++tx) {
                        MMA(&sum[aBlockId][ty][tx], a, b[tx]);
                    }
                }
            }
        }
        aPtr += aXStep * SUPERBLOCK_TILES;
        bPtr += bYStep * SUPERBLOCK_TILES;
    }

    for (int aBlockId = 0; aBlockId < 2; ++aBlockId) {
        for (int ty = 0; ty < 2; ++ty) {
            for (int tx = 0; tx < 2; ++tx) {
                int blkX = blockIdx.x * (FAST_MM_TILE / TILE) + bBlockId * SUPERBLOCK_TILES + warpX + tx;
                int blkY = blockIdx.y * (FAST_MM_TILE / TILE) + aBlockId * SUPERBLOCK_TILES + warpY + ty;
                ui8 *resPtr = resData.GetRawData() + blkX * resXStep + blkY * resYStep;
                TStore::Store(storeScaleVal, storeScalePtr, tc, &sum[aBlockId][ty][tx], TCuda2DPtr<TRes>(resPtr, resStride, TILE, TILE), ResRotate::Rot::StoreRot());
            }
        }
    }
}

template <class T1, class T2, class TStoreType, class TRes, class TXSize, class TYSize, class TZSize>
void FastMatMulXYoZYeXZ(TIntrusivePtr<TGraph> c,
    const T1 &aMatr, const T2 &bMatr, TRes *pResMatr,
    TXSize &&xTiles, TYSize &&yTiles, TZSize &&zTiles, TStoreType &&storeFunc)
{
    typedef TMatMulDirect ARot;
    typedef TMatMulTranspose BRot;
    typedef TMatMulDirect ResRot;
    CudaCall(c, FastMatMulKernel<ARot, BRot, ResRot, typename T1::TElem, typename T2::TElem, typename TRes::TElem, TStoreType>)
        .Grid(zTiles, xTiles).Block(WARP_SIZE, 8)
        (aMatr, bMatr)
        .Write(pResMatr)(storeFunc.GetScale(), storeFunc.GetScalePtr(), yTiles);
}




template <class TRng>
static void FillRandom(TRng &rng, TStream &stream, TCuda2DArray<half> *p)
{
    yint xSize = p->GetXSize();
    yint ySize = p->GetYSize();
    TArray2D<half> data;
    data.SetSizes(xSize, ySize);
    for (yint y = 0; y < ySize; ++y) {
        for (yint x = 0; x < xSize; ++x) {
            data[y][x] = rng.GenRandReal3();
        }
    }
    p->Put(stream, data);
}

void TestMatMul()
{
    TMersenne<ui32> rng(1313);

    TStream stream;
    TCuda2DArray<half> aMatr;
    TCuda2DArray<half> bMatr;
    TCuda2DArray<float> resMatr;
    TCuda2DArray<float> resMatrRef;

    const int ITER_COUNT = 100;

    int xSize = 16 * 1024; // sample count
    int ySize = 4096; // combiner width of tt=256
    int zSize = 1024; // state dim
    aMatr.Allocate(ySize, xSize);
    bMatr.Allocate(ySize, zSize);
    resMatr.Allocate(zSize, xSize);
    resMatrRef.Allocate(zSize, xSize);

    TIntrusivePtr<TGraph> computer = new TGraph;
    {
        TGraph *c = computer.Get();
        //CudaCall(c, TestLoadTile).Block(WARP_SIZE, MM_BATCH)(aMatr).Write(&aChk);
        for (yint iter = 0; iter < ITER_COUNT; ++iter) {
            //MatMulXYoZYeXZ(c, aMatr, bMatr, &resMatrRef, xSize / MM_TILE, ySize / MM_TILE, zSize / MM_TILE, TStore());
            FastMatMulXYoZYeXZ(c, aMatr, bMatr, &resMatr, xSize / FAST_MM_TILE, ySize / FAST_MM_TILE, zSize / FAST_MM_TILE, TStore());
        }
    }

    FillRandom(rng, stream, &aMatr);
    FillRandom(rng, stream, &bMatr);
    stream.Sync();
    double maxTFlops = 0;
    for (;;) {
        NHPTimer::STime tStart;
        NHPTimer::GetTime(&tStart);
        computer->Run(stream);
        stream.Sync();
        double tPassed = NHPTimer::GetTimePassed(&tStart);
        double tFlops = 2. * ITER_COUNT * xSize * ySize * zSize / tPassed / 1e12;
        maxTFlops = Max(maxTFlops, tFlops);
        DebugPrintf("%g TFlops\n", maxTFlops);

        //resMatr.CopyToHost(stream);
        //resMatrRef.CopyToHost(stream);
        //stream.Sync();
        //TArray2D<float> ra, rb;
        //resMatr.GetAllData(&ra);
        //resMatrRef.GetAllData(&rb);
        //for (yint y = 0; y < ra.GetYSize(); ++y) {
        //    for (yint x = 0; x < ra.GetXSize(); ++x) {
        //        Y_ASSERT(ra[y][x] == rb[y][x]);
        //    }
        //}
    }
}
