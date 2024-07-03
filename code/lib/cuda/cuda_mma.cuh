#pragma once
#include "cuda_util.cuh"
#include "cuda_graph.cuh"


//
// 16x16 half float tiles with hardware accelerated matrix multiplication (tensor core)


namespace NCuda
{
const int TILE = 16;
const int TILE_GROUP = 4;
const int TILE_GROUP_SIZE = TILE * TILE_GROUP;


struct T4x4SMemHalfTile { int4 Data[32 * 16]; };
struct T4SMemHalfTile { int4 Data[32 * 4]; };
struct TSwizzledSmemTile { int4 Data[64]; };
struct TSwizzledSmemHalfTile { int4 Data[32]; };
struct TSwizzledSmemI8Tile { int2 Data[32]; };


///////////////////////////////////////////////////////////////////////////////////////////////////
// async cp utils
__forceinline __device__ ui32 GetSharedAddress(const void *ptr)
{
    return static_cast<ui32>(__cvta_generic_to_shared(ptr));
}

__forceinline __device__ void AsyncCopy16(void *dst, const void *src)
{
    ui32 sharedDstAddr = GetSharedAddress(dst);
    // best type of caching is unclear
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(sharedDstAddr), "l"(src));
}

__forceinline __device__ void WaitAsyncCopy()
{
    asm volatile("cp.async.wait_all;\n" ::);
}

__forceinline __device__ void AsyncCommitGroup()
{
    asm volatile("cp.async.commit_group;\n" ::);
}

__forceinline __device__ void AsyncWaitGroup1()
{
    asm volatile("cp.async.wait_group 1;\n" ::);
}

// synchronize multiple warps, uses same bar ID, modify to use different IDs
__forceinline__ __device__ void BarSync(int count)
{
    asm volatile("bar.sync 1, %0;" : : "r"(count));
}


///////////////////////////////////////////////////////////////////////////////////////////////////
template <class T1, class T2>
inline __device__ void AddElement(T1 *p, T2 val)
{
    T1 prevVal = *p;
    *p = prevVal + T1(val);
}

template <>
inline __device__ void AddElement(float2 *p, float2 val)
{
    float2 prevVal = *p;
    *p = make_float2(prevVal.x + val.x, prevVal.y + val.y);
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// reg tiles

struct RotRowMajor {};
struct RotColMajor {};


struct TTileCoord
{
    enum {
        num_elements = 8
    };
    int TX, TY; // 8x8 tile layout over 32 threads
public:
    __device__ TTileCoord()
    {
        int h = threadIdx.x;
        TX = (h & 3) * 2;
        TY = h / 4;
    }
    __forceinline __device__ int GetX(int elemIdx) const
    {
        return TX + (elemIdx & 1) + (elemIdx & 4) * 2;
    }
    __forceinline __device__ int GetY(int elemIdx) const
    {
        return TY + (elemIdx & 2) * 4;
    }
};


template <class T>
struct TRegTile
{};


///////////////////////////////////////////////////////////////////////////////////////////////////
template<>
struct TRegTile<half>
{
    enum {
        //num_elements = 8,
        num_packed_elements = 4
    };
    // 4 8x8 tiles
    union {
        int4 nnx; // all data
        ui32 nx[4]; // Val00, Val10, Val01, Val11;
        half2 xx[4]; // Val00, Val10, Val01, Val11;
        half x[8];
    };

    __device__ TRegTile() {}
    __device__ TRegTile(const TRegTile &rr)
    {
        for (int k = 0; k < num_packed_elements; ++k) {
            nx[k] = rr.nx[k];
        }
    }
    __device__ TRegTile& operator=(const TRegTile &rr)
    {
        for (int k = 0; k < num_packed_elements; ++k) {
            nx[k] = rr.nx[k];
        }
        return *this;
    }

    __device__ void Clear()
    {
        for (int i = 0; i < num_packed_elements; ++i) {
            nx[i] = 0;
        }
    }

    __device__ void FillEvery(float val)
    {
        for (int i = 0; i < num_packed_elements; ++i) {
            xx[i] = make_half2(val, val);
        }
    }

    __device__ void Scale(float x)
    {
        half2 mult = half2(x, x);
        for (int i = 0; i < num_packed_elements; ++i) {
            xx[i] *= mult;
        }
    }

    __device__ TRegTile<half> Transpose() const
    {
        TRegTile<half> res;
        asm("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;" : "=r"(res.nx[0]) : "r"(nx[0]));
        asm("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;" : "=r"(res.nx[2]) : "r"(nx[1]));
        asm("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;" : "=r"(res.nx[1]) : "r"(nx[2]));
        asm("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;" : "=r"(res.nx[3]) : "r"(nx[3]));
        return res;
    }

    // swizzled load/store
    __device__ void Load(const TSwizzledSmemHalfTile &ht)
    {
        nnx = ht.Data[threadIdx.x];
    }
    __device__ void Store(TSwizzledSmemHalfTile *p)
    {
        p->Data[threadIdx.x] = nnx;
    }

    // store half
    __device__ void Store(const TTileCoord &tc, TCuda2DPtr<half> p)
    {
        *(half2 *)&p[tc.TY][tc.TX] = xx[0];
        *(half2 *)&p[tc.TY][tc.TX + 8] = xx[2];
        *(half2 *)&p[tc.TY + 8][tc.TX] = xx[1];
        *(half2 *)&p[tc.TY + 8][tc.TX + 8] = xx[3];
    }
    __device__ void Store(const TTileCoord &tc, TCuda2DPtr<half> p, RotRowMajor)
    {
        Store(tc, p);
    }
    __device__ void Store(const TTileCoord &tc, TCuda2DPtr<half> p, RotColMajor)
    {
        Transpose().Store(tc, p);
    }

    // store float
    __device__ void Store(const TTileCoord &tc, TCuda2DPtr<float> p)
    {
        *(float2 *)&p[tc.TY][tc.TX] = make_float2(x[0], x[1]);
        *(float2 *)&p[tc.TY][tc.TX + 8] = make_float2(x[4], x[5]);
        *(float2 *)&p[tc.TY + 8][tc.TX] = make_float2(x[2], x[3]);
        *(float2 *)&p[tc.TY + 8][tc.TX + 8] = make_float2(x[6], x[7]);
    }
    __device__ void Store(const TTileCoord &tc, TCuda2DPtr<float> p, RotRowMajor)
    {
        Store(tc, p);
    }
    __device__ void Store(const TTileCoord &tc, TCuda2DPtr<float> p, RotColMajor)
    {
        p[tc.TX + 0][tc.TY] = x[0];
        p[tc.TX + 1][tc.TY] = x[1];
        p[tc.TX + 0][tc.TY + 8] = x[2];
        p[tc.TX + 1][tc.TY + 8] = x[3];
        p[tc.TX + 8][tc.TY] = x[4];
        p[tc.TX + 9][tc.TY] = x[5];
        p[tc.TX + 8][tc.TY + 8] = x[6];
        p[tc.TX + 9][tc.TY + 8] = x[7];
    }

    // store add
    __device__ void StoreAdd(const TTileCoord &tc, TCuda2DPtr<float> p)
    {
        // fills half of cache line, but rearranging is slower
        AddElement((float2 *)&p[tc.TY][tc.TX], make_float2(x[0], x[1]));
        AddElement((float2 *)&p[tc.TY][tc.TX + 8], make_float2(x[4], x[5]));
        AddElement((float2 *)&p[tc.TY + 8][tc.TX], make_float2(x[2], x[3]));
        AddElement((float2 *)&p[tc.TY + 8][tc.TX + 8], make_float2(x[6], x[7]));
    }
    __device__ void StoreAdd(const TTileCoord &tc, TCuda2DPtr<float> p, RotRowMajor)
    {
        StoreAdd(tc, p);
    }
    __device__ void StoreAdd(const TTileCoord &tc, TCuda2DPtr<float> p, RotColMajor)
    {
        AddElement(&p[tc.TX + 0][tc.TY], x[0]);
        AddElement(&p[tc.TX + 1][tc.TY], x[1]);
        AddElement(&p[tc.TX + 0][tc.TY + 8], x[2]);
        AddElement(&p[tc.TX + 1][tc.TY + 8], x[3]);
        AddElement(&p[tc.TX + 8][tc.TY], x[4]);
        AddElement(&p[tc.TX + 9][tc.TY], x[5]);
        AddElement(&p[tc.TX + 8][tc.TY + 8], x[6]);
        AddElement(&p[tc.TX + 9][tc.TY + 8], x[7]);
    }

    // sum values over each row and add to pSum[TILE] array
    __device__ void AddSumByRow(const TTileCoord &tc, float *pSum)
    {
        // compute sums
        half2 hsum0 = xx[0] + xx[2];
        float sum0 = float(hsum0.x) + float(hsum0.y);
        sum0 += __shfl_xor_sync(0xffffffff, sum0, 2);
        sum0 += __shfl_xor_sync(0xffffffff, sum0, 1);
        half2 hsum1 = xx[1] + xx[3];
        float sum1 = float(hsum1.x) + float(hsum1.y);
        sum1 += __shfl_xor_sync(0xffffffff, sum1, 2);
        sum1 += __shfl_xor_sync(0xffffffff, sum1, 1);
        // save result
        if (tc.TX == 0) {
            pSum[tc.TY] += sum0;
            pSum[tc.TY + 8] += sum1;
        }
    }

    // load column[TILE] and place it in every column of the tile
    template <class T>
    __device__ void ReplicateColumn(T *column)
    {
        int h = threadIdx.x;
        int ty = h / 4;
        half2 val0 = make_half2(column[ty], column[ty]);
        xx[0] = val0;
        xx[2] = val0;
        half2 val1 = make_half2(column[ty + 8], column[ty + 8]);
        xx[1] = val1;
        xx[3] = val1;
    }

    __device__ void Hadamard(const TRegTile<half> &tt)
    {
        for (int i = 0; i < 4; ++i) {
            xx[i] = __hmul2(xx[i], tt.xx[i]);
        }
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
template<>
struct TRegTile<float>
{
    enum {
        num_elements = 8
    };
    // 4 8x8 tiles
    union {
        float x[8];// Vall00a, Val00b, Val10a, Val10b, Val01a, Val01b, Val11a, Val11b;
        float2 xx[4];
        int4 nnx[2]; // all data
    };

    __device__ void Clear()
    {
        for (int i = 0; i < num_elements; ++i) {
            x[i] = 0;
        }
    }

    __device__ void FillEvery(float val)
    {
        for (int i = 0; i < num_elements; ++i) {
            x[i] = val;
        }
    }

    __device__ void Scale(float scale)
    {
        for (int i = 0; i < num_elements; ++i) {
            x[i] *= scale;
        }
    }

    // swizzled load/store
    __device__ void Load(const TSwizzledSmemTile &ht)
    {
        nnx[0] = ht.Data[threadIdx.x];
        nnx[1] = ht.Data[threadIdx.x + 32];
    }
    __device__ void Store(TSwizzledSmemTile *p)
    {
        p->Data[threadIdx.x] = nnx[0];
        p->Data[threadIdx.x + 32] = nnx[1];
    }

    // store
    __device__ void Store(const TTileCoord &tc, TCuda2DPtr<float> p)
    {
        // fills half of cache line, but rearranging is slower
        *(float2 *)&p[tc.TY][tc.TX] = xx[0];
        *(float2 *)&p[tc.TY][tc.TX + 8] = xx[2];
        *(float2 *)&p[tc.TY + 8][tc.TX] = xx[1];
        *(float2 *)&p[tc.TY + 8][tc.TX + 8] = xx[3];
    }
    __device__ void Store(const TTileCoord &tc, TCuda2DPtr<float> p, RotRowMajor)
    {
        Store(tc, p);
    }
    __device__ void Store(const TTileCoord &tc, TCuda2DPtr<float> p, RotColMajor)
    {
        p[tc.TX + 0][tc.TY] = x[0];
        p[tc.TX + 1][tc.TY] = x[1];
        p[tc.TX + 0][tc.TY + 8] = x[2];
        p[tc.TX + 1][tc.TY + 8] = x[3];
        p[tc.TX + 8][tc.TY] = x[4];
        p[tc.TX + 9][tc.TY] = x[5];
        p[tc.TX + 8][tc.TY + 8] = x[6];
        p[tc.TX + 9][tc.TY + 8] = x[7];
    }

    // store with accumulation
    __device__ void StoreAdd(const TTileCoord &tc, TCuda2DPtr<float> p)
    {
        // fills half of cache line, but rearranging is slower
        AddElement((float2 *)&p[tc.TY][tc.TX], xx[0]);
        AddElement((float2 *)&p[tc.TY][tc.TX + 8], xx[2]);
        AddElement((float2 *)&p[tc.TY + 8][tc.TX], xx[1]);
        AddElement((float2 *)&p[tc.TY + 8][tc.TX + 8], xx[3]);
    }
    __device__ void StoreAddAtomic(const TTileCoord &tc, TCuda2DPtr<float> p)
    {
        // fills half of cache line, but rearranging is slower
        atomicAddExact(&p[tc.TY][tc.TX], x[0]);
        atomicAddExact(&p[tc.TY][tc.TX + 1], x[1]);
        atomicAddExact(&p[tc.TY][tc.TX + 8], x[4]);
        atomicAddExact(&p[tc.TY][tc.TX + 9], x[5]);
        atomicAddExact(&p[tc.TY + 8][tc.TX], x[2]);
        atomicAddExact(&p[tc.TY + 8][tc.TX + 1], x[3]);
        atomicAddExact(&p[tc.TY + 8][tc.TX + 8], x[6]);
        atomicAddExact(&p[tc.TY + 8][tc.TX + 9], x[7]);
    }
    __device__ void StoreAdd(const TTileCoord &tc, TCuda2DPtr<float> p, RotRowMajor)
    {
        StoreAdd(tc, p);
    }
    __device__ void StoreAddAtomic(const TTileCoord &tc, TCuda2DPtr<float> p, RotRowMajor)
    {
        StoreAddAtomic(tc, p);
    }
    __device__ void StoreAdd(const TTileCoord &tc, TCuda2DPtr<float> p, RotColMajor)
    {
        AddElement(&p[tc.TX + 0][tc.TY], x[0]);
        AddElement(&p[tc.TX + 1][tc.TY], x[1]);
        AddElement(&p[tc.TX + 0][tc.TY + 8], x[2]);
        AddElement(&p[tc.TX + 1][tc.TY + 8], x[3]);
        AddElement(&p[tc.TX + 8][tc.TY], x[4]);
        AddElement(&p[tc.TX + 9][tc.TY], x[5]);
        AddElement(&p[tc.TX + 8][tc.TY + 8], x[6]);
        AddElement(&p[tc.TX + 9][tc.TY + 8], x[7]);
    }

    // store with conversion
    __device__ void Store(const TTileCoord &tc, TCuda2DPtr<half> p)
    {
        // fills half of cache line, but rearranging is slower
        *(half2 *)&p[tc.TY][tc.TX] = make_half2(x[0], x[1]);
        *(half2 *)&p[tc.TY][tc.TX + 8] = make_half2(x[4], x[5]);
        *(half2 *)&p[tc.TY + 8][tc.TX] = make_half2(x[2], x[3]);
        *(half2 *)&p[tc.TY + 8][tc.TX + 8] = make_half2(x[6], x[7]);
    }
    __device__ void Store(const TTileCoord &tc, TCuda2DPtr<half> p, RotRowMajor)
    {
        Store(tc, p);
    }
    __device__ void Store(const TTileCoord &tc, TCuda2DPtr<half> p, RotColMajor)
    {
        // can be done with less temporary registers then create full reg tile
        TRegTile<half> tmp;
        for (int k = 0; k < 4; ++k) {
            tmp.xx[k] = make_half2(xx[k].x, xx[k].y);
        }
        tmp.Transpose().Store(tc, p);
    }
    __device__ void StoreAdd(const TTileCoord &tc, TCuda2DPtr<half> p)
    {
        // fills half of cache line, but rearranging is slower
        *(half2 *)&p[tc.TY][tc.TX] += make_half2(x[0], x[1]);
        *(half2 *)&p[tc.TY][tc.TX + 8] += make_half2(x[4], x[5]);
        *(half2 *)&p[tc.TY + 8][tc.TX] += make_half2(x[2], x[3]);
        *(half2 *)&p[tc.TY + 8][tc.TX + 8] += make_half2(x[6], x[7]);
    }
    __device__ void Store(const TTileCoord &tc, TCuda2DPtr<i8> p)
    {
        p[tc.TY][tc.TX] = x[0];
        p[tc.TY][tc.TX + 1] = x[1];
        p[tc.TY][tc.TX + 8] = x[4];
        p[tc.TY][tc.TX + 9] = x[5];
        p[tc.TY + 8][tc.TX] = x[2];
        p[tc.TY + 8][tc.TX + 1] = x[3];
        p[tc.TY + 8][tc.TX + 8] = x[6];
        p[tc.TY + 8][tc.TX + 9] = x[7];
    }

    // there is no atomicAdd() for half2!
    //__device__ void StoreAddAtomic(const TTileCoord &tc, half *p, int stride)

    // place same value in each row, load this value from pData[TILE] array
    __device__ void LoadRows(const TTileCoord &tc, float *pData)
    {
        for (int elem = 0; elem < num_elements; ++elem) {
            int y = tc.GetY(elem);
            x[elem] = pData[y];
        }
    }

    // sum values over each row and add to pSum[TILE] array
    __device__ void AddSumByRow(const TTileCoord &tc, float *pSum)
    {
        // compute sums
        float sum0 = x[0] + x[1] + x[4] + x[5];
        sum0 += __shfl_xor_sync(0xffffffff, sum0, 2);
        sum0 += __shfl_xor_sync(0xffffffff, sum0, 1);
        float sum1 = x[2] + x[3] + x[6] + x[7];
        sum1 += __shfl_xor_sync(0xffffffff, sum1, 2);
        sum1 += __shfl_xor_sync(0xffffffff, sum1, 1);
        // save result
        if (tc.TX == 0) {
            pSum[tc.TY] += sum0;
            pSum[tc.TY + 8] += sum1;
        }
    }

    // calc max values over each row and and previous pMax[TILE] array, store new max to pMax[]
    template <class T>
    __device__ void StoreMaxByRow(const TTileCoord &tc, T *pMax)
    {
        // compute max over rows
        T max0 = max(max(x[0], x[1]), max(x[4], x[5]));
        max0 = max(max0, __shfl_xor_sync(0xffffffff, max0, 2));
        max0 = max(max0, __shfl_xor_sync(0xffffffff, max0, 1));
        T max1 = max(max(x[2], x[3]), max(x[6], x[7]));
        max1 = max(max1, __shfl_xor_sync(0xffffffff, max1, 2));
        max1 = max(max1, __shfl_xor_sync(0xffffffff, max1, 1));
        // save result
        if (tc.TX == 0) {
            pMax[tc.TY] = max0;
            pMax[tc.TY + 8] = max1;
        }
    }

    // load column[TILE] and place it in every column of the tile
    template <class T>
    __device__ void ReplicateColumn(const TTileCoord &tc, T *column)
    {
        float val0 = column[tc.TY];
        x[0] = val0;
        x[1] = val0;
        x[4] = val0;
        x[5] = val0;
        float val1 = column[tc.TY + 8];
        x[2] = val1;
        x[3] = val1;
        x[6] = val1;
        x[7] = val1;
    }

    template <class T>
    __device__ void Hadamard(const TRegTile<T> &tt)
    {
        for (int i = 0; i < num_elements; ++i) {
            x[i] *= (float)tt.x[i];
        }
    }

    // Hadamard + AddSumByRow
    template <class T>
    __device__ void AddSumByRowScaled(const TTileCoord &tc, const TRegTile<T> &tt, float *pSum)
    {
        // compute sums
        float sum0 = x[0] * float(tt.x[0]) + x[1] * float(tt.x[1]) + x[4] * float(tt.x[4]) + x[5] * float(tt.x[5]);
        sum0 += __shfl_xor_sync(0xffffffff, sum0, 2);
        sum0 += __shfl_xor_sync(0xffffffff, sum0, 1);
        float sum1 = x[2] * float(tt.x[2]) + x[3] * float(tt.x[3]) + x[6] * float(tt.x[6]) + x[7] * float(tt.x[7]);
        sum1 += __shfl_xor_sync(0xffffffff, sum1, 2);
        sum1 += __shfl_xor_sync(0xffffffff, sum1, 1);
        // save result
        if (tc.TX == 0) {
            pSum[tc.TY] += sum0;
            pSum[tc.TY + 8] += sum1;
        }
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
template<>
struct TRegTile<i8>
{
    enum {
        //num_elements = 8,
        num_packed_elements = 2
    };
    // 2 16x8 tiles
    union {
        int2 nnx;
        ui32 nx[2];// Val00, Val10
        i8 x[8];
    };

    __device__ TRegTile() {}
    __device__ TRegTile(const TRegTile &rr)
    {
        for (int k = 0; k < num_packed_elements; ++k) {
            nx[k] = rr.nx[k];
        }
    }
    __device__ TRegTile &operator=(const TRegTile &rr)
    {
        for (int k = 0; k < num_packed_elements; ++k) {
            nx[k] = rr.nx[k];
        }
        return *this;
    }

    __device__ void Clear()
    {
        for (int i = 0; i < num_packed_elements; ++i) {
            nx[i] = 0;
        }
    }

    // fast Transpose() seems to be impossible

    __device__ void Load(const TSwizzledSmemI8Tile &ht)
    {
        nnx = ht.Data[threadIdx.x];
    }

    __device__ void Store(TSwizzledSmemI8Tile *p)
    {
        p->Data[threadIdx.x] = nnx;
    }

    __device__ void Load(const TTileCoord &tc, TCuda2DPtr<i8> p)
    {
        nx[0] = *(ui32 *)&p[tc.TY][tc.TX * 2];
        nx[1] = *(ui32 *)&p[tc.TY + 8][tc.TX * 2];
    }

    __device__ void Store(const TTileCoord &tc, TCuda2DPtr<i8> p)
    {
        *(ui32 *)&p[tc.TY][tc.TX * 2] = nx[0];
        *(ui32 *)&p[tc.TY + 8][tc.TX * 2] = nx[1];
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
template<>
struct TRegTile<int>
{
    enum {
        num_elements = 8
    };
    // 4 8x8 tiles
    union {
        int x[8];// Vall00a, Val00b, Val10a, Val10b, Val01a, Val01b, Val11a, Val11b;
        int2 xx[4];
    };

    __device__ void Clear()
    {
        for (int i = 0; i < num_elements; ++i) {
            x[i] = 0;
        }
    }

    __device__ void FillEvery(int val)
    {
        for (int i = 0; i < num_elements; ++i) {
            x[i] = val;
        }
    }

    // store
    template <class T>
    __device__ void Store(const TTileCoord &tc, TCuda2DPtr<T> p) const
    {
        p[tc.TY][tc.TX] = x[0];
        p[tc.TY][tc.TX + 1] = x[1];
        p[tc.TY][tc.TX + 8] = x[4];
        p[tc.TY][tc.TX + 9] = x[5];
        p[tc.TY + 8][tc.TX] = x[2];
        p[tc.TY + 8][tc.TX + 1] = x[3];
        p[tc.TY + 8][tc.TX + 8] = x[6];
        p[tc.TY + 8][tc.TX + 9] = x[7];
    }

    __device__ void Store(const TTileCoord &tc, TCuda2DPtr<int> p) const
    {
        // fills half of cache line, but rearranging is slower
        *(int2 *)&p[tc.TY][tc.TX] = xx[0];
        *(int2 *)&p[tc.TY][tc.TX + 8] = xx[2];
        *(int2 *)&p[tc.TY + 8][tc.TX] = xx[1];
        *(int2 *)&p[tc.TY + 8][tc.TX + 8] = xx[3];
    }

    template <class T>
    __device__ void StoreLambda(const TTileCoord &tc, T func) const
    {
        // func(tx, ty, value)
        func(tc.TX, tc.TY, x[0]);
        func(tc.TX + 1, tc.TY, x[1]);
        func(tc.TX + 8, tc.TY, x[4]);
        func(tc.TX + 9, tc.TY, x[5]);
        func(tc.TX, tc.TY + 8, x[2]);
        func(tc.TX + 1, tc.TY + 8, x[3]);
        func(tc.TX + 8, tc.TY + 8, x[6]);
        func(tc.TX + 9, tc.TY + 8, x[7]);
    }

    __device__ void StoreScaled(const TTileCoord &tc, TCuda2DPtr<half> p, float scale) const
    {
        // fills half of cache line, but rearranging is slower
        *(half2 *)&p[tc.TY][tc.TX] = make_half2(x[0] * scale, x[1] * scale);
        *(half2 *)&p[tc.TY][tc.TX + 8] = make_half2(x[4] * scale, x[5] * scale);
        *(half2 *)&p[tc.TY + 8][tc.TX] = make_half2(x[2] * scale, x[3] * scale);
        *(half2 *)&p[tc.TY + 8][tc.TX + 8] = make_half2(x[6] * scale, x[7] * scale);
    }

    __device__ void StoreAddScaled(const TTileCoord &tc, TCuda2DPtr<half> p, float scale) const
    {
        // fills half of cache line, but rearranging is slower
        *(half2 *)&p[tc.TY][tc.TX] += make_half2(x[0] * scale, x[1] * scale);
        *(half2 *)&p[tc.TY][tc.TX + 8] += make_half2(x[4] * scale, x[5] * scale);
        *(half2 *)&p[tc.TY + 8][tc.TX] += make_half2(x[2] * scale, x[3] * scale);
        *(half2 *)&p[tc.TY + 8][tc.TX + 8] += make_half2(x[6] * scale, x[7] * scale);
    }

    __device__ void StoreScaled(const TTileCoord &tc, TCuda2DPtr<float> p, float scale) const
    {
        p[tc.TY][tc.TX] = x[0] * scale;
        p[tc.TY][tc.TX + 1] = x[1] * scale;
        p[tc.TY][tc.TX + 8] = x[4] * scale;
        p[tc.TY][tc.TX + 9] = x[5] * scale;
        p[tc.TY + 8][tc.TX] = x[2] * scale;
        p[tc.TY + 8][tc.TX + 1] = x[3] * scale;
        p[tc.TY + 8][tc.TX + 8] = x[6] * scale;
        p[tc.TY + 8][tc.TX + 9] = x[7] * scale;
    }

    __device__ void StoreAddScaled(const TTileCoord &tc, TCuda2DPtr<float> p, float scale) const
    {
        p[tc.TY][tc.TX] += x[0] * scale;
        p[tc.TY][tc.TX + 1] += x[1] * scale;
        p[tc.TY][tc.TX + 8] += x[4] * scale;
        p[tc.TY][tc.TX + 9] += x[5] * scale;
        p[tc.TY + 8][tc.TX] += x[2] * scale;
        p[tc.TY + 8][tc.TX + 1] += x[3] * scale;
        p[tc.TY + 8][tc.TX + 8] += x[6] * scale;
        p[tc.TY + 8][tc.TX + 9] += x[7] * scale;
    }

    // Hadamard + AddSumByRow
    template <class T>
    __device__ void AddSumByRowScaled(const TTileCoord &tc, const TRegTile<T> &tt, float *pSum) const
    {
        // compute sums
        float sum0 = x[0] * float(tt.x[0]) + x[1] * float(tt.x[1]) + x[4] * float(tt.x[4]) + x[5] * float(tt.x[5]);
        sum0 += __shfl_xor_sync(0xffffffff, sum0, 2);
        sum0 += __shfl_xor_sync(0xffffffff, sum0, 1);
        float sum1 = x[2] * float(tt.x[2]) + x[3] * float(tt.x[3]) + x[6] * float(tt.x[6]) + x[7] * float(tt.x[7]);
        sum1 += __shfl_xor_sync(0xffffffff, sum1, 2);
        sum1 += __shfl_xor_sync(0xffffffff, sum1, 1);
        // save result
        if (tc.TX == 0) {
            pSum[tc.TY] += sum0;
            pSum[tc.TY + 8] += sum1;
        }
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
// load from swizzled format
__forceinline __device__ TRegTile<half> LoadTile(const TSwizzledSmemHalfTile &ht)
{
    TRegTile<half> res;
    res.Load(ht);
    return res;
}

__forceinline __device__ TRegTile<i8> LoadTile(const TSwizzledSmemI8Tile &ht)
{
    TRegTile<i8> res;
    res.Load(ht);
    return res;
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// load 16x16 fp16 tile from shared memory
// prone to bank conflicts
__forceinline __device__ void LoadFromSmem(TRegTile<half> *p, TCuda2DPtr<half> data)
{
    int h = threadIdx.x;
    int offsetX = (h & 16) >> 1;
    int offsetY = h & 15;
    ui32 sharedAddr = GetSharedAddress(&data[offsetY][offsetX]);
    asm volatile ("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(p->nx[0]), "=r"(p->nx[1]), "=r"(p->nx[2]), "=r"(p->nx[3])
        : "r"(sharedAddr));
}

__forceinline __device__ void LoadFromSmemTransposed(TRegTile<half> *p, TCuda2DPtr<half> data)
{
    int h = threadIdx.x;
    int offsetX = (h & 16) >> 1;
    int offsetY = h & 15;
    ui32 sharedAddr = GetSharedAddress(&data[offsetY][offsetX]);
    asm volatile ("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(p->nx[0]), "=r"(p->nx[2]), "=r"(p->nx[1]), "=r"(p->nx[3])
        : "r"(sharedAddr));
}

__forceinline __device__ void CopyTileToSmem(TCuda2DPtr<half> dst, TCuda2DPtr<half> src)
{
    int h = threadIdx.x;
    int sy = h / 2;
    int sx = h & 1;
    *(float4 *)&dst[sy][sx * 8] = *(float4 *)&src[sy][sx * 8];
}

__forceinline __device__ void CopyTileToSmemAsync(TCuda2DPtr<half> dst, TCuda2DPtr<half> src)
{
    int h = threadIdx.x;
    int sy = h / 2;
    int sx = h & 1;
    AsyncCopy16(&dst[sy][sx * 8], &src[sy][sx * 8]);
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// shared memory 64x64 fp16 tile
// we operate 8x1 blocks (16 bytes), block address is computed as [y][x ^ (y&7)]
// using xor operation avoids bank conflicts in both situations
//   when storing blocks to smem horizontally (copying mem -> smem)
//   when load blocks from smem vertically (copying smem -> registers)

__forceinline __device__ void LoadTile(TRegTile<half> *p, const T4x4SMemHalfTile &ht, int x, int y)
{
    int h = threadIdx.x;
    int y7 = h & 7;
    int tx = h / 16;
    int ty = h & 15;
    int threadOffset = (ty * 8 + tx);
    int rowAddr = threadOffset + y * (16 * 8) + x * 2;
    ui32 sharedAddr = GetSharedAddress(&ht.Data[rowAddr ^ y7]);
    asm volatile ("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(p->nx[0]), "=r"(p->nx[1]), "=r"(p->nx[2]), "=r"(p->nx[3])
        : "r"(sharedAddr));
}

__forceinline __device__ void LoadTileTransposed(TRegTile<half> *p, const T4x4SMemHalfTile &ht, int x, int y)
{
    int h = threadIdx.x;
    int y7 = h & 7;
    int tx = h / 16;
    int ty = h & 15;
    int threadOffset = (ty * 8 + tx);
    int rowAddr = threadOffset + x * (16 * 8) + y * 2;
    ui32 sharedAddr = GetSharedAddress(&ht.Data[rowAddr ^ y7]);

    asm volatile ("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(p->nx[0]), "=r"(p->nx[2]), "=r"(p->nx[1]), "=r"(p->nx[3])
        : "r"(sharedAddr));
}

__forceinline __device__ TRegTile<half> LoadTile(const T4x4SMemHalfTile &ht, int x, int y)
{
    TRegTile<half> res;
    LoadTile(&res, ht, x, y);
    return res;
}

__forceinline __device__ TRegTile<half> LoadTileTransposed(const T4x4SMemHalfTile &ht, int x, int y)
{
    TRegTile<half> res;
    LoadTileTransposed(&res, ht, x, y);
    return res;
}

// load with 4 warps
__forceinline __device__ void Copy4x4Tile(T4x4SMemHalfTile *p, int warpId, TCuda2DPtr<half> data)
{
    int h = threadIdx.x;
    for (int baseY = 0; baseY < 64; baseY += 16) {
        int x = h & 7;
        int y = (h / 8) + warpId * 4 + baseY;
        int y7 = y & 7;
        p->Data[(y * 8 + x) ^ y7] = *(int4 *)&data[y][x * 8];
    }
}

// load with 4 warps
__forceinline __device__ void Copy4x4TileAsync(T4x4SMemHalfTile *p, int warpId, TCuda2DPtr<half> data)
{
    int h = threadIdx.x;
    for (int baseY = 0; baseY < 64; baseY += 16) {
        int x = h & 7;
        int y = (h / 8) + warpId * 4 + baseY;
        int y7 = y & 7;
        AsyncCopy16(&p->Data[(y * 8 + x) ^ y7], &data[y][x * 8]);
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// shared memory 64x16 fp16 tile
__forceinline __device__ void LoadTile(TRegTile<half> *p, const T4SMemHalfTile &ht, int x)
{
    LoadTile(p, *(const T4x4SMemHalfTile *)&ht, x, 0);
}

__forceinline __device__ void LoadTileTransposed(TRegTile<half> *p, const T4SMemHalfTile &ht, int x)
{
    LoadTileTransposed(p, *(const T4x4SMemHalfTile *)&ht, 0, x);
}

__forceinline __device__ TRegTile<half> LoadTile(const T4SMemHalfTile &ht, int x)
{
    TRegTile<half> res;
    LoadTile(&res, ht, x);
    return res;
}

__forceinline __device__ TRegTile<half> LoadTileTransposed(const T4SMemHalfTile &ht, int x)
{
    TRegTile<half> res;
    LoadTileTransposed(&res, ht, x);
    return res;
}

// load with 4 warps
__forceinline __device__ void Copy4Tile(T4SMemHalfTile *p, int warpId, TCuda2DPtr<half> data)
{
    int h = threadIdx.x;
    int x = h & 7;
    int y = (h / 8) + warpId * 4;
    int y7 = y & 7;
    p->Data[(y * 8 + x) ^ y7] = *(int4 *)&data[y][x * 8];
}

__forceinline __device__ void Copy4TileAsync(T4SMemHalfTile *p, int warpId, TCuda2DPtr<half> data)
{
    int h = threadIdx.x;
    int x = h & 7;
    int y = (h / 8) + warpId * 4;
    int y7 = y & 7;
    AsyncCopy16(&p->Data[(y * 8 + x) ^ y7], &data[y][x * 8]);
}

// load with 1 warp
__forceinline __device__ void Copy4Tile(T4SMemHalfTile *p, TCuda2DPtr<half> data)
{
    for (int warpId = 0; warpId < 4; ++warpId) {
        Copy4Tile(p, warpId, data);
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// shared memory 128x16 i8 tile
const int I8_TILE_GROUP = 8;
const int I8_TILE_GROUP_SIZE = TILE * I8_TILE_GROUP;

struct T8SMemI8Tile { int4 Data[16 * 8]; };

// load with 4 warps
__forceinline __device__ void Copy8Tile(T8SMemI8Tile *p, int warpId, TCuda2DPtr<i8> data)
{
    int h = threadIdx.x;
    int x = h & 7;
    int y = (h / 8) + warpId * 4;
    int y7 = y & 7;
    p->Data[(y * 8 + x) ^ y7] = *(int4 *)&data[y][x * 16];
}

// load with 1 warp
__forceinline __device__ void Copy8Tile(T8SMemI8Tile *p, TCuda2DPtr<i8> data)
{
    for (int warpId = 0; warpId < 4; ++warpId) {
        Copy8Tile(p, warpId, data);
    }
}

__forceinline __device__ void LoadTile(TRegTile<i8> *p, const T8SMemI8Tile &ht, int tileId)
{
    int h = threadIdx.x;
    int y7 = h & 7;
    int ty = h; // we use ldmatrix.x2, so only first 16 threads are utilized, first 8 threads load upper 16x8 tile, second 8 threads load bottom 16x8 tile
    int rowAddr = ty * 8 + tileId;
    ui32 sharedAddr = GetSharedAddress(&ht.Data[rowAddr ^ y7]);
    asm volatile ("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
        : "=r"(p->nx[0]), "=r"(p->nx[1])
        : "r"(sharedAddr));
}

__forceinline __device__ TRegTile<i8> LoadTile(const T8SMemI8Tile &ht, int x)
{
    TRegTile<i8> res;
    LoadTile(&res, ht, x);
    return res;
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// shared memory 64x16 i8 tile, 64 is half cache line, so this code underutilizes global mem bandwidth when used with large strides

struct T4SMemI8Tile { int4 Data[16 * 4]; };

// load with 2 warps
__forceinline __device__ void Copy4Tile(T4SMemI8Tile *p, int warpId, TCuda2DPtr<i8> data)
{
    if (warpId >= 2) {
        return;
    }
    int h = threadIdx.x;
    int x = h & 3;
    int y = (h / 4) + warpId * 8;
    int y7 = y & 7;
    p->Data[(y * 4 + x) ^ y7] = *(int4 *)&data[y][x * 16];
}

__forceinline __device__ void Copy4TileAsync(T4SMemI8Tile *p, int warpId, TCuda2DPtr<i8> data)
{
    if (warpId >= 2) {
        return;
    }
    int h = threadIdx.x;
    int x = h & 3;
    int y = (h / 4) + warpId * 8;
    int y7 = y & 7;
    AsyncCopy16(&p->Data[(y * 4 + x) ^ y7], &data[y][x * 16]);
}

// load with 1 warp
__forceinline __device__ void Copy4Tile(T4SMemI8Tile *p, TCuda2DPtr<i8> data)
{
    for (int warpId = 0; warpId < 2; ++warpId) {
        Copy4Tile(p, warpId, data);
    }
}

__forceinline __device__ void LoadTile(TRegTile<i8> *p, const T4SMemI8Tile &ht, int tileId)
{
    int h = threadIdx.x;
    int y7 = h & 7;
    int ty = h; // we use ldmatrix.x2, so only first 16 threads are utilized, first 8 threads load upper 16x8 tile, second 8 threads load bottom 16x8 tile
    int rowAddr = ty * 4 + tileId;
    ui32 sharedAddr = GetSharedAddress(&ht.Data[rowAddr ^ y7]);
    asm volatile ("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
        : "=r"(p->nx[0]), "=r"(p->nx[1])
        : "r"(sharedAddr));
}

__forceinline __device__ TRegTile<i8> LoadTile(const T4SMemI8Tile &ht, int x)
{
    TRegTile<i8> res;
    LoadTile(&res, ht, x);
    return res;
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// shared memory 64x16 and 64x64 fp16 tile, conversion from i8 source
union TConvertFloat4Half8
{
    int4 Int4;
    half Half8[8];
};

union TConvertFloat2Char8
{
    int2 Int2;
    i8 Char8[8];
};

__forceinline __device__ int4 ConvertChar8toHalf8(int2 arg)
{
    TConvertFloat2Char8 src;
    src.Int2 = arg;
    TConvertFloat4Half8 dst;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        dst.Half8[k] = src.Char8[k];
    }
    return dst.Int4;
}

// load with 4 warps
__forceinline __device__ void Copy4Tile(T4SMemHalfTile *p, int warpId, TCuda2DPtr<i8> data)
{
    int h = threadIdx.x;
    int x = h & 7;
    int y = (h / 8) + warpId * 4;
    int y7 = y & 7;
    int2 src = *(int2 *)&data[y][x * 8];
    p->Data[(y * 8 + x) ^ y7] = ConvertChar8toHalf8(src);
}

// load with 1 warp
__forceinline __device__ void Copy4Tile(T4SMemHalfTile *p, TCuda2DPtr<i8> data)
{
    for (int warpId = 0; warpId < 4; ++warpId) {
        Copy4Tile(p, warpId, data);
    }
}

// load with 4 warps
__forceinline __device__ void Copy4x4Tile(T4x4SMemHalfTile *p, int warpId, TCuda2DPtr<i8> data)
{
    int h = threadIdx.x;
    for (int baseY = 0; baseY < 64; baseY += 16) {
        int x = h & 7;
        int y = (h / 8) + warpId * 4 + baseY;
        int y7 = y & 7;
        int2 src = *(int2 *)&data[y][x * 8];
        p->Data[(y * 8 + x) ^ y7] = ConvertChar8toHalf8(src);
    }
}


// convert 4 tiles with 4 warps, layout is different, use correct layout for half & i8 blocks
__forceinline __device__ void Convert4Tile(T4SMemHalfTile *p, int warpId, const T4SMemI8Tile &src)
{
    int h = threadIdx.x;
    int x = h & 7;
    int y = (h / 8) + warpId * 4;
    int y7 = y & 7;
    int2 *pSrc = (int2 *)&src.Data[(y * 4 + x / 2) ^ y7];
    p->Data[(y * 8 + x) ^ y7] = ConvertChar8toHalf8(pSrc[x & 1]);
}

// convert 4 tiles with 4 warps
__forceinline __device__ void Convert4Tile(T4SMemHalfTile *p, int warpId, const T4SMemHalfTile &src)
{
    // no conversion needed
    int id = warpId * WARP_SIZE + threadIdx.x;
    p->Data[id] = src.Data[id];
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// tile mma, Matrix Multiply Add operations
// mul row col
__forceinline __device__ void MMA(TRegTile<float> *pD, const TRegTile<half> &a, const TRegTile<half> &b, const TRegTile<float> &c)
{
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
        " { %0, %1, %2, %3 }," // D
        " { %4, %5, %6, %7 }," // A
        " { %8, %9 }," // B
        " { %10, %11, %12, %13 };" // C
        :
    "=f"(pD->x[0]), "=f"(pD->x[1]), "=f"(pD->x[2]), "=f"(pD->x[3]) // "=f" means overwrite, "+f" means read-modify-write
        :
        "r"(a.nx[0]), "r"(a.nx[1]), "r"(a.nx[2]), "r"(a.nx[3]),
        "r"(b.nx[0]), "r"(b.nx[2]),
        "f"(c.x[0]), "f"(c.x[1]), "f"(c.x[2]), "f"(c.x[3])
        );
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
        " { %0, %1, %2, %3 }," // D
        " { %4, %5, %6, %7 }," // A
        " { %8, %9 }," // B
        " { %10, %11, %12, %13 };" // C
        :
    "=f"(pD->x[4]), "=f"(pD->x[5]), "=f"(pD->x[6]), "=f"(pD->x[7])
        :
        "r"(a.nx[0]), "r"(a.nx[1]), "r"(a.nx[2]), "r"(a.nx[3]),
        "r"(b.nx[1]), "r"(b.nx[3]),
        "f"(c.x[4]), "f"(c.x[5]), "f"(c.x[6]), "f"(c.x[7])
        );
}

__forceinline __device__ void MMA(TRegTile<float> *pD, const TRegTile<half> &a, const TRegTile<half> &b)
{
    MMA(pD, a, b, *pD);
}

__forceinline __device__ void MMA(TRegTile<half> *pD, const TRegTile<half> &a, const TRegTile<half> &b, const TRegTile<half> &c)
{
    asm("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16"
        " { %0, %1 }," // D
        " { %2, %3, %4, %5 }," // A
        " { %6, %7 }," // B
        " { %8, %9 };" // C
        :
    "=r"(pD->nx[0]), "=r"(pD->nx[1])
        :
        "r"(a.nx[0]), "r"(a.nx[1]), "r"(a.nx[2]), "r"(a.nx[3]),
        "r"(b.nx[0]), "r"(b.nx[2]),
        "r"(c.nx[0]), "r"(c.nx[1])
        );
    asm("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16"
        " { %0, %1 }," // D
        " { %2, %3, %4, %5 }," // A
        " { %6, %7 }," // B
        " { %8, %9 };" // C
        :
    "=r"(pD->nx[2]), "=r"(pD->nx[3])
        :
        "r"(a.nx[0]), "r"(a.nx[1]), "r"(a.nx[2]), "r"(a.nx[3]),
        "r"(b.nx[1]), "r"(b.nx[3]),
        "r"(c.nx[2]), "r"(c.nx[3])
        );
}

__forceinline __device__ void MMA(TRegTile<half> *pD, const TRegTile<half> &a, const TRegTile<half> &b)
{
    MMA(pD, a, b, *pD);
}


__forceinline __device__ void MMA(TRegTile<int> *pD, const TRegTile<i8> &a, const TRegTile<i8> &b, const TRegTile<int> &c)
{
    asm("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32"
        " { %0, %1, %2, %3 }," // D
        " { %4, %5 }," // A
        " { %6 }," // B
        " { %7, %8, %9, %10 };" // C
        :
    "=r"(pD->x[0]), "=r"(pD->x[1]), "=r"(pD->x[2]), "=r"(pD->x[3]) // "=f" means overwrite, "+f" means read-modify-write
        :
        "r"(a.nx[0]), "r"(a.nx[1]),
        "r"(b.nx[0]),
        "r"(c.x[0]), "r"(c.x[1]), "r"(c.x[2]), "r"(c.x[3])
        );
    asm("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32"
        " { %0, %1, %2, %3 }," // D
        " { %4, %5 }," // A
        " { %6 }," // B
        " { %7, %8, %9, %10 };" // C
        :
    "=r"(pD->x[4]), "=r"(pD->x[5]), "=r"(pD->x[6]), "=r"(pD->x[7])
        :
        "r"(a.nx[0]), "r"(a.nx[1]),
        "r"(b.nx[1]),
        "r"(c.x[4]), "r"(c.x[5]), "r"(c.x[6]), "r"(c.x[7])
        );
}

__forceinline __device__ void MMA(TRegTile<int> *pD, const TRegTile<i8> &a, const TRegTile<i8> &b)
{
    MMA(pD, a, b, *pD);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
struct TMmaRowMajor
{
    typedef RotRowMajor StoreRot;

    static __device__ TRegTile<half> FragA(const TRegTile<half> &tt) { return tt; }
    static __device__ TRegTile<half> FragB(const TRegTile<half> &tt) { return tt.Transpose(); }
    static __device__ TRegTile<half> FragA(const T4x4SMemHalfTile &ht, int x, int y) { return LoadTile(ht, x, y); }
    static __device__ TRegTile<half> FragB(const T4x4SMemHalfTile &ht, int x, int y) { return LoadTileTransposed(ht, y, x); }
    static __device__ TRegTile<half> FragA(const T4SMemHalfTile &ht, int k) { return LoadTile(ht, k); }
    static __device__ TRegTile<half> FragB(const T4SMemHalfTile &ht, int k) { return LoadTileTransposed(ht, k); }
    static __device__ TRegTile<half> FragA(const TCuda2DPtr<half> data) { TRegTile<half> res; LoadFromSmem(&res, data); return res; }
    static __device__ TRegTile<half> FragB(const TCuda2DPtr<half> data) { TRegTile<half> res; LoadFromSmemTransposed(&res, data); return res; }
    static __device__ TRegTile<i8> FragA(const TRegTile<i8> &tt) { return tt; }
    static __device__ TRegTile<i8> FragA(const T8SMemI8Tile &ht, int k) { return LoadTile(ht, k); }
    static __device__ TRegTile<i8> FragA(const T4SMemI8Tile &ht, int k) { return LoadTile(ht, k); }
};

struct TMmaColMajor
{
    typedef RotColMajor StoreRot;

    static __device__ TRegTile<half> FragA(const TRegTile<half> &tt) { return tt.Transpose(); }
    static __device__ TRegTile<half> FragB(const TRegTile<half> &tt) { return tt; }
    static __device__ TRegTile<half> FragA(const T4x4SMemHalfTile &ht, int x, int y) { return LoadTileTransposed(ht, x, y); }
    static __device__ TRegTile<half> FragB(const T4x4SMemHalfTile &ht, int x, int y) { return LoadTile(ht, y, x); }
    static __device__ TRegTile<half> FragA(const T4SMemHalfTile &ht, int k) { return LoadTileTransposed(ht, k); }
    static __device__ TRegTile<half> FragB(const T4SMemHalfTile &ht, int k) { return LoadTile(ht, k); }
    static __device__ TRegTile<half> FragA(const TCuda2DPtr<half> data) { TRegTile<half> res; LoadFromSmemTransposed(&res, data); return res; }
    static __device__ TRegTile<half> FragB(const TCuda2DPtr<half> data) { TRegTile<half> res; LoadFromSmem(&res, data); return res; }
    static __device__ TRegTile<i8> FragB(const TRegTile<i8> &tt) { return tt; }
    static __device__ TRegTile<i8> FragB(const T8SMemI8Tile &ht, int k) { return LoadTile(ht, k); }
    static __device__ TRegTile<i8> FragB(const T4SMemI8Tile &ht, int k) { return LoadTile(ht, k); }
};


void TestMMA();
}
