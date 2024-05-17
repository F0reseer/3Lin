#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>


namespace NCuda
{
///////////////////////////////////////////////////////////////////////////////////////////////////
// cuda stream
class TStream : public TNonCopyable
{
    cudaStream_t Stream;
public:
    TStream() { cudaStreamCreate(&Stream); }
    ~TStream() { cudaStreamDestroy(Stream); }
    void Sync() { cudaStreamSynchronize(Stream); }
    operator cudaStream_t() const { return Stream; }
};


#ifdef NDEBUG
#define CUDA_ASSERT( B )
#else
#define CUDA_ASSERT( B ) if (!(B)) { printf("assert failed\n"); }
#endif

///////////////////////////////////////////////////////////////////////////////////////////////////
// cuda arrays
enum EMemType
{
    MT_HOST,
    MT_DEVICE,
};

struct TMemoryBlob
{
    void *Ptr = 0;
    yint Stride = 0; // in bytes
    yint RowCount = 0;

    TMemoryBlob(void *p, yint stride, yint rowCount) : Ptr(p), Stride(stride), RowCount(rowCount) {}
    TMemoryBlob(void *p, yint stride) : Ptr(p), Stride(stride), RowCount(1) {}
    yint GetSize() const { return RowCount * Stride; }
    bool IsSameSize(const TMemoryBlob &x) const
    {
        return Stride == x.Stride && RowCount == x.RowCount;
    }
    template <class T>
    T *GetElementAddress(yint x, yint y) const
    {
        CUDA_ASSERT(y >= 0 && y < RowCount);
        char *buf = (char *)Ptr;
        return (T*)(buf + y * Stride + x * sizeof(T));
    }
};


template<class T>
class TCudaPOD
{
    const void *Owner; // pointer to owner of this pod (array/vector for example)
    void *Data;
public:
    TCudaPOD(const void *owner, void *pData) : Owner(owner), Data(pData) {}
    T *GetDevicePtr() const { return (T*)Data; }
    const void *GetOwner() const { return Owner; }
};


template<class T>
class TCudaVector : public TNonCopyable
{
    yint Count, SizeInBytes;
    char *HostBuf;
    void *DeviceBuf;
    void *DeviceData;

    enum {
        CUDA_ALLOC,
        CUDA_MAP,
    };

    void AllocateImpl(yint count, int deviceAlloc, int hostAllocFlag)
    {
        Free();
        Count = count;
        SizeInBytes = sizeof(T) * Count;
        Y_VERIFY(SizeInBytes <= 0x20000000ll); // we are using int offsets for perf
        if (deviceAlloc == CUDA_ALLOC) {
            Y_VERIFY(cudaMalloc(&DeviceBuf, SizeInBytes) == cudaSuccess);
            DeviceData = DeviceBuf;
        }
        if (hostAllocFlag != -1) {
            Y_VERIFY(cudaHostAlloc(&HostBuf, SizeInBytes, hostAllocFlag) == cudaSuccess);
            if (DeviceData == 0) {
                Y_ASSERT(deviceAlloc == CUDA_MAP);
                Y_VERIFY(cudaHostGetDevicePointer(&DeviceData, HostBuf, 0) == cudaSuccess);
            }
        }
    }
    void Free()
    {
        if (HostBuf) {
            cudaFreeHost(HostBuf);
            HostBuf = 0;
        }
        if (DeviceBuf) {
            cudaFree(DeviceBuf);
            DeviceBuf = 0;
        }
    }
    void CopyToDevice(const TStream &stream, yint sizeInBytes)
    {
        if (DeviceBuf) {
            cudaMemcpyAsync(DeviceBuf, HostBuf, sizeInBytes, cudaMemcpyHostToDevice, stream);
        }
    }
    void CopyToHostImpl(const TStream &stream, yint sizeInBytes)
    {
        if (DeviceBuf) {
            cudaMemcpyAsync(HostBuf, DeviceBuf, sizeInBytes, cudaMemcpyDeviceToHost, stream);
        }
    }
public:
    TCudaVector() : Count(0), HostBuf(0), DeviceBuf(0), DeviceData(0) {}
    ~TCudaVector() { Free(); }
    void AllocateWC(yint count)
    {
        AllocateImpl(count, CUDA_ALLOC, cudaHostAllocWriteCombined);
    }
    void Allocate(yint count)
    {
        AllocateImpl(count, CUDA_ALLOC, cudaHostAllocDefault);
    }
    void AllocateHost(yint count)
    {
        AllocateImpl(count, CUDA_MAP, cudaHostAllocDefault);
    }
    void AllocateCuda(yint count)
    {
        AllocateImpl(count, CUDA_ALLOC, -1);
    }
    void Put(const TStream &stream, const TVector<T> &data)
    {
        Y_VERIFY(YSize(data) <= Count);
        if (!data.empty()) {
            yint sizeInBytes = sizeof(T) * YSize(data);
            memcpy(HostBuf, &data[0], sizeInBytes);
            CopyToDevice(stream, sizeInBytes);
        }
    }
    void CopyToDevice(const TStream &stream)
    {
        CopyToDevice(SizeInBytes);
    }
    void CopyToHost(const TStream &stream)
    {
        CopyToHostImpl(stream, SizeInBytes);
    }
    void CopyToHost(const TStream &stream, yint sz)
    {
        Y_ASSERT(sz <= Count);
        CopyToHostImpl(stream, sz * sizeof(T));
    }
    void ClearHostMem()
    {
        memset(HostBuf, 0, SizeInBytes);
    }
    void ClearDeviceMem(const TStream &stream)
    {
        if (DeviceBuf) {
            cudaMemsetAsync(DeviceBuf, 0, SizeInBytes, stream);
        } else {
            // no device memory in this case
            Y_VERIFY(0);
            //ClearHostMem();
        }
    }
    TMemoryBlob GetDeviceMem() const
    {
        return TMemoryBlob(DeviceData, SizeInBytes);
    }
    TMemoryBlob GetHostMem() const
    {
        return TMemoryBlob(HostBuf, SizeInBytes);
    }
    TMemoryBlob GetMem(EMemType mt) const
    {
        return (mt == MT_HOST) ? GetHostMem() : GetDeviceMem();
    }
    T *GetDevicePtr() const
    {
        return (T*)DeviceData;
    }
    T* GetHostPtr() const
    {
        return (T*)HostBuf;
    }
    TCudaPOD<T> GetElement(yint idx) const
    {
        Y_ASSERT(idx >= 0 && idx < Count);
        return TCudaPOD<T>(this, GetDevicePtr() + idx);
    }
    void GetAllData(TVector<T> *res)
    {
        GetData(res, Count);
    }
    void GetData(TVector<T> *res, yint sz)
    {
        Y_ASSERT(sz <= Count);
        res->resize(sz);
        memcpy(&(*res)[0], HostBuf, sizeof(T) * sz);
    }
    yint GetSize() const
    {
        return Count;
    }
};


template<class T>
struct TCuda2DPtr
{
    ui8 *Data;
    int Stride; // int or size_t?
#ifndef NDEBUG
    int XSize, YSize;
#endif

    __host__ __device__ TCuda2DPtr(void *data, int stride, int xSize, int ySize) : Data((ui8*)data), Stride(stride)
    {
#ifndef NDEBUG
        XSize = xSize;
        YSize = ySize;
#endif
    }
    __forceinline __device__ ui8 *GetRawData() const { return Data; }
    __forceinline __device__ T *operator[](int y) const
    {
        CUDA_ASSERT(y >= 0 && y < YSize);
        return (T *)(Data + y * Stride);
    }
    __forceinline __device__ TCuda2DPtr<T> Fragment(int xOffset, int yOffset) const
    {
#ifndef NDEBUG
        CUDA_ASSERT(xOffset < XSize && yOffset < YSize);
        return TCuda2DPtr<T>(Data + yOffset * Stride + xOffset * sizeof(T), Stride, XSize - xOffset, YSize - yOffset);
#else
        return TCuda2DPtr<T>(Data + yOffset * Stride + xOffset * sizeof(T), Stride, 0, 0);
#endif
    }
    __forceinline __device__ int GetStrideInBytes() const
    {
        return Stride;
    }
};


template<class T>
struct THost2DPtr
{
    yint XSize;
    yint YSize;
    T *Ptr;
    int Stride;

    THost2DPtr(yint xSize, yint ySize, T *p, int s) : XSize(xSize), YSize(ySize), Ptr(p), Stride(s) {}
    T *operator[](yint y) const
    {
        Y_ASSERT(y >= 0 && y < YSize);
        return Ptr + y * Stride;
    }
    yint GetXSize() const { return XSize; }
    yint GetYSize() const { return YSize; }
};


template<class T>
class TCuda2DArrayFragment
{
    const void *Owner; // pointer to owner of this pod (array/vector for example)
    yint Stride;
    yint ColumnCount;
    yint RowCount;
    void *DeviceData;
public:
    typedef T TElem;

    TCuda2DArrayFragment(const void *owner, yint stride, yint columnCount, yint rowCount, void *pData)
        : Owner(owner), Stride(stride), ColumnCount(columnCount), RowCount(rowCount), DeviceData(pData)
    {
    }
    TCuda2DPtr<T> GetDevicePtr() const
    {
        return TCuda2DPtr<T>(DeviceData, Stride, ColumnCount, RowCount);
    }
    const void *GetOwner() const { return Owner; }
};


template<class T>
class TCuda2DArray : public TNonCopyable
{
    size_t Stride, ColumnCount, RowCount; // stride in bytes
    char *HostBuf;
    char *DeviceBuf;
    void *DeviceData;

    enum {
        CUDA_ALLOC,
        CUDA_MAP,
    };

    yint SelectStride(yint columnCount)
    {
        yint widthInBytes = columnCount * sizeof(T);
        if ((widthInBytes & (widthInBytes - 1)) == 0) {
            return columnCount;
        } else {
            if (widthInBytes < 128) {
                yint res = 1 * sizeof(T);
                while (res < widthInBytes) {
                    res *= 2;
                }
                return res / sizeof(T);
            } else {
                // can result in rows to cache lines misalignment for non pow2 sizeof(T)
                // need Stride in bytes everywhere to fix this
                return (DivCeil(widthInBytes, 128) * 128) / sizeof(T);
            }
        }
    }

    void AllocateImpl(yint columnCount, yint rowCount, int deviceAlloc, int hostAllocFlag)
    {
        Y_ASSERT(DeviceBuf == 0 && HostBuf == 0);
        ColumnCount = columnCount;
        RowCount = rowCount;
        Stride = SelectStride(columnCount) * sizeof(T);
        yint sizeInBytes = Stride * RowCount;
        Y_VERIFY(sizeInBytes <= 0x80000000ll); // we are using int offsets for perf
        if (deviceAlloc == CUDA_ALLOC) {
            Y_VERIFY(cudaMalloc(&DeviceBuf, sizeInBytes) == cudaSuccess);
            DeviceData = DeviceBuf;
        }
        if (hostAllocFlag != -1) {
            Y_VERIFY(cudaHostAlloc(&HostBuf, sizeInBytes, hostAllocFlag) == cudaSuccess);
            if (DeviceData == 0) {
                Y_ASSERT(deviceAlloc == CUDA_MAP);
                Y_VERIFY(cudaHostGetDevicePointer(&DeviceData, HostBuf, 0) == cudaSuccess);
            }
        }
    }
    void CopyToHostImpl(const TStream &stream, yint rowCount)
    {
        if (DeviceBuf) {
            cudaMemcpyAsync(HostBuf, DeviceBuf, Stride * rowCount, cudaMemcpyDeviceToHost, stream);
        }
    }
    void CopyToDevice(const TStream &stream, yint rowCount)
    {
        if (DeviceBuf) {
            cudaMemcpyAsync(DeviceBuf, HostBuf, Stride * rowCount, cudaMemcpyHostToDevice, stream);
        }
    }
public:
    typedef T TElem;

    TCuda2DArray() : Stride(0), ColumnCount(0), RowCount(0), HostBuf(0), DeviceBuf(0), DeviceData(0) {}
    ~TCuda2DArray()
    {
        if (HostBuf) {
            cudaFreeHost(HostBuf);
        }
        if (DeviceBuf) {
            cudaFree(DeviceBuf);
        }
    }
    void AllocateWC(yint columnCount, yint rowCount)
    {
        AllocateImpl(columnCount, rowCount, CUDA_ALLOC, cudaHostAllocWriteCombined);
    }
    void Allocate(yint columnCount, yint rowCount)
    {
        AllocateImpl(columnCount, rowCount, CUDA_ALLOC, cudaHostAllocDefault);
    }
    void AllocateHost(yint columnCount, yint rowCount)
    {
        AllocateImpl(columnCount, rowCount, CUDA_MAP, cudaHostAllocDefault);
    }
    void AllocateCuda(yint columnCount, yint rowCount)
    {
        AllocateImpl(columnCount, rowCount, CUDA_ALLOC, -1);
    }
    void PutHost(const TArray2D<T> &a)
    {
        yint rowCount = a.GetYSize();
        Y_ASSERT(rowCount <= RowCount);
        for (yint row = 0; row < rowCount; ++row) {
            yint widthInBytes = a.GetXSize() * sizeof(T);
            if (widthInBytes > 0) {
                Y_ASSERT(widthInBytes <= Stride);
                char *dst = HostBuf + Stride * row;
                memcpy(dst, &a[row][0], widthInBytes);
                if (widthInBytes < Stride) {
                    // write combined memory works faster with fully written cache lines
                    memset(dst + widthInBytes, 0, Stride - widthInBytes);
                }
            }
        }
    }
    void Put(const TStream &stream, const TArray2D<T> &a)
    {
        PutHost(a);
        CopyToDevice(stream, a.GetYSize());
    }
    void Clear(const TStream &stream)
    {
        ClearDeviceMem(stream);
        ClearHostMem();
    }
    void CopyToHost(const TStream &stream)
    {
        CopyToHostImpl(stream, RowCount);
    }
    void CopyToHost(const TStream &stream, yint rowCount)
    {
        Y_ASSERT(rowCount <= RowCount);
        CopyToHostImpl(stream, rowCount);
    }
    void CopyToDevice(const TStream &stream)
    {
        CopyToDevice(stream, RowCount);
    }
    void ClearHostMem()
    {
        memset(HostBuf, 0, Stride * RowCount);
    }
    void ClearDeviceMem(const TStream &stream)
    {
        if (DeviceBuf) {
            cudaMemsetAsync(DeviceBuf, 0, Stride * RowCount, stream);
        } else {
            ClearHostMem();
        }
    }
    TMemoryBlob GetDeviceMem() const
    {
        return TMemoryBlob(DeviceData, Stride, RowCount);
    }
    TMemoryBlob GetHostMem() const
    {
        return TMemoryBlob(HostBuf, Stride, RowCount);
    }
    TMemoryBlob GetMem(EMemType mt) const
    {
        return (mt == MT_HOST) ? GetHostMem() : GetDeviceMem();
    }
    TCuda2DPtr<T> GetDevicePtr() const
    {
        return TCuda2DPtr<T>(DeviceData, Stride, ColumnCount, RowCount);
    }
    THost2DPtr<T> GetHostPtr() const
    {
        return THost2DPtr<T>(ColumnCount, RowCount, (T*)HostBuf, Stride / sizeof(T));
    }
    void GetAllData(TArray2D<T> *res) const
    {
        res->SetSizes(ColumnCount, RowCount);
        for (yint y = 0; y < RowCount; ++y) {
            for (yint x = 0; x < ColumnCount; ++x) {
                (*res)[y][x] = *(T*)(HostBuf + y * Stride + x * sizeof(T));
            }
        }
    }
    void GetAllData(TVector<TVector<T>> *res) const
    {
        res->resize(RowCount);
        for (yint y = 0; y < RowCount; ++y) {
            TVector<T> &dst = (*res)[y];
            dst.resize(ColumnCount);
            for (yint x = 0; x < ColumnCount; ++x) {
                dst[x] = *(T *)(HostBuf + y * Stride + x * sizeof(T));
            }
        }
    }
    yint GetXSize() const { return ColumnCount; }
    yint GetYSize() const { return RowCount; }
    TCuda2DArrayFragment<T> MakeFragment(yint xOffset, yint xSize, yint yOffset, yint ySize)
    {
        Y_ASSERT(xOffset + xSize <= GetXSize());
        Y_ASSERT(yOffset + ySize <= GetYSize());
        char *pFragmentData = ((char *)DeviceData) + yOffset * Stride + xOffset * sizeof(T);
        return TCuda2DArrayFragment<T>(this, Stride, xSize, ySize, pFragmentData);
    }
};


// host float <-> device half interop
void GetAllData(const TCuda2DArray<half> &arr, TArray2D<float> *p);
void GetAllData(const TCuda2DArray<half> &arr, TVector<TVector<float>> *p);
inline void GetAllData(const TCuda2DArray<float> &arr, TArray2D<float> *p)
{
    arr.GetAllData(p);
}
inline void GetAllData(const TCuda2DArray<float> &arr, TVector<TVector<float>> *p)
{
    arr.GetAllData(p);
}

void Put(TStream &stream, TCuda2DArray<half> *arr, const TArray2D<float> &src);
void Put(TStream &stream, TCudaVector<half> *arr, const TVector<float> &src);
}
