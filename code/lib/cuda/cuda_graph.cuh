#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include "cuda_util.cuh"

namespace NCuda
{

enum EOpDep
{
    DEP_NONE = 0,
    DEP_READ = 1, // op reads data
    DEP_ATOMICWRITE = 2, // op writes data atomically, can have several simlultaneious such ops
    DEP_READWRITE = 3, // op modifies or rewrites data, other writes should be complete before this op
    DEP_OVERWRITE = DEP_READWRITE, // no concurrent writes are allowed

    DEP_IS_READ = 1,
    DEP_IS_WRITE = 2,
};


///////////////////////////////////////////////////////////////////////////////////////////////////
class TGraphOp : public TThrRefBase
{
    bool NeedSetParams = false;
    TVector<const void *> ReadSet;
    TVector<const void *> WriteSet;

protected:
    cudaGraph_t Graph = 0;
    cudaGraphNode_t Node = 0;

    virtual void SetParams(cudaGraphExec_t) {}
    virtual void CreateNode() = 0;
    void ParamsUpdated()
    {
        NeedSetParams = false;
    }

public:
    TGraphOp(cudaGraph_t graph) : Graph(graph) {}
    void OnParamsChange()
    {
        NeedSetParams = true;
    }
    void UpdateParams(cudaGraphExec_t execGraph)
    {
        if (NeedSetParams) {
            SetParams(execGraph);
        }
    }

    // deps
    void AddDeps(EOpDep dep, const void *p)
    {
        if (dep & DEP_IS_READ) {
            ReadSet.push_back(p);
        }
        if (dep & DEP_IS_WRITE) {
            WriteSet.push_back(p);
        }
    }
    void AddDepOverwrite(const void *p)
    {
        // we want to wait other writes to complete, can do it by adding to ReadSet
        ReadSet.push_back(p);
        WriteSet.push_back(p);
    }
    const void *MakeDepPtrFromDevicePtr(void *p)
    {
        // mark device pointer with lowest bit to avoid host&device pointers match
        char *ptr = (char *)p;
        return (ptr + 1);
    }

    friend class TGraph;
};


///////////////////////////////////////////////////////////////////////////////////////////////////
template <class T>
class TOpParameter : public TNonCopyable
{
    struct TRef
    {
        TIntrusivePtr<TGraphOp> Op;
        T *Data;
    };
    T Val = T();
    TVector<TRef> RefArr;
public:
    void Set(const T &newValue)
    {
        if (Val != newValue) {
            Val = newValue;
            for (const TRef &x : RefArr) {
                *x.Data = newValue;
                x.Op->OnParamsChange();
            }
        }
    }
    void AddRef(T *p, TGraphOp *op)
    {
        TRef x;
        x.Data = p;
        x.Op = op;
        RefArr.push_back(x);
        *p = Val;
    }
    const T &Get()
    {
        return Val;
    }
};



///////////////////////////////////////////////////////////////////////////////////////////////////
// CUDA Kernel call op
class TKernelOp : public TGraphOp
{
    enum {
        PARAM_BUF_SIZE = 256
    };
    cudaKernelNodeParams Params;
    char KParamBuf[PARAM_BUF_SIZE];
    TVector<void *> KParamList;
    yint KParamPtr = 0;

    void CreateNode() override
    {
        Params.kernelParams = KParamList.data();
        Y_ASSERT(Node == 0);
        cudaError_t err = cudaGraphAddKernelNode(&Node, Graph, 0, 0, &Params);
        if (err != cudaSuccess) {
            abort();
        }
        ParamsUpdated();
    }

    void SetParams(cudaGraphExec_t execGraph)
    {
        cudaError_t err = cudaGraphExecKernelNodeSetParams(execGraph, Node, &Params);
        if (err != cudaSuccess) {
            abort();
        }
        ParamsUpdated();
    }

    template <class T>
    void AddParam(const T &val, EOpDep dep)
    {
        Y_ASSERT((dep & DEP_IS_WRITE) == 0); // allow write to arrays only atm
        Y_VERIFY(KParamPtr + sizeof(T) <= PARAM_BUF_SIZE);
        T *pParamPlace = (T *)(KParamBuf + KParamPtr);
        KParamList.push_back(pParamPlace);
        *pParamPlace = val;
        KParamPtr += sizeof(T);
    }

    template <class T>
    void AddParam(const TCudaPOD<T> &param, EOpDep dep)
    {
        void *pDeviceData = param.GetDevicePtr();
        const void *owner = param.GetOwner();
        Y_ASSERT(owner);
        AddParam(pDeviceData, DEP_NONE);
        AddDeps(DEP_READ, owner); // should depend on writes to whole owner like clearmem
        AddDeps(dep, MakeDepPtrFromDevicePtr(pDeviceData));
    }

    template <class T>
    void AddParam(const TCuda2DArrayFragment<T> &param, EOpDep dep)
    {
        const void *owner = param.GetOwner();
        Y_ASSERT(owner);
        AddParam(param.GetDevicePtr(), DEP_NONE);
        AddDeps(dep, owner);
    }

    template <class T>
    void AddParam(const TCudaVector<T> &param, EOpDep dep)
    {
        AddParam(param.GetDevicePtr(), DEP_NONE);
        AddDeps(dep, &param);
    }

    template <class T>
    void AddParam(const TCuda2DArray<T> &param, EOpDep dep)
    {
        AddParam(param.GetDevicePtr(), DEP_NONE);
        AddDeps(dep, &param);
    }

    template <class T>
    void AddParam(TOpParameter<T> &param, EOpDep dep) // has to be non constant
    {
        T *pParamPlace = (T *)(KParamBuf + KParamPtr);
        AddParam(param.Get(), DEP_NONE);
        param.AddRef(pParamPlace, this);
        Y_ASSERT((dep & DEP_IS_WRITE) == 0); // writing to parameters is prohibited
    }

public:
    TKernelOp(cudaGraph_t graph, void *kernel) : TGraphOp(graph)
    {
        Zero(Params);
        Params.func = kernel;
        Params.blockDim = dim3(WARP_SIZE);
        Params.gridDim = dim3(1);
    }

    // Grid
    TKernelOp &Grid(int x, int y = 1, int z = 1)
    {
        Params.gridDim = dim3(x, y, z);
        return *this;
    }
    TKernelOp &Grid(TOpParameter<int> &x, int y = 1, int z = 1)
    {
        Params.gridDim = dim3(0, y, z);
        Y_ASSERT(sizeof(Params.gridDim.x) == sizeof(int));
        x.AddRef((int*)&Params.gridDim.x, this);
        return *this;
    }
    TKernelOp &Grid(int x, TOpParameter<int> &y, int z = 1)
    {
        Params.gridDim = dim3(x, 0, z);
        Y_ASSERT(sizeof(Params.gridDim.y) == sizeof(int));
        y.AddRef((int *)&Params.gridDim.y, this);
        return *this;
    }

    // Block
    TKernelOp &Block(int x, int y = 1, int z = 1)
    {
        Params.blockDim = dim3(x, y, z);
        return *this;
    }

    // pass kernel parameters
    template <typename T>
    TKernelOp &operator()(const T &param)
    {
        AddParam(param, DEP_READ);
        return *this;
    }
    template <typename T>
    TKernelOp &operator()(T &param)
    {
        AddParam(param, DEP_READ);
        return *this;
    }
    template <typename T, typename... TRest>
    TKernelOp &operator()(const T &param, TRest&&... x)
    {
        AddParam(param, DEP_READ);
        return (*this)(x...);
    }
    template <typename T, typename... TRest>
    TKernelOp &operator()(T &param, TRest&&... x)
    {
        AddParam(param, DEP_READ);
        return (*this)(x...);
    }

    // kernel target params 
    template <typename T>
    TKernelOp &Write(T *param)
    {
        AddParam(*param, DEP_READWRITE);
        return *this;
    }
    template <typename T, typename... TRest>
    TKernelOp &Write(T *param, TRest... x)
    {
        AddParam(*param, DEP_READWRITE);
        return Write(x...);
    }

    // no write-write dependencies
    template <typename T>
    TKernelOp &AtomicWrite(T *param)
    {
        AddParam(*param, DEP_ATOMICWRITE);
        return *this;
    }
    template <typename T, typename... TRest>
    TKernelOp &AtomicWrite(T *param, TRest... x)
    {
        AddParam(*param, DEP_ATOMICWRITE);
        return AtomicWrite(x...);
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
// Host op calls

template <typename T, typename TParam>
struct THostOpCallParam;

template <typename TSelf>
struct THostOpCallAddParam
{
    // const args
    template <class TParam>
    auto operator()(const TParam &param)
    {
        return THostOpCallParam<TSelf, TParam>(*static_cast<TSelf *>(this), param, DEP_READ);
    }
    template <class TParam, typename... TRest>
    auto operator()(const TParam &param, TRest&&... x)
    {
        THostOpCallParam<TSelf, TParam> res(*static_cast<TSelf *>(this), param, DEP_READ);
        return res(x...);
    }
    // ref args
    template <class TParam>
    auto operator()(TParam &param)
    {
        return THostOpCallParam<TSelf, TParam>(*static_cast<TSelf *>(this), param, DEP_READ);
    }
    template <class TParam, typename... TRest>
    auto operator()(TParam &param, TRest&&... x)
    {
        THostOpCallParam<TSelf, TParam> res(*static_cast<TSelf *>(this), param, DEP_READ);
        return res(x...);
    }

    // write args
    template <typename TParam>
    auto Write(TParam *param)
    {
        return THostOpCallParam<TSelf, TParam>(*static_cast<TSelf *>(this), *param, DEP_READWRITE);
    }
    template <typename TParam, typename... TRest>
    auto Write(TParam *param, TRest&&... x)
    {
        THostOpCallParam<TSelf, TParam> res(*static_cast<TSelf *>(this), *param, DEP_READWRITE);
        return res.Write(x...);
    }

    template <typename TParam>
    auto AtomicWrite(TParam *param)
    {
        return THostOpCallParam<TSelf, TParam>(*static_cast<TSelf *>(this), *param, DEP_ATOMICWRITE);
    }

    template <typename TParam, typename... TRest>
    auto AtomicWrite(TParam *param, TRest&&... x)
    {
        THostOpCallParam<TSelf, TParam> res(*static_cast<TSelf *>(this), *param, DEP_ATOMICWRITE);
        return res.AtomicWrite(x...);
    }
};


template <typename TFunc, class TParam>
struct THostOpCallParam : public THostOpCallAddParam<THostOpCallParam<TFunc, TParam>>
{
    TFunc Func;
    TParam Param;

    THostOpCallParam(TFunc f, const TParam &x, EOpDep dep) : Func(f), Param(x)
    {
        Y_ASSERT((dep & DEP_IS_WRITE) == 0); // allow write to arrays only atm
    }

    void BindToGraphOp(TGraphOp *p)
    {
        Func.BindToGraphOp(p);
    }

    template <typename... CallArgs>
    void Call(CallArgs&&... args)
    {
        Func.Call(Param, args...);
    }
};


template <typename TFunc, class T>
struct THostOpCallParam<TFunc, TOpParameter<T>> : public THostOpCallAddParam<THostOpCallParam<TFunc, TOpParameter<T>>>
{
    TFunc Func;
    T ParamVal;
    TOpParameter<T> &Param;

    THostOpCallParam(TFunc f, TOpParameter<T> &param, EOpDep dep) : Func(f), Param(param)
    {
        Y_ASSERT((dep & DEP_IS_WRITE) == 0); // allow write to arrays only atm
    }

    void BindToGraphOp(TGraphOp *p)
    {
        Param.AddRef(&ParamVal, p);
        Func.BindToGraphOp(p);
    }

    template <typename... CallArgs>
    void Call(CallArgs&&... args)
    {
        Func.Call(ParamVal, args...);
    }
};


template <typename TFunc, class T>
struct THostOpCallParam<TFunc, TCudaPOD<T>> : public THostOpCallAddParam<THostOpCallParam<TFunc, TCudaPOD<T>>>
{
    // not implemented yet
};

template <typename TFunc, class T>
struct THostOpCallParam<TFunc, TCuda2DArrayFragment<T>> : public THostOpCallAddParam<THostOpCallParam<TFunc, TCuda2DArrayFragment<T>>>
{
    // not implemented yet
};

template <typename TFunc, class T>
struct THostOpCallParam<TFunc, TCudaVector<T>> : public THostOpCallAddParam<THostOpCallParam<TFunc, TCudaVector<T>>>
{
    TFunc Func;
    const TCudaVector<T> *ArgPtr;
    T *Param;
    EOpDep Dep;

    THostOpCallParam(TFunc f, const TCudaVector<T> &x, EOpDep dep) : Func(f), ArgPtr(&x), Param(x.GetHostPtr()), Dep(dep) {}

    void BindToGraphOp(TGraphOp *p)
    {
        p->AddDeps(Dep, ArgPtr);
        Func.BindToGraphOp(p);
    }

    template <typename... CallArgs>
    void Call(CallArgs&&... args)
    {
        Func.Call(Param, args...);
    }
};


template <typename TFunc, class T>
struct THostOpCallParam<TFunc, TCuda2DArray<T>> : public THostOpCallAddParam<THostOpCallParam<TFunc, TCuda2DArray<T>>>
{
    TFunc Func;
    const TCuda2DArray<T> *ArgPtr;
    THost2DPtr<T> Param;
    EOpDep Dep;

    THostOpCallParam(TFunc f, const TCuda2DArray<T> &x, EOpDep dep) : Func(f), ArgPtr(&x), Param(x.GetHostPtr()), Dep(dep) {}

    void BindToGraphOp(TGraphOp *p)
    {
        p->AddDeps(Dep, ArgPtr);
        Func.BindToGraphOp(p);
    }

    template <typename... CallArgs>
    void Call(CallArgs&&... args)
    {
        Func.Call(Param, args...);
    }
};


template <typename TFunc>
struct THostOpCall : public THostOpCallAddParam<THostOpCall<TFunc>>
{
    TFunc Func;

    THostOpCall(TFunc f) : Func(f) {}

    void BindToGraphOp(TGraphOp *p)
    {
        // TFunc is an actual function
    }

    template <typename... CallArgs>
    void Call(CallArgs&&... args)
    {
        Func(args...);
    }
};

template <typename T>
inline THostOpCall<T> HostCallImplementation(T f)
{
    return THostOpCall<T>(f);
}


class THostOp : public TGraphOp
{
    struct IHoldOpCallback : public TThrRefBase
    {
        virtual void Call() = 0;
        virtual void BindToGraphOp(TGraphOp *) = 0;
    };

    template <class T>
    struct THoldOpCallback : IHoldOpCallback
    {
        T Lambda;
        THoldOpCallback(T arg) : Lambda(arg) {}
        void Call() override { Lambda.Call(); }
        void BindToGraphOp(TGraphOp *p) override { Lambda.BindToGraphOp(p); }
    };

    cudaHostNodeParams Params;
    TIntrusivePtr<IHoldOpCallback> ArgHolder;

    void CreateNode() override
    {
        if (cudaGraphAddHostNode(&Node, Graph, 0, 0, &Params) != cudaSuccess) {
            abort();
        }
    }

    static void CUDART_CB HostCall(void *userData)
    {
        ((IHoldOpCallback *)userData)->Call();
    }

public:
    template <class T>
    THostOp(cudaGraph_t graph, T lambda) : TGraphOp(graph)
    {
        Params.fn = HostCall;
        ArgHolder = new THoldOpCallback<T>(lambda);
        Params.userData = ArgHolder.Get();
        ArgHolder->BindToGraphOp(this);
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
class TMemsetOp : public TGraphOp
{
    cudaMemsetParams Params;
    int RowSize = 0;
    int RowCount = 0; // potential type mismatch with Params.height, so keep it separated
    int MaxRowCount = 0;

    template <class T>
    void Init(T &arr)
    {
        TMemoryBlob blob = arr.GetDeviceMem();
        RowSize = blob.Stride;
        RowCount = blob.RowCount;
        MaxRowCount = blob.RowCount;
        Zero(Params);
        Params.dst = blob.Ptr;
        Params.elementSize = 1;
        Params.height = 1;
        Params.pitch = 0;
        Params.value = 0;
        Params.width = blob.Stride * RowCount;
        // deps
        AddDepOverwrite(&arr);
    }

    void CreateNode() override
    {
        Y_ASSERT(RowCount <= MaxRowCount);
        Params.width = RowSize * RowCount;
        Y_ASSERT(Node == 0);
        cudaError_t err = cudaGraphAddMemsetNode(&Node, Graph, 0, 0, &Params);
        Y_VERIFY(err == cudaSuccess);
        ParamsUpdated();
    }

    void SetParams(cudaGraphExec_t execGraph)
    {
        Y_ASSERT(RowCount <= MaxRowCount);
        Params.width = RowSize * RowCount;
        cudaError_t err = cudaGraphExecMemsetNodeSetParams(execGraph, Node, &Params);
        Y_VERIFY(err == cudaSuccess);
        ParamsUpdated();
    }

public:
    template <class T>
    TMemsetOp(cudaGraph_t graph, T &arr) : TGraphOp(graph)
    {
        Init(arr);
    }
    template <class T>
    TMemsetOp(cudaGraph_t graph, T &arr, TOpParameter<int> &ySize) : TGraphOp(graph)
    {
        Init(arr);
        ySize.AddRef(&RowCount, this);
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
class TMemcpyOp : public TGraphOp
{
    void *Dst = 0;
    void *Src = 0;
    size_t Size = 0;
    cudaMemcpyKind OpType;

    void CreateNode() override
    {
        if (cudaGraphAddMemcpyNode1D(&Node, Graph, 0, 0, Dst, Src, Size, OpType) != cudaSuccess) {
            abort();
        }
    }
public:
    template <class TDst, class TSrc>
    TMemcpyOp(cudaGraph_t graph, TDst *dst, const TSrc &src, EMemType srcMemType, EMemType dstMemType, cudaMemcpyKind opType) : TGraphOp(graph), OpType(opType)
    {
        TMemoryBlob srcBlob = src.GetMem(srcMemType);
        TMemoryBlob dstBlob = dst->GetMem(dstMemType);
        Y_VERIFY(srcBlob.IsSameSize(dstBlob));
        Dst = dstBlob.Ptr;
        Src = srcBlob.Ptr;
        Size = dstBlob.GetSize();
        // deps
        AddDeps(DEP_READ, &src);
        AddDepOverwrite(dst);
    }
};


const int KERNEL_COPY_BLOCK = 32;
__global__ void KernelCopyImpl(int4 *dst, int4 *src, int len);


///////////////////////////////////////////////////////////////////////////////////////////////////
// default kernel block size
THashMap<TString, dim3> &GetKernelBlockSize();
#define KERNEL_BLOCK_SIZE(a, ...) namespace { struct TKernelBlock##a { TKernelBlock##a() {\
    Y_ASSERT(GetKernelBlockSize().find(KERNEL_UNIT #a) == GetKernelBlockSize().end());\
    GetKernelBlockSize()[KERNEL_UNIT #a] = dim3(__VA_ARGS__);\
} } setKernelBlockSize##a; }


///////////////////////////////////////////////////////////////////////////////////////////////////
// CUDA Graph
struct TCudaOpDependencies;
class TGraph : public TThrRefBase
{
    cudaGraph_t Graph;
    cudaGraphExec_t ExecGraph = 0;
    TVector<TIntrusivePtr<TGraphOp>> OpArr;

    void MakeLinearDeps(TCudaOpDependencies *pDep);
    void MakeDeps(TCudaOpDependencies *pDep);
    void CreateExecGraph();
public:
    TGraph()
    {
        cudaGraphCreate(&Graph, 0);
    }
    ~TGraph()
    {
        if (ExecGraph != 0) {
            cudaGraphExecDestroy(ExecGraph);
        }
        cudaGraphDestroy(Graph);
    }
    TKernelOp &CudaCallImplementation(const TString &kernelUnit, const TString &kernelName, void *kernel)
    {
        TKernelOp *p = new TKernelOp(Graph, kernel);
        TString kernelFuncName = kernelName.substr(0, kernelName.find('<'));
        const THashMap<TString, dim3> &kernelBlockSize = GetKernelBlockSize();
        auto it = kernelBlockSize.find(kernelUnit + kernelFuncName);
        if (it != kernelBlockSize.end()) {
            p->Block(it->second.x, it->second.y, it->second.z);
        }
        OpArr.push_back(p);
        return *p;
    }
    template <class T>
    void operator+=(T lambda)
    {
        OpArr.push_back(new THostOp(Graph, lambda));
    }

    // memory ops
    template <class T>
    void ClearMem(T &arr)
    {
        OpArr.push_back(new TMemsetOp(Graph, arr));
    }
    template <class T>
    void ClearMem(T &arr, TOpParameter<int> &ySize)
    {
        OpArr.push_back(new TMemsetOp(Graph, arr, ySize));
    }
    template <class TDst, class TSrc>
    void CopyToHost(TDst *dst, const TSrc &src)
    {
        OpArr.push_back(new TMemcpyOp(Graph, dst, src, MT_DEVICE, MT_HOST, cudaMemcpyDeviceToHost));
    }
    template <class TDst, class TSrc>
    void CopyToDevice(TDst *dst, const TSrc &src)
    {
        OpArr.push_back(new TMemcpyOp(Graph, dst, src, MT_HOST, MT_DEVICE, cudaMemcpyHostToDevice));
    }

    // use kernel to copy arrays (avoid WDDM induced delays on Windows)
    template <class TDst, class TSrc>
    void KernelCopy(TDst *dst, const TSrc &src, yint rowCount)
    {
        TMemoryBlob srcBlob = src.GetMem(MT_DEVICE);
        TMemoryBlob dstBlob = dst->GetMem(MT_DEVICE);
        Y_VERIFY(srcBlob.Stride == dstBlob.Stride && srcBlob.RowCount >= rowCount && dstBlob.RowCount >= rowCount);
        TIntrusivePtr<TKernelOp> p = new TKernelOp(Graph, (void*)KernelCopyImpl);
        (*p)(dstBlob.Ptr, srcBlob.Ptr, srcBlob.Stride * rowCount);
        (*p).Block(WARP_SIZE, KERNEL_COPY_BLOCK);
        // deps
        p->AddDeps(DEP_READ, &src);
        p->AddDepOverwrite(dst);
        OpArr.push_back(p.Get());
    }
    template <class TDst, class TSrc>
    void KernelCopy(TDst *dst, const TSrc &src)
    {
        TMemoryBlob srcBlob = src.GetMem(MT_DEVICE);
        TMemoryBlob dstBlob = dst->GetMem(MT_DEVICE);
        Y_VERIFY(srcBlob.IsSameSize(dstBlob));
        KernelCopy(dst, src, srcBlob.RowCount);
    }

    // run
    void Run(TStream &stream);
};

#define CudaCall(c, kernel, ...) c->CudaCallImplementation(KERNEL_UNIT, #kernel, (void*)kernel, ##__VA_ARGS__)
#define HostCall(c, kernel, ...) *c += HostCallImplementation(kernel, ##__VA_ARGS__)

}
