#include "stdafx.h"
#include "cuda_graph.cuh"


namespace NCuda
{
// copying more then cache line at once breaks cpu/gpu synchronization somehow, so we copy 32*4 = 128 bytes at once
__global__ void KernelCopyImpl(int4 *dst, int4 *src, int lenBytes)
{
    int thrOffset = threadIdx.y * WARP_SIZE + threadIdx.x;
    int len = lenBytes / sizeof(*src);
    for (int blkOffset = 0; blkOffset < len; blkOffset += WARP_SIZE * KERNEL_COPY_BLOCK) {
        int offset = blkOffset + thrOffset;
        if (offset < len) {
            dst[offset] = src[offset];
        }
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
THashMap<TString, dim3> &GetKernelBlockSize()
{
    static THashMap<TString, dim3> kernelBlockSize;
    return kernelBlockSize;
}


struct TCudaOpDependencies
{
    struct TLink
    {
        int From = 0;
        int To = 0;

        TLink() {}
        TLink(int from, int to) : From(from), To(to) {}
        bool operator==(const TLink &x) const { return From == x.From && To == x.To; }
    };
    struct TLinkHash
    {
        yint operator()(const TLink &a) const
        {
            return a.From * 0xbf1765d37f0cf121ll + a.To * 0x49260d554d12eb93ll;
        }
    };

    TVector<cudaGraphNode_t> AllNodes;
    TVector<TLink> Links;

    TCudaOpDependencies(const TVector<cudaGraphNode_t> &allNodes) : AllNodes(allNodes) {}
    void AddDep(int from, int to)
    {
        Links.push_back(TLink(from, to));
    }
    void AddDeps(cudaGraph_t graph)
    {
        if (!Links.empty()) {
            TVector<cudaGraphNode_t> fromArr;
            TVector<cudaGraphNode_t> toArr;
            for (const TLink &lnk : Links) {
                fromArr.push_back(AllNodes[lnk.From]);
                toArr.push_back(AllNodes[lnk.To]);
            }
            if (cudaGraphAddDependencies(graph, fromArr.data(), toArr.data(), YSize(fromArr)) != cudaSuccess) {
                abort();
            }
        }
    }
    void TransitiveReduction()
    {
        yint sz = YSize(AllNodes);
        THashMap<cudaGraphNode_t, yint, TPtrHash> nodeId;
        for (yint k = 0; k < sz; ++k) {
            nodeId[AllNodes[k]] = k;
        }
        TVector<TVector<int>> linksTo;
        linksTo.resize(sz);
        for (const TLink &lnk : Links) {
            Y_ASSERT(lnk.From < lnk.To);
            linksTo[lnk.To].push_back(lnk.From);
        }
        TArray2D<bool> reach;
        reach.SetSizes(sz, sz);
        reach.FillZero();
        for (yint k = 0; k < sz; ++k) {
            reach[k][k] = true;
        }
        TVector<TLink> newLinks;
        for (yint to = 0; to < sz; ++to) {
            Sort(linksTo[to].begin(), linksTo[to].end(), [](int a, int b) { return a > b; });
            for (int from : linksTo[to]) {
                if (!reach[to][from]) {
                    newLinks.push_back(TLink(from, to));
                    for (yint z = 0; z <= from; ++z) {
                        reach[to][z] |= reach[from][z];
                    }
                }
            }
        }
        Links = newLinks;
    }
};


void TGraph::MakeLinearDeps(TCudaOpDependencies *pDep)
{
    for (yint k = 1; k < YSize(OpArr); ++k) {
        pDep->AddDep(k - 1, k);
    }
}


void TGraph::MakeDeps(TCudaOpDependencies *pDep)
{
    // read after write
    THashMap<const void *, TVector<int>, TPtrHash> writeOp;
    for (yint k = 0; k < YSize(OpArr); ++k) {
        TGraphOp *op = OpArr[k].Get();
        // read should start after write is complete
        for (const void *data : op->ReadSet) {
            auto it = writeOp.find(data);
            if (it != writeOp.end()) {
                for (int writeOpIndex : it->second) {
                    pDep->AddDep(writeOpIndex, k);
                }
            }
        }
        for (const void *data : op->WriteSet) {
            writeOp[data].push_back(k);
        }
    }
    // write should start after read is complete
    writeOp.clear();
    for (yint k = YSize(OpArr) - 1; k >= 0; --k) {
        TGraphOp *op = OpArr[k].Get();
        for (const void *data : op->ReadSet) {
            auto it = writeOp.find(data);
            if (it != writeOp.end()) {
                for (int writeOpIndex : it->second) {
                    pDep->AddDep(k, writeOpIndex);
                }
            }
        }
        for (const void *data : op->WriteSet) {
            writeOp[data].push_back(k);
        }
    }
}


void TGraph::CreateExecGraph()
{
    Y_ASSERT(ExecGraph == 0);

    for (const TIntrusivePtr<TGraphOp> &op : OpArr) {
        op->CreateNode();
    }

    TVector<cudaGraphNode_t> allNodes;
    for (yint k = 0; k < YSize(OpArr); ++k) {
        allNodes.push_back(OpArr[k]->Node);
    }

    TCudaOpDependencies dep(allNodes);
    //MakeLinearDeps(&dep);
    MakeDeps(&dep);
    dep.TransitiveReduction();
    dep.AddDeps(Graph);

    //cudaGraphDebugDotPrint(Graph, "D:/g.dot", cudaGraphDebugDotFlagsVerbose);
    cudaError_t err = cudaGraphInstantiateWithFlags(&ExecGraph, Graph, 0);
    Y_VERIFY(err == cudaSuccess);
}


void TGraph::Run(TStream &stream)
{
    if (ExecGraph == 0) {
        CreateExecGraph();
    } else {
        for (TIntrusivePtr<TGraphOp> &op : OpArr) {
            op->UpdateParams(ExecGraph);
        }
    }
    cudaError_t err = cudaGraphLaunch(ExecGraph, stream);
    Y_VERIFY(err == cudaSuccess);
}

}
