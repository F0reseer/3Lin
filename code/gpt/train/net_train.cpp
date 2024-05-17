#include "stdafx.h"
#include "net_train.h"
#include "network.h"
#include "train.h"
#include <gpt/model/gpt_cuda.cuh>
#include <gpt/model/par_matrix.h>
#include <gpt/model/par_delta_accum.h>
#include <gpt/att/sliding_window.h>
#include <lib/hp_timer/hp_timer.h>
#include <lib/net/ip_address.h>
#include <typeinfo>
#include <emmintrin.h>


using NCuda::TModelMatrix;
using NCuda::TModelMatrixDelta;
using NCuda::TModelMatrixBitTail;

namespace NNetTrain
{

static TGuid NetTrainToken(0xbadf00d, 0x31337, 0xceedb0c0, 0x31415926);


///////////////////////////////////////////////////////////////////////////////////////////////////
struct TNetTrainContext;
struct TCommandPacket : public TThrRefBase
{
    virtual void Exec(TNetTrainContext *pCtx) {}
    virtual int operator&(IBinSaver &f) { return 0; }
    virtual yint GetP2PIteration() { return -1; }
};


typedef TCommandPacket *(*CreateObject)();

typedef ui32 TTypeId;
static THashMap<const std::type_info *, TTypeId, TPtrHash> Type2TypeId;
static THashMap<TTypeId, CreateObject> TypeId2Constructor;

#define REGISTER_PACKET(obj, id)\
    static TCommandPacket* Create##id() { return new obj(); }\
    struct TRegisterPacket##id { TRegisterPacket##id() {\
        Y_ASSERT(TypeId2Constructor.find(id) == TypeId2Constructor.end());\
        Type2TypeId[&typeid(obj)] = id;\
        TypeId2Constructor[id] = Create##id;\
    } } registerPacket##id;


static void SerializeCommand(TIntrusivePtr<TCommandPacket> cmd, TVector<ui8> *p)
{
    TMemStream mem;
    TCommandPacket *cmdPkt = cmd.Get();
    TTypeId objTypeId = Type2TypeId[&typeid(*cmdPkt)];
    mem.Write(&objTypeId, sizeof(objTypeId));
    {
        IBinSaver bs(mem, false);
        bs.Add(cmd.Get());
    }
    mem.ExtractData(p);
}

static TIntrusivePtr<TCommandPacket> DeserializeCommand(TVector<ui8> *p)
{
    TMemStream mem(p);
    TTypeId objTypeId = 0;
    mem.Read(&objTypeId, sizeof(objTypeId));
    TIntrusivePtr<TCommandPacket> cmd = TypeId2Constructor[objTypeId]();
    IBinSaver bs(mem, true);
    bs.Add(cmd.Get());
    return cmd;
}


///////////////////////////////////////////////////////////////////////////////////////////////////
static TIntrusivePtr<TCommandPacket> RecvCommand(TIntrusivePtr<INetwork> net)
{
    TIntrusivePtr<TNetPacket> p = net->Recv();
    if (p.Get()) {
        return DeserializeCommand(&p->Data);
    } else {
        return 0;
    }
}


static void SendCommand(TIntrusivePtr<INetwork> masterNet, TNetworkAddr addr, TIntrusivePtr<TCommandPacket> cmd)
{
    TIntrusivePtr<TNetPacket> pkt = new TNetPacket;
    pkt->Addr = addr;
    SerializeCommand(cmd, &pkt->Data);
    masterNet->Send(pkt);
}


template <class TRes>
static void WaitCommandResult(TIntrusivePtr<INetwork> net, TNetworkAddr addr, TRes *pRes)
{
    for (;;) {
        TIntrusivePtr<TNetPacket> p = net->Recv();
        if (p.Get()) {
            Y_VERIFY(p->Addr == addr);
            SerializeMem(true, &p->Data, *pRes);
            return;
        }
    }

}

enum ECommandResult
{
    CMD_OK,
};

static void SendCommandResult(TIntrusivePtr<INetwork> masterNet, ECommandResult res)
{
    masterNet->SendData(0, res);
}


template <class TRet>
void CollectCommandResults(TIntrusivePtr<INetwork> masterNet, yint workerCount, TVector<TRet> *pResArr)
{
    pResArr->resize(workerCount);
    yint confirmCount = 0;
    while (confirmCount < workerCount) {
        TIntrusivePtr<TNetPacket> p = masterNet->Recv();
        if (p.Get()) {
            SerializeMem(true, &p->Data, (*pResArr)[p->Addr - 1]);
            ++confirmCount;
        }
    }
}


template <class TRet>
void BroadcastCommand(TIntrusivePtr<INetwork> masterNet, yint workerCount, TIntrusivePtr<TCommandPacket> cmd, TVector<TRet> *pResArr)
{
    TVector<ui8> pktData;
    SerializeCommand(cmd, &pktData);
    //
    for (yint workerAddr = 1; workerAddr <= workerCount; ++workerAddr) {
        TIntrusivePtr<TNetPacket> pkt = new TNetPacket;
        pkt->Addr = workerAddr;
        pkt->Data = pktData; // change TNetPacket to reuse same data
        masterNet->Send(pkt);
    }
    CollectCommandResults(masterNet, workerCount, pResArr);
}


///////////////////////////////////////////////////////////////////////////////////////////////////
//
class TMMNetDeltaReduceGen;
struct TNetTrainContext
{
    TIntrusivePtr<INetwork> MasterNet;
    TIntrusivePtr<INetwork> P2PNet;
    TIntrusivePtr<IModel> Model;
    TIntrusivePtr<IComputeContext> Ctx;
    TIntrusivePtr<TMMNetDeltaReduceGen> NetDeltaReduce;
    TVector<ui8> ModelSnapshot;
    TThread P2PThread;
    yint WorkerCount = 0;
};


///////////////////////////////////////////////////////////////////////////////////////////////////
class TCalcModelError : public TCommandPacket
{
    TTrainConfig TrainConfig;
    TVector<TFragment> FragArr;
    SAVELOAD_OVERRIDE(TrainConfig, FragArr);
public:
    TCalcModelError() {}
    TCalcModelError(const TTrainConfig &tc, const TVector<TFragment> &fragArr) : TrainConfig(tc), FragArr(fragArr) {}
    void Exec(TNetTrainContext *p) override
    {
        double res = CalcModelErr(FragArr, TrainConfig.Window, p->Ctx.Get());
        p->MasterNet->SendData(0, res);
    }
};
REGISTER_PACKET(TCalcModelError, 1);

static double DistributedCalcModelErr(const TTrainConfig &tc, TIntrusivePtr<INetwork> net, yint workerCount, const TVector<TVector<TFragment>> &batches)
{
    if (batches.empty()) {
        return 0;
    }
    yint batchCount = YSize(batches);
    for (yint b = 0; b < batchCount; ++b) {
        TIntrusivePtr<TNetPacket> pkt = new TNetPacket;
        pkt->Addr = 1 + (b % workerCount);
        SerializeCommand(new TCalcModelError(tc, batches[b]), &pkt->Data);
        net->Send(pkt);
    }
    // collect results
    double sum = 0;
    yint confirmCount = 0;
    while (confirmCount < batchCount) {
        TIntrusivePtr<TNetPacket> p = net->Recv();
        if (p.Get()) {
            double score = 0;
            SerializeMem(true, &p->Data, score);
            sum += score;
            ++confirmCount;
        }
    }
    return sum / batchCount;
}


///////////////////////////////////////////////////////////////////////////////////////////////////
class TDeltaMatrix : public TCommandPacket
{
    yint P2PIteration = 0;
    yint MatrixId = 0;
    yint SumLevel = 0;
    TModelMatrixDelta BitDelta;
    SAVELOAD_OVERRIDE(P2PIteration, MatrixId, SumLevel, BitDelta);
public:
    TDeltaMatrix() {}
    TDeltaMatrix(yint p2pIteration, yint matrixId, yint sumLevel, const TModelMatrixDelta &bitDelta)
        : P2PIteration(p2pIteration), MatrixId(matrixId), SumLevel(sumLevel), BitDelta(bitDelta)
    {
    }
    void Exec(TNetTrainContext *p) override;
    yint GetP2PIteration() override
    {
        return P2PIteration;
    }
};
REGISTER_PACKET(TDeltaMatrix, 2);


// OnDelta() and AddRemoteDelta() are called from different threads
class TMMNetDeltaReduce : public IMMDeltaHook
{
    const static ui64 LOCAL_DATA = 0x8000; // debug is easier if we know which data is ready

    enum {
        DELTA_READY = 0,
        DELTA_COMPUTE = 1,
    };

    struct TReduceLevel : public TThrRefBase
    {
        std::atomic<yint> ReadyCount;
        TModelMatrixDelta RemoteSum;
        TModelMatrixDelta LocalSum;
        TModelMatrixBitTail Tail;

        TReduceLevel(yint xSize, yint ySize, bool hasRowDisp)
        {
            Tail.Init(xSize, ySize, hasRowDisp);
            ReadyCount = 0;
        }
    };

    TIntrusivePtr<IMMDeltaHook> DeltaHook;
    yint P2PIteration = 0;
    yint MatrixId = 0;
    TIntrusivePtr<INetwork> P2PNet;
    yint WorkerCount = 0;
    TIntrusivePtr<TModelMatrix> ModelMatrix;
    TVector<TIntrusivePtr<TReduceLevel>> ReduceArr;
    TArray2D<float> DeltaTail;
    TModelMatrixDelta PrevIterDelta;
    bool CanUseStaleGradient = false;
    volatile int StaleDeltaState;

    static void SumDeltas(TReduceLevel *pLevel, TModelMatrixDelta *pRes)
    {
        TModelMatrix::SumBitDelta(pLevel->LocalSum, pLevel->RemoteSum, &pLevel->Tail, pRes);
        Y_VERIFY(pLevel->ReadyCount.load() == LOCAL_DATA + 1);
        pLevel->ReadyCount = 0;
    }

    void AddDeltaCount(yint level, ui64 c)
    {
        TReduceLevel &rl = *ReduceArr[level];
        if (rl.ReadyCount.fetch_add(c) + c == LOCAL_DATA + 1) {
            bool isFinalLevel = (level + 1 == YSize(ReduceArr));
            TModelMatrixDelta *finalDelta = CanUseStaleGradient ? &PrevIterDelta : &ModelMatrix->GetBitDelta();
            TModelMatrixDelta *resSum = isFinalLevel ? finalDelta : &ReduceArr[level + 1]->LocalSum;
            SumDeltas(&rl, resSum);
            if (isFinalLevel) {
                if (CanUseStaleGradient) {
                    Y_VERIFY(StaleDeltaState == DELTA_COMPUTE);
                    StaleDeltaState = DELTA_READY;
                } else {
                    Y_ASSERT(finalDelta == &ModelMatrix->GetBitDelta());
                    ModelMatrix->SetOp(TModelMatrix::OP_ADD_BIT_DELTA);
                }
            } else {
                TNetworkAddr peerAddr = P2PNet->GetMyAddr() ^ (1ull << (level + 1));
                SendCommand(P2PNet, peerAddr, new TDeltaMatrix(P2PIteration, MatrixId, level + 1, *resSum));
                AddDeltaCount(level + 1, LOCAL_DATA);
            }
        }
    }

    void OnDelta() override
    {
        DeltaHook->OnDelta();
        if (ModelMatrix->GetOp() == TModelMatrix::OP_NONE) {
            return;
        }
        Y_VERIFY(ModelMatrix->GetOp() == TModelMatrix::OP_ADD_DELTA);
        if (WorkerCount == 1) {
            return;
        }
        //DebugPrintf("On delta, matrix %g\n", MatrixId * 1.);
        ModelMatrix->SetOp(TModelMatrix::OP_WAIT);

        TModelMatrixDelta &localSum = ReduceArr[0]->LocalSum;
        ModelMatrix->CompressDelta(&localSum, &DeltaTail);

        if (CanUseStaleGradient) {
            Y_VERIFY(StaleDeltaState == DELTA_READY);
            StaleDeltaState = DELTA_COMPUTE;
            if (!PrevIterDelta.IsEmpty()) {
                ModelMatrix->GetBitDelta().Swap(&PrevIterDelta);
                ModelMatrix->SetOp(TModelMatrix::OP_ADD_BIT_DELTA);
            } else {
                ModelMatrix->SetOp(TModelMatrix::OP_NONE);
            }
        }

        TNetworkAddr peerAddr = P2PNet->GetMyAddr() ^ 1;
        SendCommand(P2PNet, peerAddr, new TDeltaMatrix(P2PIteration, MatrixId, 0, localSum));
        AddDeltaCount(0, LOCAL_DATA);
    }

public:
    TMMNetDeltaReduce(yint matrixId, TIntrusivePtr<TModelMatrix> p, IMMDeltaHook *deltaHook, TIntrusivePtr<INetwork> p2pNet, yint workerCount)
        : DeltaHook(deltaHook), MatrixId(matrixId), P2PNet(p2pNet), WorkerCount(workerCount), ModelMatrix(p)
    {
        Y_VERIFY((workerCount & (workerCount - 1)) == 0);
        yint levelCount = 0;
        while ((1ll << levelCount) < workerCount) {
            ++levelCount;
        }
        yint xSize = p->GetXSize();
        yint ySize = p->GetYSize();
        ReduceArr.resize(levelCount);
        for (yint k = 0; k < levelCount; ++k) {
            ReduceArr[k] = new TReduceLevel(xSize, ySize, p->HasRowDisp());
        }
        DeltaTail.SetSizes(xSize, ySize);
        DeltaTail.FillZero();
        CanUseStaleGradient = ModelMatrix->CanUseStaleGradient();
        StaleDeltaState = DELTA_READY;
    }

    void AddRemoteDelta(yint deltaP2PIteration, yint sumLevel, TModelMatrixDelta *pBitDelta)
    {
        if (deltaP2PIteration != P2PIteration) {
            DebugPrintf("delta iteration mismatch, remote %g, current %g\n", deltaP2PIteration * 1., P2PIteration * 1.);
        }
        pBitDelta->Swap(&ReduceArr[sumLevel]->RemoteSum);
        AddDeltaCount(sumLevel, 1); 
    }

    void SetP2PIteration(yint iter)
    {
        // delay can be postponed up to OnDelta() ? then we would need to queue remote deltas within AddRemoteDelta()
        while (StaleDeltaState != DELTA_READY) {
            _mm_pause();
        }
        for (TIntrusivePtr<TReduceLevel> &lvl : ReduceArr) {
            Y_VERIFY(lvl->ReadyCount == 0);
        }
        P2PIteration = iter;
    }
};


class TMMNetDeltaReduceGen : public IMMDeltaHookGen
{
    TIntrusivePtr<TMMDeltaAccumulateGen> DeltaAccum;
    TIntrusivePtr<INetwork> P2PNet;
    yint WorkerCount = 0;
    TVector<TIntrusivePtr<TMMNetDeltaReduce>> Arr;
    volatile yint CurrentP2PIteration = 0;

    IMMDeltaHook *CreateDeltaHook(yint idx, TIntrusivePtr<TModelMatrix> p) override
    {
        IMMDeltaHook *deltaHook = DeltaAccum->CreateDeltaHook(idx, p);
        TMMNetDeltaReduce *res = new TMMNetDeltaReduce(idx, p, deltaHook, P2PNet, WorkerCount);
        if (YSize(Arr) <= idx) {
            Arr.resize(idx + 1);
        }
        Arr[idx] = res;
        return res;
    }

    void OnIterationStart() override
    {
        DeltaAccum->OnIterationStart();
        yint newIter = CurrentP2PIteration + 1;
        for (yint k = 0; k < YSize(Arr); ++k) {
            Arr[k]->SetP2PIteration(newIter);
        }
        CurrentP2PIteration = newIter; // after setting iteration for delta hooks
    }

public:
    TMMNetDeltaReduceGen(TIntrusivePtr<INetwork> p2pNet, yint workerCount) : P2PNet(p2pNet), WorkerCount(workerCount)
    {
        DeltaAccum = new TMMDeltaAccumulateGen();
    }

    void AddRemoteDelta(yint deltaP2PIteration, yint matrixId, yint sumLevel, TModelMatrixDelta *pBitDelta)
    {
        Arr[matrixId]->AddRemoteDelta(deltaP2PIteration, sumLevel, pBitDelta);
    }

    yint GetCurrentP2PIteration() const
    {
        return CurrentP2PIteration;
    }

    void SetAddToModel(EAddToModel addToModel)
    {
        DeltaAccum->SetAddToModel(addToModel);
    }
};

void TDeltaMatrix::Exec(TNetTrainContext *p)
{
    p->NetDeltaReduce->AddRemoteDelta(P2PIteration, MatrixId, SumLevel, &BitDelta);
}


///////////////////////////////////////////////////////////////////////////////////////////////////
class TBackprop : public TCommandPacket
{
    yint Iter = 0;
    TTrainConfig TrainConfig;
    float Step = 0;
    EAddToModel AddToModel = GRADIENT_APPLY;
    TVector<TVector<TFragment>> FragArr;
    SAVELOAD_OVERRIDE(Iter, TrainConfig, Step, AddToModel, FragArr);
public:
    TBackprop() {}
    TBackprop(yint iter, const TTrainConfig &tc, float step, EAddToModel addToModel, const TVector<TVector<TFragment>> &fragArr)
        : Iter(iter), TrainConfig(tc), Step(step), AddToModel(addToModel), FragArr(fragArr)
    {
    }
    void Exec(TNetTrainContext *p) override
    {
        p->NetDeltaReduce->SetAddToModel(AddToModel);
        yint rngSeed = Iter * p->WorkerCount + p->P2PNet->GetMyAddr();
        TXRng iterRng(rngSeed);
        const TTrainConfig &tc = TrainConfig;
        for (yint deviceId = 0; deviceId < YSize(FragArr); ++deviceId) {
            MakeTrain(iterRng, FragArr[deviceId], tc.TokenDrop, tc.ChannelDrop, tc.Window, p->Ctx.Get(), deviceId);
        }
        p->Ctx->Backprop(Step);
        SendCommandResult(p->MasterNet, CMD_OK);
    }
};
REGISTER_PACKET(TBackprop, 3);


///////////////////////////////////////////////////////////////////////////////////////////////////
// P2P network
class TGetP2PPort : public TCommandPacket
{
    void Exec(TNetTrainContext *p) override
    {
        yint port = p->P2PNet->GetPort();
        p->MasterNet->SendData(0, port);
    }
};
REGISTER_PACKET(TGetP2PPort, 4);


static void P2PWorkerThread(void *p)
{
    TNetTrainContext &ctx = *(TNetTrainContext *)p;
    TVector<TIntrusivePtr<TCommandPacket>> delayedArr;
    yint p2pIteration = -1;
    for (;;) {
        yint newP2PIteration = ctx.NetDeltaReduce->GetCurrentP2PIteration();
        if (p2pIteration != newP2PIteration) {
            p2pIteration = newP2PIteration;
            yint dst = 0;
            for (yint k = 0; k < YSize(delayedArr); ++k) {
                TIntrusivePtr<TCommandPacket> cmd = delayedArr[k];
                if (cmd->GetP2PIteration() == p2pIteration) {
                    cmd->Exec(&ctx);
                } else {
                    Y_VERIFY(cmd->GetP2PIteration() > p2pIteration);
                    delayedArr[dst++] = cmd;
                }
            }
            delayedArr.resize(dst);
        }
        // recv command
        TIntrusivePtr<TCommandPacket> cmd = RecvCommand(ctx.P2PNet);
        if (cmd.Get()) {
            //DebugPrintf("P2P got command %s\n", typeid(*cmd.Get()).name());
            if (cmd->GetP2PIteration() != p2pIteration) {
                // postpone commands from future iterations
                Y_VERIFY(cmd->GetP2PIteration() > p2pIteration);
                delayedArr.push_back(cmd);
            } else {
                cmd->Exec(&ctx);
            }
        }
    }
}


class TP2PConnect : public TCommandPacket
{
    TNetworkAddr Addr = 0;
    TVector<TString> PeerList;
    SAVELOAD_OVERRIDE(Addr, PeerList);

    void Exec(TNetTrainContext *p) override
    {
        yint workerCount = YSize(PeerList);
        p->NetDeltaReduce = new TMMNetDeltaReduceGen(p->P2PNet, workerCount);
        ConnectP2P(p->P2PNet, Addr, PeerList);
        p->P2PThread.Create(P2PWorkerThread, p);
    }
public:
    TP2PConnect() {}
    TP2PConnect(TNetworkAddr addr, const TVector<TString> &peerList) : Addr(addr), PeerList(peerList)
    {
    }
};
REGISTER_PACKET(TP2PConnect, 5);


class TP2PWaitComplete : public TCommandPacket
{
    yint PeerCount = 0;
    SAVELOAD_OVERRIDE(PeerCount);

    void Exec(TNetTrainContext *p) override
    {
        while (p->P2PNet->GetPeerCount() < PeerCount) {
            SchedYield();
        }
        p->P2PNet->StopAcceptingConnections();
        DebugPrintf("p2p network complete\n");
    }
public:
    TP2PWaitComplete() {}
    TP2PWaitComplete(yint peerCount) : PeerCount(peerCount) {}
};
REGISTER_PACKET(TP2PWaitComplete, 6);


static void CreateP2PNetwork(TIntrusivePtr<INetwork> net, const TVector<TString> &peerList)
{
    yint workerCount = YSize(peerList);

    TVector<yint> p2pPortArr;
    BroadcastCommand(net, workerCount, new TGetP2PPort(), &p2pPortArr);

    TVector<TString> p2pPeers = peerList;
    for (yint k = 0; k < workerCount; ++k) {
        NNet::ReplacePort(&p2pPeers[k], p2pPortArr[k]);
    }

    DebugPrintf("p2p connect\n");
    for (yint k = 0; k < workerCount; ++k) {
        SendCommand(net, k + 1, new TP2PConnect(k, p2pPeers));
    }
    for (yint k = 0; k < workerCount; ++k) {
        SendCommand(net, k + 1, new TP2PWaitComplete(workerCount - 1));
    }
    DebugPrintf("p2p network complete\n");
}


///////////////////////////////////////////////////////////////////////////////////////////////////
class TCreateModel : public TCommandPacket
{
    yint DeviceCount = 0;
    yint WorkerCount = 0;
    TModelParams Params;
    yint GpuBufferLen;
    SAVELOAD_OVERRIDE(DeviceCount, WorkerCount, Params, GpuBufferLen);
public:
    TCreateModel() {}
    TCreateModel(yint deviceCount, yint workerCount, const TModelParams &params, yint gpuBufferLen)
        : DeviceCount(deviceCount), WorkerCount(workerCount), Params(params), GpuBufferLen(gpuBufferLen)
    {
    }
    void Exec(TNetTrainContext *p) override
    {
        p->Model = CreateModel(DeviceCount, Params, p->NetDeltaReduce.Get());
        p->Ctx = NCUDA_GPT::CreateContext(p->Model, GpuBufferLen);
        p->WorkerCount = WorkerCount;
        SendCommandResult(p->MasterNet, CMD_OK);
    }
};
REGISTER_PACKET(TCreateModel, 7);


///////////////////////////////////////////////////////////////////////////////////////////////////
class TMakeParamsSnapshot : public TCommandPacket
{
public:
    void Exec(TNetTrainContext *p) override
    {
        TModelParams params;
        p->Ctx->GetParams(&params);
        SerializeMem(false, &p->ModelSnapshot, params);
        yint sz = YSize(p->ModelSnapshot);
        p->MasterNet->SendData(0, sz);
    }
};
REGISTER_PACKET(TMakeParamsSnapshot, 8);


class TGetParamsSnapshotFragment : public TCommandPacket
{
    yint Offset = 0;
    yint Size = 0;
    SAVELOAD_OVERRIDE(Offset, Size);
public:
    TGetParamsSnapshotFragment() {}
    TGetParamsSnapshotFragment(yint offset, yint size) : Offset(offset), Size(size) {}
    void Exec(TNetTrainContext *p) override
    {
        TVector<ui8> frag;
        frag.resize(Size);
        memcpy(frag.data(), p->ModelSnapshot.data() + Offset, Size);
        p->MasterNet->SendData(0, frag);
    }
};
REGISTER_PACKET(TGetParamsSnapshotFragment, 9);


class TModelParamsFetcher
{
    enum {
        FRAG_COUNT = 1000
    };
    bool IsFetchingFlag = false;
    yint TotalSize = 0;
    yint Offset = 0;
    yint FragSize = 0;
    TVector<ui8> Buf;
    TString ResFilename;
public:
    bool IsFetching() const { return IsFetchingFlag; }
    void StartFetch(yint sz, const TString &resFilename)
    {
        Y_VERIFY(!IsFetchingFlag);
        IsFetchingFlag = true;
        TotalSize = sz;
        Offset = 0;
        FragSize = sz / FRAG_COUNT + 1;
        Buf.resize(sz);
        ResFilename = resFilename;
    }
    TGetParamsSnapshotFragment *MakeDownloadCommand()
    {
        Y_VERIFY(IsFetchingFlag);
        yint sz = Min<yint>(FragSize, TotalSize - Offset);
        return new TGetParamsSnapshotFragment(Offset, sz);
    }
    void GotDownloadCommandResult(const TVector<ui8> &result)
    {
        yint sz = YSize(result);
        memcpy(Buf.data() + Offset, result.data(), sz);
        Offset += sz;
        Y_VERIFY(Offset <= TotalSize);
        if (Offset == TotalSize) {
            TFileStream f(false, ResFilename.c_str());
            f.Write(Buf.data(), YSize(Buf));
            IsFetchingFlag = false;
        }
    }
};

///////////////////////////////////////////////////////////////////////////////////////////////////
void RunWorker(yint port)
{
    TNetTrainContext ctx;
    ctx.MasterNet = CreateNetworNode(port, NetTrainToken);
    ctx.P2PNet = CreateNetworNode(0, NetTrainToken);
    DebugPrintf("waiting master connect on port %g\n", port * 1.);
    ConnectMaster(ctx.MasterNet);
    DebugPrintf("executing incoming commands\n");
    for (;;) {
        TIntrusivePtr<TCommandPacket> cmd = RecvCommand(ctx.MasterNet);
        if (cmd.Get()) {
            //DebugPrintf("Worker got command %s\n", typeid(*cmd.Get()).name());
            cmd->Exec(&ctx);
        }
    }
}


void RunMaster(yint startIteration, yint deviceCount, const TVector<TString> &workerArr, const TTrainContext &trainCtx, const TModelParams &params)
{
    yint workerCount = YSize(workerArr);
    Y_VERIFY(workerCount > 0 && (workerCount & (workerCount - 1)) == 0 && "pow2 worker count only is supported atm");

    TNetTrainContext ctx;
    ctx.MasterNet = CreateNetworNode(0, NetTrainToken);
    ConnectWorkers(ctx.MasterNet, workerArr);
    TVector<ECommandResult> cmdResults;

    DebugPrintf("create p2p network\n");
    CreateP2PNetwork(ctx.MasterNet, workerArr);

    DebugPrintf("create model\n");
    BroadcastCommand(ctx.MasterNet, workerCount, new TCreateModel(deviceCount, YSize(workerArr), params, trainCtx.GetMaxNodeCount()), &cmdResults);

    NHPTimer::STime tStart;
    NHPTimer::GetTime(&tStart);
    const TTrainConfig &tc = trainCtx.GetConfig();
    TModelParamsFetcher modelFetch;
    TNetworkAddr modelFetchAddr = 1;
    for (yint iter = startIteration; iter <= trainCtx.GetMaxIters(); ++iter) {
        if ((iter % trainCtx.GetEvalInterval()) == 0) {
            if (trainCtx.IsSaveModel() && !modelFetch.IsFetching()) {
                // make model params snapshot on first host
                SendCommand(ctx.MasterNet, modelFetchAddr, new TMakeParamsSnapshot());
                yint sz;
                WaitCommandResult(ctx.MasterNet, modelFetchAddr, &sz);
                modelFetch.StartFetch(sz, Sprintf("d:/eden_gpt_%.8gk.bin", iter / 1000.));
            }
            float trainErr = DistributedCalcModelErr(tc, ctx.MasterNet, workerCount, trainCtx.GetScoreTrainBatches()) * trainCtx.GetCompression();
            float testErr = DistributedCalcModelErr(tc, ctx.MasterNet, workerCount, trainCtx.GetScoreTestBatches()) * trainCtx.GetCompression();
            if (testErr != 0) {
                DebugPrintf("iter %.8gk, %g sec, train err %g, test err %g\n", iter / 1000., NHPTimer::GetTimePassed(&tStart), trainErr, testErr); fflush(0);
            } else {
                DebugPrintf("iter %.8gk, %g sec, train err %g\n", iter / 1000., NHPTimer::GetTimePassed(&tStart), trainErr); fflush(0);
            }
        }

        // fetch model snapshot one fragment per iteration
        if (modelFetch.IsFetching()) {
            SendCommand(ctx.MasterNet, modelFetchAddr, modelFetch.MakeDownloadCommand());
            TVector<ui8> result;
            WaitCommandResult(ctx.MasterNet, modelFetchAddr, &result);
            modelFetch.GotDownloadCommandResult(result);
        }

        // accumulate several batches
        EAddToModel addToModel = tc.DoAccumulate(iter) ? GRADIENT_ACCUMULATE : GRADIENT_APPLY;

        float step = trainCtx.GetStep(iter);
        TXRng iterRng(iter);
        for (yint workerAddr = 1; workerAddr <= workerCount; ++workerAddr) {
            // generate train fragments
            TVector<TVector<TFragment>> fragArr;
            fragArr.resize(deviceCount);
            for (yint deviceId = 0; deviceId < deviceCount; ++deviceId) {
                yint batchId = iter * deviceCount * workerCount + (workerAddr - 1) * deviceCount + deviceId;
                trainCtx.MakeTrainBatches(iterRng, batchId, &fragArr[deviceId]);
            }
            SendCommand(ctx.MasterNet, workerAddr, new TBackprop(iter, tc, step, addToModel, fragArr));
        }
        CollectCommandResults(ctx.MasterNet, workerCount, &cmdResults);
    }
}
}
