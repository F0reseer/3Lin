#include "stdafx.h"
#include "net_data.h"
#include <lib/guid/guid.h>
#include <lib/net/tcp_cmds.h>


using namespace NNet;

const yint TRAIN_DATA_PORT = 18183;

static TGuid TrainDataToken(0x1e4da508, 0x5338525c, 0x40b87806, 0x4662301b);


///////////////////////////////////////////////////////////////////////////////////////////////////
namespace NNetData
{

struct IDataCmd : public TCommandBase
{
    virtual TIntrusivePtr<TTcpPacket> Exec(TIntrusivePtr<IDataSource> data) = 0;
};
static TCommandFabric<IDataCmd> cmdFabric;


class TGetStats : public IDataCmd
{
    TIntrusivePtr<TTcpPacket> Exec(TIntrusivePtr<IDataSource> data) override
    {
        IDataSource::TDataStats stats = data->GetStats();
        return MakePacket(stats);
    }
};
REGISTER_PACKET(cmdFabric, TGetStats, 1);


class TSampleFragments : public IDataCmd
{
    IDataSource::ETrainTest TRT = IDataSource::TRAIN;
    yint RngSeed = 1313;
    yint FragCount = 0;
    yint FragLen = 0;
    SAVELOAD(TRT, RngSeed, FragCount, FragLen);

    TIntrusivePtr<TTcpPacket> Exec(TIntrusivePtr<IDataSource> data) override
    {
        TVector<TFragment> fragArr;
        data->SampleFragments(TRT, RngSeed, FragCount, FragLen, &fragArr);
        return MakePacket(fragArr);

    }
public:
    TSampleFragments() {}
    TSampleFragments(IDataSource::ETrainTest trt, yint rngSeed, yint fragCount, yint fragLen)
        : TRT(trt), RngSeed(rngSeed), FragCount(fragCount), FragLen(fragLen)
    {
    }
};
REGISTER_PACKET(cmdFabric, TSampleFragments, 2);
}
using namespace NNetData;


///////////////////////////////////////////////////////////////////////////////////////////////////
void RunDataServer(TIntrusivePtr<ITcpSendRecv> net, TIntrusivePtr<IDataSource> data)
{
    TIntrusivePtr<TSyncEvent> ev = new TSyncEvent();
    TIntrusivePtr<ITcpAccept> dataAccept = net->StartAccept(TRAIN_DATA_PORT, TrainDataToken, ev);
    TIntrusivePtr<TTcpRecvQueue> dataQueue = new TTcpRecvQueue(ev);

    TXRng rng(GetCycleCount());
    for (;;) {
        ev->Wait();

        // accept new data connections
        TIntrusivePtr<ITcpConnection> conn;
        while (dataAccept->GetNewConnection(&conn)) {
            conn->SetExitOnError(false);
            net->StartSendRecv(conn, dataQueue);
        }

        // process data requests
        TIntrusivePtr<TTcpPacketReceived> pkt;
        while (dataQueue->Dequeue(&pkt)) {
            TIntrusivePtr<IDataCmd> cmd = DeserializeCommand(cmdFabric, &pkt->Data);
            net->Send(pkt->Conn, cmd->Exec(data));
        }
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
template <class T>
static void GetResponse(TIntrusivePtr<TSyncEvent> ev, TIntrusivePtr<TTcpRecvQueue> q, T *pRes)
{
    for (;;) {
        ev->Wait();
        TIntrusivePtr<TTcpPacketReceived> pkt;
        if (q->Dequeue(&pkt)) {
            SerializeMem(IO_READ, &pkt->Data, *pRes);
            return;
        }
    }
}

class TNetDataSource : public IDataSource
{
    TIntrusivePtr<ITcpSendRecv> Net;
    TIntrusivePtr<TSyncEvent> Event;
    TIntrusivePtr<TTcpRecvQueue> DataQueue;
    TIntrusivePtr<ITcpConnection> DataConn;
    TDataStats DataStats;

    const TDataStats &GetStats() const override
    {
        return DataStats;
    }
    void SampleFragments(ETrainTest trt, yint rngSeed, yint fragCount, yint len, TVector<TFragment> *pFragArr) override
    {
        SendCommand(cmdFabric, Net, DataConn, new TSampleFragments(trt, rngSeed, fragCount, len));
        GetResponse(Event, DataQueue, pFragArr);
    }
public:
    TNetDataSource(TIntrusivePtr<ITcpSendRecv> net, const TString &addr) : Net(net)
    {
        Event = new TSyncEvent();
        DataQueue = new TTcpRecvQueue(Event);
        DataConn = Connect(addr, TRAIN_DATA_PORT, TrainDataToken);
        Net->StartSendRecv(DataConn, DataQueue);

        SendCommand(cmdFabric, Net, DataConn, new TGetStats());
        GetResponse(Event, DataQueue, &DataStats);
    }
};


TIntrusivePtr<IDataSource> ConnectDataServer(TIntrusivePtr<ITcpSendRecv> net, const TString &addr)
{
    return new TNetDataSource(net, addr);
}
