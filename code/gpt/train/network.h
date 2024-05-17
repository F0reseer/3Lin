#pragma once
#include <util/mem_io.h>
#include <lib/guid/guid.h>


// worker port
const yint DEFAULT_WORKER_PORT = 10000;

typedef int TNetworkAddr;

struct TNetPacket : public TThrRefBase
{
    TNetworkAddr Addr = 0;
    TVector<ui8> Data;
};

struct INetwork : public TThrRefBase
{
    virtual yint GetPort() = 0;
    virtual yint GetMyAddr() = 0;
    virtual void Send(TIntrusivePtr<TNetPacket> p) = 0;
    virtual TIntrusivePtr<TNetPacket> Recv() = 0;

    // connect peers
    virtual void SetMyAddr(TNetworkAddr addr) = 0;
    virtual void Connect(const TString &hostName, TNetworkAddr addr) = 0;
    virtual yint GetPeerCount() = 0;
    virtual void StopAcceptingConnections() = 0;

    // utils
    template <class T>
    void SendData(TNetworkAddr addr, T &x)
    {
        TIntrusivePtr<TNetPacket> pkt = new TNetPacket;
        pkt->Addr = addr;
        SerializeMem(false, &pkt->Data, x);
        Send(pkt);
    }
};

TIntrusivePtr<INetwork> CreateNetworNode(yint port, const TGuid &token);
void ConnectWorkers(TIntrusivePtr<INetwork> net, const TVector<TString> &peerList);
void ConnectMaster(TIntrusivePtr<INetwork> net);
void ConnectP2P(TIntrusivePtr<INetwork> net, yint myAddr, const TVector<TString> &peerList);

void TestNetwork(bool bMaster);
