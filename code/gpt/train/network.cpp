#include "stdafx.h"
#include "network.h"
#include <lib/net/ip_address.h>
#include <lib/net/net_util.h>
#include <util/thread.h>


using namespace NNet;


static void MakeFastSocket(SOCKET s)
{
    int flag = 1;
    setsockopt(s, IPPROTO_TCP, TCP_NODELAY, (char *)&flag, sizeof(int));
    int bufSize = 1 << 20;
    setsockopt(s, SOL_SOCKET, SO_SNDBUF, (char *)&bufSize, sizeof(int));
    setsockopt(s, SOL_SOCKET, SO_RCVBUF, (char *)&bufSize, sizeof(int));
    MakeNonBlocking(s);
}


class TNetwork : public INetwork
{
    // packet structure
    struct TNetPacketHeader
    {
        yint Size;
    };

    struct THandshake
    {
        TGuid Token;
        TNetworkAddr PeerAddr = 0;

        THandshake() {}
        THandshake(const TGuid &token, TNetworkAddr addr) : Token(token), PeerAddr(addr) {}
    };


private:
    struct TPeer
    {
        TNetworkAddr Addr = 0;
        SOCKET Sock = INVALID_SOCKET;

        // recv data
        TNetPacketHeader RecvHeader;
        yint RecvHeaderOffset = 0;
        TIntrusivePtr<TNetPacket> RecvPacket;
        yint RecvOffset = -1;

        // send data
        TNetPacketHeader SendHeader;
        yint SendHeaderOffset = 0;
        TVector<TIntrusivePtr<TNetPacket>> SendQueue;
        yint SendOffset = -1;

    public:
        void AssignSocket(SOCKET s)
        {
            Y_ASSERT(Sock == INVALID_SOCKET);
            MakeFastSocket(s);
            Sock = s;
        }

        void OnFail()
        {
            DebugPrintf("net send/recv failed\n");
            exit(0); //abort();
        }

        void Recv(TSingleConsumerJobQueue<TIntrusivePtr<TNetPacket>> *pList)
        {
            if (RecvOffset == -1) {
                Y_ASSERT(RecvPacket == nullptr);
                char *data = (char*) &RecvHeader;
                int headerSize = sizeof(TNetPacketHeader);
                int rv = recv(Sock, data + RecvHeaderOffset, headerSize - RecvHeaderOffset, 0);
                if (rv == SOCKET_ERROR || rv == 0) {
                    DebugPrintf("recv header fail, rv = %g\n", rv * 1.);
                    OnFail();
                }
                RecvHeaderOffset += rv;
                if (RecvHeaderOffset == headerSize) {
                    RecvPacket = new TNetPacket();
                    RecvPacket->Addr = Addr;
                    RecvPacket->Data.resize(RecvHeader.Size);
                    RecvOffset = 0;
                    RecvHeaderOffset = 0;
                }
            } else {
                yint sz = YSize(RecvPacket->Data) - RecvOffset;
                yint rv = recv(Sock, (char *)RecvPacket->Data.data() + RecvOffset, sz, 0);
                if (rv == 0 || rv == SOCKET_ERROR) {
                    DebugPrintf("recv fail, rv = %g\n", rv * 1.);
                    OnFail();
                } else if (rv == sz) {
                    RecvOffset = -1;
                    pList->Enqueue(RecvPacket);
                    RecvPacket = nullptr;
                } else {
                    RecvOffset += rv;
                }
            }
        }

        void Send()
        {
            if (!SendQueue.empty()) {
                if (SendOffset < 0) {
                    Y_ASSERT(SendQueue[0]->Addr == Addr);
                    if (SendHeaderOffset == 0) {
                        SendHeader.Size = YSize(SendQueue[0]->Data);
                    }
                    char *data = (char *) &SendHeader;
                    int headerSize = sizeof(TNetPacketHeader);
                    yint rv = send(Sock, data + SendHeaderOffset, headerSize - SendHeaderOffset, 0);
                    if (rv == 0 || rv == SOCKET_ERROR) {
                        DebugPrintf("send header fail, rv = %g\n", rv * 1.);
                        OnFail();
                    }
                    SendHeaderOffset += rv;
                    if (SendHeaderOffset == headerSize) {
                        SendHeaderOffset = 0;
                        SendOffset = 0;
                    }
                } else {
                    const TVector<ui8> &data = SendQueue[0]->Data;
                    yint sz = YSize(data) - SendOffset;
                    yint rv = send(Sock, (const char *)data.data() + SendOffset, sz, 0);
                    if (rv == 0 || rv == SOCKET_ERROR) {
                        DebugPrintf("send data fail, rv = %g\n", rv * 1.);
                        OnFail();
                    } else if (rv == sz) {
                        SendQueue.erase(SendQueue.begin());
                        SendOffset = -1;
                    } else {
                        SendOffset += rv;
                    }
                }
            }
        }
    };


private:
    struct TConnectAttempt
    {
        TNetworkAddr Addr = 0;
        SOCKET Sock = INVALID_SOCKET;
        THandshake Handshake;
        yint RecvOffset = 0;

    public:
        TConnectAttempt() {}

        TConnectAttempt(SOCKET s) : Sock(s)
        {
            MakeFastSocket(Sock);
        }

        bool Recv(TNetwork *p, const TGuid &token)
        {
            char *data = (char *)&Handshake;
            int headerSize = sizeof(Handshake);
            int rv = recv(Sock, data + RecvOffset, headerSize - RecvOffset, 0);
            if (rv == SOCKET_ERROR || rv == 0) {
                return false;
            }
            RecvOffset += rv;
            if (RecvOffset == headerSize) {
                if (Handshake.Token != token) {
                    return false;
                }
                DebugPrintf("accept peer %g connect\n", Handshake.PeerAddr * 1.);
                p->AddPeer(Sock, Handshake.PeerAddr);
                Sock = INVALID_SOCKET; // prevent closesocket()
                return false;
            }
            return true;
        }

        void Cancel()
        {
            if (Sock != INVALID_SOCKET) {
                closesocket(Sock);
                Sock = INVALID_SOCKET;
            }
        }
    };


private:
    TGuid Token;
    yint MyPort;
    TNetworkAddr MyAddr = 0;
    TThread Thr;
    SOCKET Listen = INVALID_SOCKET;
    TVector<TPeer> PeerArr;
    std::atomic<yint> CurrentPeerCount;
    TVector<TConnectAttempt> ConnectArr;
    TVector<pollfd> FS;
    TSingleConsumerJobQueue<TIntrusivePtr<TNetPacket>> SendList;
    TSingleConsumerJobQueue<TIntrusivePtr<TNetPacket>> RecvList;
    TSingleConsumerJobQueue<TPeer> NewPeers;
    volatile bool Exit = false;
    volatile bool AcceptConnections = true;


private:
    void ListenOps()
    {
        if (Listen == INVALID_SOCKET) {
            Y_ASSERT(ConnectArr.empty());
            return;
        }
        if (!AcceptConnections) {
            closesocket(Listen);
            Listen = INVALID_SOCKET;
            for (TConnectAttempt &connect : ConnectArr) {
                connect.Cancel();
            }
            ConnectArr.resize(0);
            return;
        }
        yint connectCount = YSize(ConnectArr);
        ClearPodArray(&FS, connectCount + 1);
        for (yint k = 0; k < connectCount; ++k) {
            FS[k].fd = ConnectArr[k].Sock;
            FS[k].events = POLLRDNORM;
        }
        FS[connectCount].fd = Listen;
        FS[connectCount].events = POLLRDNORM;

        poll(FS.data(), YSize(FS), 0); // no timeout

        yint dst = 0;
        for (yint k = 0; k < connectCount; ++k) {
            TConnectAttempt &connect = ConnectArr[k];
            int events = FS[k].revents;
            bool keepConnecting = true;
            if ((events) & ~(POLLRDNORM | POLLWRNORM)) {
                DebugPrintf("Nontrivial connect events %x\n", events); fflush(0);
                keepConnecting = false;
            } else if (events & POLLRDNORM) {
                keepConnecting = connect.Recv(this, Token);
            }
            if (keepConnecting) {
                ConnectArr[dst++] = connect;
            } else {
                connect.Cancel();
            }
        }
        ConnectArr.resize(dst);
        // accept new connect attempts
        if (FS[connectCount].revents != 0) {
            //sockaddr_in6 incomingAddr;
            //int nIncomingAddrLen = sizeof(incomingAddr);
            SOCKET s = accept(Listen, 0, 0);//(sockaddr*)&incomingAddr, &nIncomingAddrLen);
            if (s == INVALID_SOCKET) {
                DebugPrintf("accept() failed for signaled socket\n");
                abort();
            }
            ConnectArr.push_back(TConnectAttempt(s));
        }
    }

    void PollAndPerformOps()
    {
        yint peerCount = YSize(PeerArr);
        ClearPodArray(&FS, peerCount);
        for (yint k = 0; k < peerCount; ++k) {
            FS[k].fd = PeerArr[k].Sock;
            FS[k].events = POLLRDNORM | POLLWRNORM;
        }

        poll(FS.data(), YSize(FS), 0); // no timeout

        for (yint k = 0; k < peerCount; ++k) {
            TPeer &peer = PeerArr[k];
            int events = FS[k].revents;
            if ((events) & ~(POLLRDNORM | POLLWRNORM)) {
                DebugPrintf("Nontrivial events %x\n", events); fflush(0);
                peer.OnFail();
                continue; // has to remove this peer here
            }
            if (events & POLLRDNORM) {
                peer.Recv(&RecvList);
            }
            if (events & POLLWRNORM) {
                peer.Send();
            }
        }
    }

    yint GetPeerIndex(TNetworkAddr addr) const
    {
        for (yint k = 0; k < YSize(PeerArr); ++k) {
            if (PeerArr[k].Addr == addr) {
                return k;
            }
        }
        return -1;
    }

public:
    void AddPeer(SOCKET s, TNetworkAddr peerAddr)
    {
        TPeer newPeer;
        newPeer.Addr = peerAddr;
        newPeer.AssignSocket(s);
        NewPeers.Enqueue(newPeer);
    }

    void WorkerThread()
    {
        TPeer newPeer;
        while (!Exit) {
            TVector<TIntrusivePtr<TNetPacket>> jobArr;
            while (NewPeers.DequeueFirst(&newPeer)) {
                Y_VERIFY(GetPeerIndex(newPeer.Addr) == -1);
                PeerArr.push_back(newPeer);
                CurrentPeerCount = YSize(PeerArr);
            }
            if (SendList.DequeueAll(&jobArr)) {
                Reverse(jobArr.begin(), jobArr.end());
                for (TIntrusivePtr<TNetPacket> p : jobArr) {
                    bool found = false;
                    for (yint k = 0; k < YSize(PeerArr); ++k) {
                        if (PeerArr[k].Addr == p->Addr) {
                            PeerArr[k].SendQueue.push_back(p);
                            found = true;
                            break;
                        }
                    }
                    Y_VERIFY(found);
                }
            } else {
                PollAndPerformOps();
                ListenOps();
            }
        }
    }

private:
    void StartListen(yint listenPort)
    {
        Y_ASSERT(Listen == INVALID_SOCKET);
        Y_ASSERT(AcceptConnections);
        Listen = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        if (Listen == INVALID_SOCKET) {
            DebugPrintf("Failed to create socket\n");
            abort();
        }
        sockaddr_in name;
        Zero(name);
        name.sin_family = AF_INET;
        //name.sin_addr = inaddr_any;
        name.sin_port = htons(listenPort);
        if (bind(Listen, (sockaddr *)&name, sizeof(name)) != 0) {
            DebugPrintf("Port %d already in use\n", ntohs(name.sin_port));
            abort();
        }
        if (listen(Listen, SOMAXCONN) != 0) {
            DebugPrintf("listen() failed\n");
            abort();
        }
        sockaddr_in localAddr;
        socklen_t len = sizeof(localAddr);
        if (getsockname(Listen, (sockaddr *)&localAddr, &len)) {
            Y_VERIFY(0 && "no self address");
        }
        MyPort = ntohs(localAddr.sin_port);
    }


    ~TNetwork()
    {
        Exit = true;
        Thr.Join();
        // for graceful shutdown have to close all sockets, not just Listen. This code is never called in practice
        closesocket(Listen);
        DebugPrintf("network sockets leak\n");
    }

public:
    TNetwork(yint listenPort, const TGuid &token) : Token(token), CurrentPeerCount(0)
    {
        StartListen(listenPort);
        Thr.Create(this);
    }

    yint GetPort() override
    {
        return MyPort;
    }

    yint GetMyAddr() override
    {
        return MyAddr;
    }

    void Send(TIntrusivePtr<TNetPacket> p) override
    {
        if (p->Addr == MyAddr) {
            // packet is sent to self
            RecvList.Enqueue(p);
        } else {
            SendList.Enqueue(p);
        }
    }

    TIntrusivePtr<TNetPacket> Recv() override
    {
        TIntrusivePtr<TNetPacket> res;
        RecvList.DequeueFirst(&res);
        return res;
    }

    void SetMyAddr(TNetworkAddr addr) override
    {
        MyAddr = addr;
    }

    void Connect(const TString &hostName, TNetworkAddr addr) override
    {
        sockaddr_in sa;
        TString pureHost;
        if (!ParseInetName(&sa, &pureHost, hostName.c_str(), DEFAULT_WORKER_PORT)) {
            DebugPrintf("Failed to parse server address %s\n", hostName.c_str());
            abort();
        }
        SOCKET s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        if (s == INVALID_SOCKET) {
            DebugPrintf("socket() failed\n");
            abort();
        }
        if (connect(s, (sockaddr *)&sa, sizeof(sa)) == SOCKET_ERROR) {
            DebugPrintf("Failed to connect to %s\n", hostName.c_str());
            abort();
        }
        THandshake handshake(Token, MyAddr);
        yint rv = send(s, (const char *)&handshake, sizeof(handshake), 0);
        if (rv != sizeof(handshake)) {
            DebugPrintf("Failed to send handshake\n");
            abort();
        }
        AddPeer(s, addr);
    }

    yint GetPeerCount() override
    {
        return CurrentPeerCount;
    }

    void StopAcceptingConnections() override
    {
        AcceptConnections = false;
    }
};


TIntrusivePtr<INetwork> CreateNetworNode(yint port, const TGuid &token)
{
    return new TNetwork(port, token);
}

void ConnectWorkers(TIntrusivePtr<INetwork> net, const TVector<TString> &peerList)
{
    net->SetMyAddr(0);
    net->StopAcceptingConnections();
    for (yint k = 0; k < YSize(peerList); ++k) {
        DebugPrintf("connect (%s)\n", peerList[k].c_str());
        net->Connect(peerList[k], k + 1);
        TNetworkAddr addr = k + 1;
        net->SendData(k + 1, addr);
    }
}


void ConnectMaster(TIntrusivePtr<INetwork> net)
{
    for (;;) {
        TIntrusivePtr<TNetPacket> p = net->Recv();
        if (p.Get()) {
            TNetworkAddr myAddr;
            SerializeMem(true, &p->Data, myAddr);
            net->SetMyAddr(myAddr);
            DebugPrintf("Connected to master with addr %g\n", myAddr * 1.);
            net->StopAcceptingConnections();
            break;
        }
    }
}


void ConnectP2P(TIntrusivePtr<INetwork> net, yint myAddr, const TVector<TString> &peerList)
{
    DebugPrintf("connect peers\n");
    net->SetMyAddr(myAddr);
    for (yint k = myAddr + 1; k < YSize(peerList); ++k) {
        DebugPrintf("connect %g (%s)\n", k * 1., peerList[k].c_str());
        net->Connect(peerList[k], k);
    }
}


void TestNetwork(bool bMaster)
{
    TGuid testToken(1, 2, 3, 4);
    TIntrusivePtr<INetwork> net = CreateNetworNode(10000, testToken);
    if (!bMaster) {
        ConnectMaster(net);
    } else {
        TVector<TString> workers;
        workers.push_back("192.168.2.27:10000");
        ConnectWorkers(net, workers);
    }
    yint inFlyPacket = 0;
    for (;;) {
        if (inFlyPacket < 5) {
            TIntrusivePtr<TNetPacket> pkt = new TNetPacket;
            pkt->Addr = bMaster ? 1 : 0;
            pkt->Data.resize(256 * 256 / 8);
            net->Send(pkt);
            ++inFlyPacket;
        }
        TIntrusivePtr<TNetPacket> rp = net->Recv();
        if (rp.Get()) {
            --inFlyPacket;
        }
    }
}
