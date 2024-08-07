#pragma once
#include "poller.h"
#include "http_request.h"
#include <lib/hp_timer/hp_timer.h>

namespace NNet
{
class THttpServer : public TThrRefBase
{
    SOCKET Listen;
    int ListenPort;

    struct TRecvRequestBuffer
    {
        TVector<char> Buffer;
        int Offset = 0;
        float ElapsedTime = 0;
    };
    struct TSendReplyBuffer
    {
        TVector<char> Buffer;
        int Offset = 0;
    };
    THashMap<SOCKET, TRecvRequestBuffer> ReqSockets;
    THashMap<SOCKET, TSendReplyBuffer> RespSockets;
    NHPTimer::STime PrevTime;

public:
    class TRequest
    {
        TIntrusivePtr<THttpServer> Srv;
        SOCKET Sock = INVALID_SOCKET;

        void Reply(const TString &reply, const TVector<char> &data, const char *content);
    public:
        THttpRequest Req;

        void ReplyNotFound();
        void ReplyXML(const TString &reply);
        void ReplyHTML(const TString &reply);
        void ReplyPlainText(const TString &reply);
        void ReplyBin(const TVector<char> &reply);
        void ReplyBMP(const TVector<char> &data);

        friend class THttpServer;
    };

private:
    void ReplyBadRequest(SOCKET s);
    bool RecvQuery(SOCKET s, TRecvRequestBuffer *pBuf, TVector<TRequest> *pReqArr);
    void ParseQuery(SOCKET s, TRecvRequestBuffer *pBuf, TVector<TRequest> *pReqArr);
    void SendReply(SOCKET s, TVector<char> *pData);
    ~THttpServer();

public:
    THttpServer(int port);
    int GetPort() const { return ListenPort; }
    void Poll(TTcpPoller *pl);
    void OnPoll(TTcpPoller *pl, TVector<TRequest> *pReqArr);
};


inline void GetQueries(float timeout, TTcpPoller *poller, TIntrusivePtr<THttpServer> p, TVector<THttpServer::TRequest> *pQueries)
{
    poller->Start();
    p->Poll(poller);
    poller->Poll(timeout);
    poller->Start();
    p->OnPoll(poller, pQueries);
}
}
