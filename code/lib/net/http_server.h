#pragma once

namespace NNet
{
struct THttpRequest;

class THttpServer
{
    SOCKET sAccept;
    int nAcceptPort;
    fd_set FS;

    struct TRecvRequestBuffer
    {
        char Buffer[100000];
        int Offset;

        TRecvRequestBuffer() : Offset(0) {}
    };
    typedef THashMap<SOCKET,TRecvRequestBuffer> TRecvSocketsHash;
    TRecvSocketsHash ReqSockets;

    enum EReadReqResult
    {
        FAILED=0,
        OK,
        WAIT,
        CLOSED
    };
    EReadReqResult ReadRequestData(SOCKET s, TRecvRequestBuffer *buf);
    bool ReadRequest(SOCKET s, TRecvRequestBuffer *buf, THttpRequest *pReq);
    SOCKET AcceptNonBlockingImpl(THttpRequest *pReq);

public:
    THttpServer(int _nAcceptPort);
    ~THttpServer();

    int GetPort() const { return nAcceptPort; }
    bool CanAccept(float timeoutSec);
    SOCKET AcceptNonBlocking(THttpRequest *pReq);
};


void ReplyBadRequest(SOCKET s);
void ReplyNotFound(SOCKET s);

// for replying part by part
bool SendHeader(SOCKET s, const char *type, const char *encoding = NULL); // false if send() fails
bool SendRetry(SOCKET s, const char *buf, yint len); // false if send() fails
void CloseConnection(SOCKET s);

void HttpReplyXML(SOCKET s, const string &reply);
void HttpReplyHTML(SOCKET s, const string &reply);
void HttpReplyPlainText(SOCKET s, const string &reply);
void HttpReplyBin(SOCKET s, const TVector<char> &reply);
void HttpReplyBMP(SOCKET s, const TVector<char> &data);
}
