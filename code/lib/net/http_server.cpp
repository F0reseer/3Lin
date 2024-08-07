#include "stdafx.h"
#include <errno.h>
#include "net_util.h"
#include "http_server.h"
#include "http_request.h"
#include "ip_address.h"
//#include <ws2ipdef.h>

namespace NNet
{
#if (!defined(_win_) && !defined(_darwin_))
    int SEND_FLAGS = MSG_NOSIGNAL;
#else
    int SEND_FLAGS = 0;
#endif

////////////////////////////////////////////////////////////////////////////////

static bool StartListen(SOCKET *psAccept, int *port)
{
    SOCKET &sAccept = *psAccept;
    if (sAccept != INVALID_SOCKET) {
        closesocket(sAccept);
    }
    //
    sAccept = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (sAccept == INVALID_SOCKET) {
        return false;
    }
    {
        //int flag = 0;
        //setsockopt(sAccept, IPPROTO_IPV6, IPV6_V6ONLY, (const char*)&flag, sizeof(flag));

        int flag = 1;
        setsockopt(sAccept, SOL_SOCKET, SO_REUSEADDR, (const char*)&flag, sizeof(flag));
    }

    sockaddr_in name;
    Zero(name);
    name.sin_family = AF_INET;
    //name.sin_addr = inaddr_any;
    name.sin_port = htons((ui16)*port);

    if (bind(sAccept, (sockaddr*)&name, sizeof(name)) != 0) {
        Y_ASSERT(0);
        closesocket(sAccept);
        sAccept = INVALID_SOCKET;
        return false;
    }
    if (listen(sAccept, SOMAXCONN) != 0) {
        Y_ASSERT(0);
        closesocket(sAccept);
        sAccept = INVALID_SOCKET;
        return false;
    }
    if (*port == 0) {
        // figure out assigned port
        sockaddr_in resAddr;
        socklen_t len = sizeof(resAddr);
        if (getsockname(sAccept, (sockaddr*)&resAddr, &len)) {
            Y_ASSERT(0);
            closesocket(sAccept);
            sAccept = INVALID_SOCKET;
            return false;
        }
        *port = ntohs(resAddr.sin_port);
    }
    MakeNonBlocking(sAccept);
    return true;
}


static char *GetRequest(char *pszReq)
{
    if (strncmp(pszReq, "GET", 3) == 0)
        pszReq += 3;
    else if (strncmp(pszReq, "POST", 4) == 0)
        pszReq += 4;
    else
        return 0;
    while (*pszReq && isspace((unsigned char)*pszReq))
        ++pszReq;
    char *pszRes = pszReq;
    while (*pszReq && !isspace((unsigned char)*pszReq))
        ++pszReq;
    *pszReq = 0;
    return pszRes;
}


////////////////////////////////////////////////////////////////////////////////
struct THttpReplayWriter
{
    TVector<char> Buf;

    void Write(yint sz, const void *data)
    {
        if (sz > 0) {
            yint ptr = YSize(Buf);
            Buf.resize(ptr + sz);
            memcpy(Buf.data() + ptr, data, sz);
        }
    }
    void Write(const TString &str)
    {
        Write(YSize(str), str.data());
    }
    void Write(const TVector<char> &vec)
    {
        Write(YSize(vec), vec.data());
    }
    void Write(const char *str)
    {
        Write(strlen(str), str);
    }
};


////////////////////////////////////////////////////////////////////////////////
THttpServer::THttpServer(int port)
    : Listen(INVALID_SOCKET), ListenPort(port)
{
    if (!StartListen(&Listen, &ListenPort)) {
        fprintf(stderr, "StartListen() failed: %s\n", strerror(errno));
    }
    NHPTimer::GetTime(&PrevTime);
}


THttpServer::~THttpServer()
{
    for (auto i = ReqSockets.begin(); i != ReqSockets.end(); ++i) {
        closesocket(i->first);
    }
    if (Listen != INVALID_SOCKET) {
        closesocket(Listen);
    }
}


void THttpServer::Poll(TTcpPoller *pl)
{
    for (auto it = ReqSockets.begin(); it != ReqSockets.end(); ++it) {
        pl->AddSocket(it->first, POLLRDNORM);
    }
    for (auto it = RespSockets.begin(); it != RespSockets.end(); ++it) {
        pl->AddSocket(it->first, POLLWRNORM);
    }
    pl->AddSocket(Listen, POLLRDNORM);
}


void THttpServer::ParseQuery(SOCKET s, TRecvRequestBuffer *pBuf, TVector<TRequest> *pReqArr)
{
    if (pBuf->Offset == YSize(pBuf->Buffer)) {
        pBuf->Buffer.push_back(0);
    } else {
        pBuf->Buffer[pBuf->Offset] = 0;
    }
    const char *hdrFin = strstr(pBuf->Buffer.data(), "\r\n\r\n");
    if (hdrFin) {
        char *pszRequest = GetRequest(pBuf->Buffer.data());
        TRequest req;
        req.Srv = this;
        req.Sock = s;
        if (pszRequest && ParseRequest(&req.Req, pszRequest)) {
            yint dataStart = (yint)(hdrFin - pBuf->Buffer.data() + 4);
            if (dataStart != pBuf->Offset) {
                TVector<char> data;
                data.swap(pBuf->Buffer);
                data.erase(data.begin(), data.begin() + dataStart);
                req.Req.Data.swap(data);
            }
            pReqArr->push_back(req);
            return;
        }
    }
    ReplyBadRequest(s);
}


// returns true if query was processed
bool THttpServer::RecvQuery(SOCKET s, TRecvRequestBuffer *pBuf, TVector<TRequest> *pReqArr)
{
    const yint MAX_QUERY_SIZE = 1000000;
    if (pBuf->Offset == YSize(pBuf->Buffer)) {
        if (pBuf->Offset > MAX_QUERY_SIZE) {
            // query is too long
            closesocket(s);
            return true;
        }
        if (pBuf->Offset == 0) {
            pBuf->Buffer.resize(16384);
        } else {
            pBuf->Buffer.resize(YSize(pBuf->Buffer) * 2);
        }
    }
    int rv = recv(s, pBuf->Buffer.data() + pBuf->Offset, YSize(pBuf->Buffer) - pBuf->Offset, 0);
    if (rv == 0) {
        // full query is received
        ParseQuery(s, pBuf, pReqArr);
        return true;
    } else if (rv == SOCKET_ERROR) {
        yint err = errno;
        if (err != 0 && err != EWOULDBLOCK && err != EAGAIN) {
            fprintf(stderr, "unexpected recv() error: %s\n", strerror(errno));
            closesocket(s);
            return true;
        }
    } else {
        Y_VERIFY(rv > 0);
        pBuf->Offset += rv;
        if (pBuf->Offset > 4 && strncmp(pBuf->Buffer.data() + pBuf->Offset - 4, "\r\n\r\n", 4) == 0) {
            // query seems to be received
            ParseQuery(s, pBuf, pReqArr);
            return true;
        }
    }
    return false;
}


void THttpServer::OnPoll(TTcpPoller *pl, TVector<TRequest> *pReqArr)
{
    const float REQUEST_TIMEOUT = 5;

    float deltaT = NHPTimer::GetTimePassed(&PrevTime);
    deltaT = ClampVal<float>(deltaT, 0, 0.5); // avoid spurious too large time steps

    for (auto it = ReqSockets.begin(); it != ReqSockets.end(); ) {
        auto k = it++;
        SOCKET s = k->first;
        TRecvRequestBuffer &reqBuf = k->second;
        yint events = pl->CheckSocket(s);
        if (events & ~(POLLRDNORM | POLLWRNORM)) {
            closesocket(s);
            ReqSockets.erase(k);
            continue;
        } else if (events &POLLRDNORM) {
            if (RecvQuery(s, &reqBuf, pReqArr)) {
                ReqSockets.erase(k);
                continue;
            }
        }
        reqBuf.ElapsedTime += deltaT;
        if (reqBuf.ElapsedTime > REQUEST_TIMEOUT) {
            closesocket(s);
            ReqSockets.erase(k);
        }
    }

    for (auto it = RespSockets.begin(); it != RespSockets.end();) {
        auto k = it++;
        SOCKET s = k->first;
        TSendReplyBuffer &resp = k->second;
        yint events = pl->CheckSocket(s);
        if (events & ~(POLLRDNORM | POLLWRNORM)) {
            //DebugPrintf("send(), non-trivial poll flags %g\n", events * 1.);
        } else if (events & POLLWRNORM) {
            int sz = Min<yint>(1ll << 24, YSize(resp.Buffer) - resp.Offset);
            int rv = send(s, resp.Buffer.data() + resp.Offset, sz, SEND_FLAGS);
            if (rv == SOCKET_ERROR) {
                yint err = errno;
                if (err != 0 && err != EWOULDBLOCK && err != EAGAIN) {
                    DebugPrintf("send(), unexpected errno %g\n", err * 1.);
                } else {
                    continue;
                }
            } else {
                resp.Offset += rv;
                if (resp.Offset < YSize(resp.Buffer)) {
                    continue;
                }
            }
        }
        closesocket(s);
        RespSockets.erase(k);
    }

    yint events = pl->CheckSocket(Listen);
    if ((events) & ~(POLLRDNORM | POLLWRNORM)) {
        DebugPrintf("Nontrivial accept events %x\n", events); fflush(0);
        abort();
    } else if (events & POLLRDNORM) {
        SOCKET s = accept(Listen, (sockaddr *)nullptr, nullptr);
        if (s == INVALID_SOCKET) {
            int err = errno;
            // somehow errno 0 can happen on windows
            if (err != 0 && err != EWOULDBLOCK && err != EAGAIN) {
                DebugPrintf("accept() failed for signaled socket, errno %d\n", err);
                abort();
            }
        } else {
            MakeNonBlocking(s);
            ReqSockets[s];
        }
    }
}


void THttpServer::SendReply(SOCKET s, TVector<char> *pData)
{
    Y_VERIFY(!pData->empty());
    int rv = send(s, pData->data(), YSize(*pData), SEND_FLAGS);
    if (rv != YSize(*pData)) {
        TSendReplyBuffer &buf = RespSockets[s];
        pData->swap(buf.Buffer);
        buf.Offset = (rv > 0) ? rv : 0;
    } else {
        closesocket(s);
    }
}



////////////////////////////////////////////////////////////////////////////////
static TString FormHeader(const char *type, const char *encoding = NULL)
{
    TString reply;
    reply += "HTTP/1.0 200 OK\r\n"
        "Connection: close\r\n"
        "Content-Type: ";
    reply += type;
    reply += "\r\n";

    if (encoding) {
        reply += "Content-Encoding: ";
        reply += encoding;
        reply += "\r\n";
    }

    reply += "Cache-control: no-cache, max-age=0\r\n"
        "Expires: Thu, 01 Jan 1970 00:00:01 GMT\r\n"
        "\r\n";

    return reply;
}

void THttpServer::ReplyBadRequest(SOCKET s)
{
    char reply[] = "HTTP/1.0 400 Bad request\r\nConnection: close\r\n\r\n";
    THttpReplayWriter rep;
    rep.Write(reply);
    SendReply(s, &rep.Buf);
}

void THttpServer::TRequest::Reply(const TString &reply, const TVector<char> &data, const char *content)
{
    THttpReplayWriter rep;
    rep.Write(FormHeader(content));
    rep.Write(reply);
    rep.Write(data);
    Srv->SendReply(Sock, &rep.Buf);
}

void THttpServer::TRequest::ReplyNotFound()
{
    char reply[] = "HTTP/1.0 404 Not Found\r\nConnection: close\r\n\r\n";
    THttpReplayWriter rep;
    rep.Write(reply);
    Srv->SendReply(Sock, &rep.Buf);
}

void THttpServer::TRequest::ReplyXML(const TString &reply)
{
    Reply(reply, TVector<char>(), "text/xml");
}

void THttpServer::TRequest::ReplyHTML(const TString &reply)
{
    Reply(reply, TVector<char>(), "text/html");
}

void THttpServer::TRequest::ReplyPlainText(const TString &reply)
{
    Reply(reply, TVector<char>(), "text/plain");
}

void THttpServer::TRequest::ReplyBin(const TVector<char> &data)
{
    Reply("", data, "application/octet-stream");
}

void THttpServer::TRequest::ReplyBMP(const TVector<char> &data)
{
    Reply("", data, "image/x-MS-bmp");
}

}
