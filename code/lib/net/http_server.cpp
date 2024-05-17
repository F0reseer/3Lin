#include "stdafx.h"
#include <errno.h>
#include "http_server.h"
#include "http_request.h"
#include "ip_address.h"
//#include <ws2ipdef.h>

namespace NNet
{
static string FormHeader(const char *type, const char *encoding = NULL)
{
    string reply;
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

void ReplyBadRequest(SOCKET s)
{
    char reply[] = "HTTP/1.0 400 Bad request\r\nConnection: close\r\n\r\n";
    if (!SendRetry(s, reply, (int)strlen(reply)))
        return;
    CloseConnection(s);
}

void ReplyNotFound(SOCKET s)
{
    char reply[] = "HTTP/1.0 404 Not Found\r\nConnection: close\r\n\r\n";
    if (!SendRetry(s, reply, (int)strlen(reply)))
        return;
    CloseConnection(s);
}

void CloseConnection(SOCKET s)
{
#ifdef _win_
    shutdown(s, SD_SEND);
#else
    shutdown(s, SHUT_WR);
#endif
    closesocket(s);
}

bool SendRetry(SOCKET s, const char *buf, yint len)
{
    int flags = 0;
#if (!defined(_win_) && !defined(_darwin_))
    flags = MSG_NOSIGNAL;
#endif

    while (len > 0) {
        yint res = send(s, buf, len, flags);
        if (res < 0) {
            CloseConnection(s);
            return false;
        }
        buf += res;
        len -= res;
    }
    return true;
}

bool SendHeader(SOCKET s, const char *type, const char *encoding)
{
    string header = FormHeader(type, encoding);
    return SendRetry(s, header.c_str(), (int)header.length());
}

////////////////////////////////////////////////////////////////////////////////

static bool StartListen(SOCKET *psAccept, int *port)
{
    SOCKET &sAccept = *psAccept;
    if (sAccept != INVALID_SOCKET)
        closesocket(sAccept);
    //
    sAccept = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (sAccept == INVALID_SOCKET)
        return false;
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
        ASSERT(0);
        closesocket(sAccept);
        sAccept = INVALID_SOCKET;
        return false;
    }
    if (listen(sAccept, SOMAXCONN) != 0) {
        ASSERT(0);
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

THttpServer::THttpServer(int _nAcceptPort)
    : sAccept(INVALID_SOCKET), nAcceptPort(_nAcceptPort)
{
    if (!StartListen(&sAccept, &nAcceptPort)) {
        fprintf(stderr, "StartListen() failed: %s\n", strerror(errno));
    }
    FD_ZERO(&FS);
}


THttpServer::~THttpServer()
{
    if (sAccept != INVALID_SOCKET) {
        closesocket(sAccept);
        sAccept = INVALID_SOCKET;
    }
}


THttpServer::EReadReqResult THttpServer::ReadRequestData(SOCKET s, TRecvRequestBuffer *buf)
{
    fd_set fs;
    FD_ZERO(&fs);
    FD_SET(s, &fs);
    timeval timeout = {0, 0};
    if (select(s + 1, &fs, 0, &fs, &timeout) == 0) {
        return WAIT;
    }
    int rv = recv(s, buf->Buffer + buf->Offset, (int)ARRAY_SIZE(buf->Buffer) - 1 - buf->Offset, 0);
    if (rv == SOCKET_ERROR) {
        fprintf(stderr, "recv() error: %s\n", strerror(errno));
        CloseConnection(s);
        return FAILED;
    } else if (rv == 0) {  // peer closed connection gracefully
        CloseConnection(s);
        return CLOSED;
    }

    buf->Buffer[buf->Offset + rv] = 0;
    buf->Offset += rv;
    const char *hdrFin = strstr(buf->Buffer, "\r\n\r\n");
    if (hdrFin == 0) {
        return WAIT;
    }
    return OK;
}


bool THttpServer::ReadRequest(SOCKET s, TRecvRequestBuffer *buf, THttpRequest *pReq)
{
    const char *hdrFin = strstr(buf->Buffer, "\r\n\r\n");
    ASSERT(hdrFin != 0);
    char *pszRequest = GetRequest(buf->Buffer);
    if (!pszRequest || !ParseRequest(pReq, pszRequest)) {
        ReplyBadRequest(s);
        return false;
    }

    int dataStart = (int)(hdrFin - buf->Buffer + 4);
    pReq->Data.resize(0);
    int rv = buf->Offset;
    if (rv != dataStart || rv >= (int)ARRAY_SIZE(buf->Buffer) - 1) {
        for (;rv != 0;) {
            yint start = YSize(pReq->Data), dataLen = rv - dataStart;
            if (dataLen > 0) {
                pReq->Data.resize(start + dataLen);
                memcpy(&pReq->Data[start], buf->Buffer + dataStart, dataLen);
            }
            rv = recv(s, buf->Buffer, ARRAY_SIZE(buf->Buffer), 0);
            if (rv == SOCKET_ERROR) {
                ReplyBadRequest(s);
                return false;
            }
            dataStart = 0;
        }
    }
    return true;
}


bool THttpServer::CanAccept(float timeoutSec)
{
    FD_ZERO(&FS);
    SOCKET maxSocket = 0;
    if (sAccept != INVALID_SOCKET) {
        FD_SET(sAccept, &FS);
        maxSocket = sAccept;
    }

    for (TRecvSocketsHash::iterator i = ReqSockets.begin(); i != ReqSockets.end(); ++i) {
        SOCKET s = i->first;
        FD_SET(s, &FS);
        if (s > maxSocket)
            maxSocket = s;
    }

    timeval tv = MakeTimeval(timeoutSec);
    int rv = select(maxSocket + 1, &FS, 0, 0, &tv);
    Y_ASSERT(rv != SOCKET_ERROR);
    return rv > 0;
}


SOCKET THttpServer::AcceptNonBlockingImpl(THttpRequest *pReq)
{
    for (TRecvSocketsHash::iterator i = ReqSockets.begin(); i != ReqSockets.end(); ) {
        TRecvSocketsHash::iterator k = i++;
        SOCKET s = k->first;
        if (FD_ISSET(s, &FS)) {
            EReadReqResult rr = ReadRequestData(s, &k->second);

            switch (rr) {
            case OK:
                if (ReadRequest(s, &k->second, pReq)) {
                    ReqSockets.erase(k);
                    return s;
                } else {
                    ReqSockets.erase(k);
                }
                break;
            case CLOSED:
            case FAILED:
                ReqSockets.erase(k);
                break;
            case WAIT:
                break;
            }
        }
    }
    if (FD_ISSET(sAccept, &FS)) {
        //sockaddr_in6 incomingAddr;
        //int nIncomingAddrLen = sizeof(incomingAddr);
        SOCKET s = accept(sAccept, 0, 0);//(sockaddr*)&incomingAddr, &nIncomingAddrLen);
        if (s == INVALID_SOCKET) {
            if (!StartListen(&sAccept, &nAcceptPort)) {
                fprintf(stderr, "StartListen() failed: %s\n", strerror(errno));
            }
            return INVALID_SOCKET;
        }

        TRecvRequestBuffer buf;
        EReadReqResult rr = ReadRequestData(s, &buf);
        switch (rr) {
        case OK:
            if (ReadRequest(s, &buf, pReq)) {
                return s;
            }
            break;
        case WAIT:
            ReqSockets[s] = buf;
            break;
        case CLOSED:
        case FAILED:
            break;
        }
    }
    return INVALID_SOCKET;
}


SOCKET THttpServer::AcceptNonBlocking(THttpRequest *pReq)
{
    SOCKET s = AcceptNonBlockingImpl(pReq);
    FD_ZERO(&FS);
    return s;
}

////////////////////////////////////////////////////////////////////////////////

static void SendReply(SOCKET s, const string &reply, const TVector<char> &data)
{
    if (!SendRetry(s, reply.c_str(), (int)reply.length()))
        return;

    if (!data.empty()) {
        if (!SendRetry(s, data.begin(), YSize(data)))
            return;
    }
    CloseConnection(s);
}

static void Reply(SOCKET s, const string &reply, const TVector<char> &data, const char *content)
{
    string fullReply = FormHeader(content) + reply;
    SendReply(s, fullReply, data);
}

void HttpReplyXML(SOCKET s, const string &reply)
{
    Reply(s, reply, TVector<char>(), "text/xml");
}

void HttpReplyHTML(SOCKET s, const string &reply)
{
    Reply(s, reply, TVector<char>(), "text/html");
}

void HttpReplyPlainText(SOCKET s, const string &reply)
{
    Reply(s, reply, TVector<char>(), "text/plain");
}

void HttpReplyBin(SOCKET s, const TVector<char> &data)
{
    Reply(s, "", data, "application/octet-stream");
}

void HttpReplyBMP(SOCKET s, const TVector<char> &data)
{
    Reply(s, "", data, "image/x-MS-bmp");
}

}
