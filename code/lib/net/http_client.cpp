#include "stdafx.h"
#include "http_client.h"
#include "ip_address.h"

namespace NNet
{
bool Fetch(const char *pszHost, const char *pszRequest, const vector<char> &reqData, vector<char> *reply)
{
    //printf("Fetch, host=%s, url=%s\n", pszHost, pszRequest);

    reply->resize(0);

    SOCKET s;
    s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (s == INVALID_SOCKET)
        return false;

    sockaddr_in dst;
    string szPureHost;
    if (!ParseInetName(&dst, &szPureHost, pszHost, 80)) {
        closesocket(s);
        return false;
    }

    if (connect(s, (sockaddr*)&dst, sizeof(dst)) == SOCKET_ERROR) {
        closesocket(s);
        return false;
    }

    string szRequest = reqData.empty() ? string("GET ") : string("POST ");
    szRequest += string(pszRequest) + " HTTP/1.1\r\n";
    szRequest += string("Host: ") + szPureHost + "\r\n";
    szRequest += "Connection: close\r\n";
    //szRequest += "Accept-Encoding: gzip\r\n";
    //szRequest += "Accept: image/gif, image/x-xbitmap, image/jpeg, image/pjpeg, application/x-shockwave-flash, application/vnd.ms-excel, application/vnd.ms-powerpoint, application/msword, */*\r\n";
    //szRequest += "Accept-Language: ru\r\n";
    szRequest += "User-Agent: ya, andrey.gulin@gmail.com\r\n";
    //szRequest += "Referer: http://web-sniffer.net/\r\n";
    szRequest += "\r\n";
    if (!reqData.empty()) {
        int headerSize = szRequest.size();
        szRequest.resize(headerSize + reqData.size());
        memcpy(&szRequest[headerSize], &reqData[0], reqData.size());
    }

    int nSent = send(s, &szRequest[0], (int)szRequest.size(), 0);
    if (nSent != szRequest.size()) {
        closesocket(s);
        return false;
    }
    //shutdown(s, SD_SEND); // this actually breaks LJ request for some reason, not required actually?

    string szRes;
    int nRecv = SOCKET_ERROR;
    for (;;) {
        char szRecvBuf[2048];
        nRecv = recv(s, szRecvBuf, 2048, 0);
        if (nRecv == 0)
            break;
        if (nRecv == SOCKET_ERROR) {
            closesocket(s);
            return false;
        }
        int nStart = (int)szRes.size();
        szRes.resize(nStart + nRecv);
        memcpy(&szRes[nStart], szRecvBuf, nRecv);
    }
    closesocket(s);
    size_t nHeaderEnd = szRes.find("\r\n\r\n");
    if (nHeaderEnd == string::npos)
        return false;

    // HTTP/1.0 200 OK?
    const char *pszHeader = szRes.c_str();
    if (strncmp(pszHeader, "HTTP", 4) != 0)
        return false;
    while (*pszHeader && pszHeader[0] != ' ')
        ++pszHeader;
    const char *pszOkString = " 200 OK";
    if (strncmp(pszHeader, pszOkString, strlen(pszOkString)) != 0)
        return false;

    size_t dataStart = nHeaderEnd + 4;
    if (dataStart > szRes.size())
        return false;

    reply->resize(szRes.size() - dataStart);
    if (!reply->empty())
        memcpy(&(*reply)[0], &szRes[dataStart], reply->size());
    return true;
}
}
