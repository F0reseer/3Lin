#include "stdafx.h"
#include "ip_address.h"

namespace NNet
{
timeval MakeTimeval(float timeoutSec)
{
    int timeoutSecint = timeoutSec;
    timeval tvTimeout = { timeoutSecint, static_cast<long>((timeoutSec - timeoutSecint) * 1000000) };
    return tvTimeout;
}


bool ParseInetName(sockaddr_in *pName, TString *pszPureHost, const TString &szAddress, int nDefaultPort)
{
    int nPort = nDefaultPort;
    Zero(*pName);

    pName->sin_family = AF_INET;
    // extract port number from address, will not work for IPv6 addresses?
    TString &szPureHost = *pszPureHost;
    szPureHost = szAddress;
    size_t nIdx = szPureHost.find( ':' );
    if (nIdx != TString::npos) {
        const char *pszPort = szPureHost.c_str() + nIdx + 1;
        nPort = atoi(pszPort);
        szPureHost.resize(nIdx);
    }
    // determine host
    pName->sin_addr.s_addr = inet_addr(szPureHost.c_str());
    if (pName->sin_addr.s_addr == INADDR_NONE) { // not resolved?
        hostent *he;
        he = gethostbyname(szPureHost.c_str()); // m.b. it is string comp.domain
        if (he == NULL)
            return false;
        pName->sin_addr.s_addr = *(unsigned long *)(he->h_addr_list[0]);
    }
    pName->sin_port = htons(nPort);
    return true;
}


void ReplacePort(TString *pAddr, int newPort)
{
    // extract port number from address, will not work for IPv6 addresses?
    size_t nIdx = pAddr->find(':');
    if (nIdx != TString::npos) {
        pAddr->resize(nIdx);
    }
    *pAddr += Sprintf(":%d", newPort);
}


TString GetHostName()
{
    char buf[10000];
    if (gethostname(buf, ARRAY_SIZE(buf) - 1)) {
        Y_ASSERT(0);
        return "???";
    }
    return buf;
}

}
