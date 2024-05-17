#pragma once

namespace NNet
{
struct THttpRequest
{
    TString Req;
    TVector<char> Data;
    THashMap<TString, TString> params;

    bool HasParam(const char *pszParam) const
    {
        THashMap<TString, TString>::const_iterator i = params.find(pszParam);
        return (i != params.end());
    }

    TString GetParam(const char *pszParam) const
    {
        THashMap<TString, TString>::const_iterator i = params.find(pszParam);
        if (i == params.end())
            return "";
        return i->second;
    }

    int GetIntParam(const char *pszParam) const
    {
        TString sz = GetParam(pszParam);
        return atoi(sz.c_str());
    }

    bool GetBoolParam(const char *pszParam) const
    {
        TString sz = GetParam(pszParam);
        return sz == "yes" || sz == "true" || atoi(sz.c_str()) == 1;
    }

    TString GetUrl() const
    {
        TString res = "/" + Req;
        bool first = true;
        for (THashMap<TString, TString>::const_iterator i = params.begin();
            i != params.end(); ++i) {
            if (first)
                res += "?" + i->first + "=" + i->second;
            else
                res += "&" + i->first + "=" + i->second;
            first = false;
        }
        return res;
    }
};

bool ParseRequest(THttpRequest *pRes, const char *pszReq);

TString EncodeCGI(const TString &arg);
TString DecodeCGI(const TString &x);
}
