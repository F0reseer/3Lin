#pragma once

namespace NNet
{
bool Fetch(const char *pszHost, const char *pszRequest, const vector<char> &reqData, vector<char> *reply);
}
