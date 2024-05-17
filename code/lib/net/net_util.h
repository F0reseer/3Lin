#pragma once

#ifdef _win_
#define pollfd WSAPOLLFD
#define poll WSAPoll
#else
#include <poll.h>
#endif

namespace NNet
{
void MakeNonBlocking(SOCKET s);
}
