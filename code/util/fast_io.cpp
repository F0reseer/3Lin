#include "stdafx.h"
#include "fast_io.h"


void TBufferedStream::ReadLarge(void *userBuffer, yint size)
{
    if (IsEof) {
        memset(userBuffer, 0, size);
        return;
    }
    char *dst = (char *)userBuffer;
    yint leftBytes = BufSize - Pos;
    memcpy(dst, Buf.data() + Pos, leftBytes);
    dst += leftBytes;
    size -= leftBytes;
    Pos = 0;
    BufSize = 0;
    if (size > BUF_SIZE) {
        yint n = Stream.Read(dst, size);
        if (n != size) {
            IsEof = true;
            memset(dst + n, 0, size - n);
        }
    } else {
        BufSize = Stream.Read(Buf.data(), BUF_SIZE);
        if (BufSize == 0) {
            IsEof = true;
        }
        Read(dst, size);
    }
}


void TBufferedStream::Flush()
{
    Stream.Write(Buf.data(), Pos);
    Pos = 0;
}


void TBufferedStream::WriteLarge(const void *userBuffer, yint size)
{
    Flush();
    if (size >= BUF_SIZE) {
        Stream.Write(userBuffer, size);
    } else {
        Write(userBuffer, size);
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
void ReadWholeFile(const TString &szFileName, TVector<char> *res)
{
    res->resize(0);
    TFileStream fs(true, szFileName);
    if (fs.IsValid()) {
        yint sz = fs.GetLength();
        res->yresize(sz);
        yint readCount = fs.Read(&(*res)[0], sz);
        if (readCount != sz) {
            res->resize(0);
        }
    }
}

