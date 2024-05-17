#pragma once

struct IBinaryStream
{
    enum {
        MAX_BLOCK_SIZE = 1 << 24
    };

	virtual ~IBinaryStream() {};
    yint Write(const void *userBuffer, yint size)
    {
        if (size < MAX_BLOCK_SIZE) {
            return WriteImpl(userBuffer, size);
        } else {
            const char *pData = (const char *)userBuffer;
            yint totalWritten = 0;
            for (yint offset = 0; offset < size;) {
                yint blkSize = Min<yint>(MAX_BLOCK_SIZE, size - offset);
                totalWritten += WriteImpl(pData + offset, blkSize);
                offset += blkSize;
            }
            return totalWritten;
        }
    }
    yint Read(void *userBuffer, yint size)
    {
        if (size < MAX_BLOCK_SIZE) {
            return ReadImpl(userBuffer, size);
        } else {
            char *pData = (char *)userBuffer;
            yint totalRead = 0;
            for (yint offset = 0; offset < size;) {
                yint blkSize = Min<yint>(MAX_BLOCK_SIZE, size - offset);
                totalRead += ReadImpl(pData + offset, blkSize);
                offset += blkSize;
            }
            return totalRead;
        }
    }
    virtual yint WriteImpl(const void *userBuffer, yint size) = 0;
    virtual yint ReadImpl(void *userBuffer, yint size) = 0;
    virtual bool IsValid() const = 0;
	virtual bool IsFailed() const = 0;
};


///////////////////////////////////////////////////////////////////////////////////////////////////
#ifdef _MSC_VER

class TFileStream : public IBinaryStream
{
    HANDLE hFile;
    bool bFailed;
public:
    TFileStream(bool bRead, const TString &szFile) : bFailed(false)
    {
        DWORD dwAccess = 0, dwCreate = 0;
        if (bRead) {
            dwAccess = GENERIC_READ;
            dwCreate = OPEN_EXISTING;
        } else {
            dwAccess = GENERIC_WRITE;
            dwCreate = OPEN_ALWAYS;
        }
        hFile = CreateFileA(szFile.c_str(), dwAccess, FILE_SHARE_READ, 0, dwCreate, FILE_ATTRIBUTE_NORMAL, 0);
    }
    ~TFileStream()
    {
        if (hFile != INVALID_HANDLE_VALUE) {
            CloseHandle(hFile);
        }
    }
    yint WriteImpl(const void *pData, yint size)
    {
        DWORD nWritten = 0;
        BOOL b = WriteFile(hFile, pData, size, &nWritten, 0);
        if (!b) {
            bFailed = true;
        }
        return nWritten;
    }
    yint ReadImpl(void *pData, yint size)
    {
        DWORD nRead = 0;
        BOOL b = ReadFile(hFile, pData, size, &nRead, 0);
        if (!b) {
            bFailed = true;
        }
        return nRead;
    }
    yint GetLength()
    {
        LARGE_INTEGER nLeng;
        GetFileSizeEx(hFile, (PLARGE_INTEGER)&nLeng);
        return nLeng.QuadPart;
    }
    yint Seek(yint pos)
    {
        LARGE_INTEGER i;
        i.QuadPart = pos;
        i.LowPart = SetFilePointer(hFile, i.LowPart, &i.HighPart, FILE_BEGIN);
        return i.QuadPart;
    }
    void Flush()
    {
    }
    bool IsValid() const { return hFile != INVALID_HANDLE_VALUE; }
    bool IsFailed() const { return bFailed; }
};

#else

class TFileStream : public IBinaryStream
{
    FILE *File = 0;
public:
    TFileStream(bool bRead, const TString &szFile)
    {
        File = fopen(szFile.c_str(), bRead ? "rb" : "wb");
    }
    ~TFileStream()
    {
        if (File) {
            fclose(File);
        }
    }
    yint WriteImpl(const void *pData, yint size)
    {
        if (File) {
            yint written = fwrite(pData, 1, size, File);
            if (written != size) {
                fclose(File);
                File = 0;
            }
            return written;
        }
        return 0;
    }
    yint ReadImpl(void *pData, yint size)
    {
        if (File) {
            yint readCount = fread(pData, 1, size, File);
            if (readCount != size) {
                fclose(File);
                File = 0;
            }
            return readCount;
        }
        return 0;
    }
    yint GetLength()
    {
        yint pos = ftello(File);
        fseeko(File, 0, SEEK_END);
        yint fsize = ftello(File);
        fseeko(File, pos, SEEK_SET);
        return fsize;
    }
    yint Seek(yint pos)
    {
        fseeko(File, pos, SEEK_SET);
        return ftello(File);
    }
    void Flush()
    {
    }
    bool IsValid() const { return File != 0; }
    bool IsFailed() const { return File != 0; } // no recovery after fail so we just close the file in such case
};

#endif


///////////////////////////////////////////////////////////////////////////////////////////////////
class TBufferedStream : public TNonCopyable
{
    enum { BUF_SIZE = 1 << 20 };
    TVector<char> Buf;
    IBinaryStream &Stream;
    yint Pos, BufSize;
    bool IsReading;
    bool IsEof;

    void ReadLarge(void *userBuffer, yint size);
    void WriteLarge(const void *userBuffer, yint size);
    void Flush();
public:
    TBufferedStream(IBinaryStream &stream, bool isReading) : Stream(stream), Pos(0), BufSize(0), IsReading(isReading), IsEof(false)
    {
        Buf.yresize(BUF_SIZE);
    }
    ~TBufferedStream()
    {
        if (!IsReading) {
            Flush();
        }
    }
    inline void Read(void *userBuffer, yint size)
    {
        Y_ASSERT(IsReading);
        if (!IsEof && size + Pos <= BufSize) {
            memcpy(userBuffer, Buf.data() + Pos, size);
            Pos += size;
        } else {
            ReadLarge(userBuffer, size);
        }
    }
    inline void Write(const void *userBuffer, yint size)
    {
        Y_ASSERT(!IsReading);
        if (Pos + size < BUF_SIZE) {
            memcpy(Buf.data() + Pos, userBuffer, size);
            Pos += size;
        } else {
            WriteLarge(userBuffer, size);
        }
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
class TSeqReader
{
    enum { BUF_SIZE = 1 << 16 };
    TVector<char> Buf;
    yint Pos, BufSize;
    TFileStream F;
    bool IsEofFlag;
public:
    TSeqReader(const TString &szFile) : Pos(0), BufSize(0), F(true, szFile), IsEofFlag(false)
    {
        Buf.resize(BUF_SIZE);
    }
    TString ReadLine()
    {
        TString szRes;
        for(;;++Pos) {
            if (Pos == BufSize) {
                Pos = 0;
                BufSize = F.Read(Buf.data(), BUF_SIZE);
                if (BufSize == 0) {
                    IsEofFlag = true;
                    break;
                }
            }
            if (Buf[Pos] == '\x0d')
                continue;
            if (Buf[Pos] == '\x0a') {
                ++Pos;
                break;
            }
            szRes += Buf[Pos];
        }
        return szRes;
    }
    bool IsEof() const { return IsEofFlag; }
    bool IsValid() const { return F.IsValid(); }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
void ReadWholeFile(const TString &szFileName, TVector<char> *res);
