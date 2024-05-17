#pragma once
#include "bin_saver.h"


class TMemStream : public IBinaryStream
{
    TVector<ui8> Data;
    yint Pos = 0;

private:
    yint WriteImpl(const void *userBuffer, yint size) override
    {
        if (size == 0) {
            return 0;
        }
        if (Pos + size > YSize(Data)) {
            Data.yresize(Pos + size);
        }
        memcpy(Data.data() + Pos, userBuffer, size);
        Pos += size;
        return size;
    }
    yint ReadImpl(void *userBuffer, yint size) override
    {
        yint res = Min<yint>(YSize(Data) - Pos, size);
        if (res > 0) {
            memcpy(userBuffer, &Data[Pos], res);
            Pos += res;
        }
        return res;
    }
    bool IsValid() const override
    {
        return true;
    }
    bool IsFailed() const override
    {
        return false;
    }

public:
    TMemStream() : Pos(0)
    {
    }

    TMemStream(TVector<ui8> *data) : Pos(0)
    {
        Data.swap(*data);
    }

    void ExtractData(TVector<ui8> *data)
    {
        data->swap(Data);
        Pos = 0;
        Data.clear();
    }
};


template<class T>
inline void SerializeMem(bool bRead, TVector<ui8> *data, T &c)
{
    if (IBinSaver::HasTrivialSerializer(&c)) {
        if (bRead) {
            Y_ASSERT(data->size() == sizeof(T));
            c = *reinterpret_cast<T *>(data->data());
        } else {
            data->yresize(sizeof(T));
            *reinterpret_cast<T *>(data->data()) = c;
        }
    } else {
        TMemStream f(data);
        {
            IBinSaver bs(f, bRead);
            bs.Add(&c);
        }
        f.ExtractData(data);
    }
}

