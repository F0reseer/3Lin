#pragma once
#include "bin_saver.h"


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
            TBufferedStream bufIO(f, bRead);
            IBinSaver bs(bufIO);
            bs.Add(&c);
        }
        f.Swap(data);
    }
}
