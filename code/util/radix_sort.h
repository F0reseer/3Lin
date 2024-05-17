#pragma once

inline ui32 GetRadixSortVal(float x)
{
    // if sign bit is 0, flip sign bit
    // if sign bit is 1, flip all bits
    ui32 f = *reinterpret_cast<ui32 *>(&x);
    ui32 mask = -int(f >> 31) | 0x80000000;
    return f ^ mask;
}

template <int Shift, class T, class TGetSortVal>
inline void RadixPass(int *counts, TVector<T> *data, TVector<T> *buf, TGetSortVal op)
{
    yint sz = YSize(*data);
    int offsets[256];
    int sum = 0;
    for (yint bin = 0; bin < 256; ++bin) {
        offsets[bin] = sum;
        sum += counts[bin];
    }
    Y_ASSERT(sum == sz);
    for (yint i = 0; i < sz; ++i) {
        ui32 val = *(ui32 *)op((*data)[i]);
        yint bin = (val >> Shift) & 0xff;
        (*buf)[offsets[bin]++] = (*data)[i];
    }
    data->swap(*buf);
}

// TGetSortVal() returns pointer to the float in struct T, function destroys float it sorts by (applies GetRadixSortVal() in place)
template <class T, class TGetSortVal>
static void RadixFloatSortDescending(TVector<T> *data, TVector<T> *buf, TGetSortVal op)
{
    yint sz = YSize(*data);
    buf->resize(sz);
    int counts[4][256];
    Zero(counts);
    for (yint i = 0; i < sz; ++i) {
        float *p = op((*data)[i]);
        float fVal = *p;
        ui32 val = GetRadixSortVal(-fVal);
        *(ui32 *)p = val; // keep modified sort value
        counts[0][val & 0xff] += 1;
        counts[1][(val >> 8) & 0xff] += 1;
        counts[2][(val >> 16) & 0xff] += 1;
        counts[3][(val >> 24) & 0xff] += 1;
    }
    RadixPass<0>(counts[0], data, buf, op);
    RadixPass<8>(counts[1], data, buf, op);
    RadixPass<16>(counts[2], data, buf, op);
    RadixPass<24>(counts[3], data, buf, op);
}


// TGetSortVal() returns pointer to ui32 value to sort by
template <class T, class TGetSortVal>
static void RadixUI32SortAscending(TVector<T> *data, TVector<T> *buf, TGetSortVal op)
{
    yint sz = YSize(*data);
    buf->resize(sz);
    int counts[4][256];
    Zero(counts);
    for (yint i = 0; i < sz; ++i) {
        ui32 val = *op((*data)[i]);
        counts[0][val & 0xff] += 1;
        counts[1][(val >> 8) & 0xff] += 1;
        counts[2][(val >> 16) & 0xff] += 1;
        counts[3][(val >> 24) & 0xff] += 1;
    }
    RadixPass<0>(counts[0], data, buf, op);
    RadixPass<8>(counts[1], data, buf, op);
    RadixPass<16>(counts[2], data, buf, op);
    RadixPass<24>(counts[3], data, buf, op);
}
