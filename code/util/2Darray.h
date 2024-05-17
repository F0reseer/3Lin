#pragma once

template < class T > 
struct CBoundCheck
{
	T *data;
	yint nSize;
	CBoundCheck(T *d, yint nS) { data = d; nSize = nS; }
	T& operator[](yint i) const { Y_ASSERT(i>=0 && i<nSize); return data[i]; }
};


template < class T >
class TArray2D
{
	typedef T *PT;
	T *data;
	T **pData;
	yint nXSize, nYSize;

	void Copy(const TArray2D &a)
    {
        if (this == &a) {
            return;
        }
        nXSize = a.nXSize;
        nYSize = a.nYSize;
        Create();
        for (yint i = 0; i < nXSize * nYSize; i++) {
            data[i] = a.data[i];
        }
    }

	void Destroy()
    {
        delete[] data;
        delete[] pData;
    }

    void Create()
    {
        data = new T[nXSize * nYSize];
        pData = new PT[nYSize];
        for (yint i = 0; i < nYSize; i++) {
            pData[i] = data + i * nXSize;
        }
    }
public:
	TArray2D(yint xsize = 1, yint ysize = 1) { nXSize = xsize; nYSize = ysize; Create(); }
	TArray2D(const TArray2D &a) { Copy(a); }
	TArray2D& operator=(const TArray2D &a) { Destroy(); Copy(a); return *this; }
	~TArray2D() { Destroy(); }
	void SetSizes(yint xsize, yint ysize) { if (nXSize == xsize && nYSize == ysize) return; Destroy(); nXSize = xsize; nYSize = ysize; Create(); }
	void Clear() { SetSizes(1,1); }
#ifdef _DEBUG
	CBoundCheck<T> operator[](yint i) const { Y_ASSERT(i>=0 && i<nYSize); return CBoundCheck<T>(pData[i], nXSize); }
#else
	T* operator[](yint i) const { ASSERT(i>=0 && i<nYSize); return pData[i]; }
#endif
	yint GetXSize() const { return nXSize; }
	yint GetYSize() const { return nYSize; }
	void FillZero() { memset(data, 0, sizeof(T) * nXSize * nYSize); }
	void FillEvery(const T &a) { for (yint i = 0; i < nXSize * nYSize; i++) data[i] = a; }
	void Swap(TArray2D &a) { swap(data, a.data); swap(pData, a.pData); swap(nXSize, a.nXSize); swap(nYSize, a.nYSize); }
};

template <class T>
yint GetXSize(const T &x) { return yint(x.GetXSize()); }
template <class T>
yint GetYSize(const T &x) { return yint(x.GetYSize()); }
