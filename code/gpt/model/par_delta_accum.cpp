#include "stdafx.h"
#include "par_delta_accum.h"
#include <immintrin.h>

using namespace NCuda;

///////////////////////////////////////////////////////////////////////////////////////////////////
template <class TDst, class TSrc>
static void AddRow(TDst &dst, const TSrc &src, yint y)
{
    yint xSize = src.GetXSize();
    Y_ASSERT(xSize == dst.GetXSize());
    __m256 *dstPtr = (__m256 *) & dst[y][0];
    const __m256 *srcPtr = (const __m256 *) & src[y][0];
    for (int x8 = 0; x8 < xSize / 8; ++x8) {
        dstPtr[x8] = _mm256_add_ps(dstPtr[x8], srcPtr[x8]);
    }
}


template <class TDst>
static void ClearRow(TDst &dst, yint y)
{
    yint xSize = dst.GetXSize();
    __m256 *dstPtr = (__m256 *) & dst[y][0];
    __m256 zero = _mm256_setzero_ps();
    for (int x8 = 0; x8 < xSize / 8; ++x8) {
        dstPtr[x8] = zero;
    }
}


void TMMDeltaAccumulate::OnDelta()
{
    NCuda::TCuda2DArray<float> &delta = ModelMatrix->GetSumDelta();
    yint ySize = delta.GetYSize();

    if (AddToModel == GRADIENT_ACCUMULATE) {
        auto hostDeltaMem = delta.GetHostPtr();
        if (!HasAccumulatedDelta) {
            // copy delta
            if (ModelMatrix->HasRowDisp()) {
                if (NonzeroRowFlag.empty()) {
                    ClearPodArray(&NonzeroRowFlag, ySize);
                    DeltaAccum.SetSizes(delta.GetXSize(), delta.GetYSize());
                    DeltaAccum.FillZero();
                }
                const TVector<ui32> &nonzeroRows = ModelMatrix->GetSumNonzeroRows();
                for (yint y : nonzeroRows) {
                    AddRow(DeltaAccum, hostDeltaMem, y);
                    NonzeroRowFlag[y] = 1;
                }
            } else {
                delta.GetAllData(&DeltaAccum);
            }
            HasAccumulatedDelta = true;
        } else {
            if (ModelMatrix->HasRowDisp()) {
                const TVector<ui32> &nonzeroRows = ModelMatrix->GetSumNonzeroRows();
                for (yint y : nonzeroRows) {
                    AddRow(DeltaAccum, hostDeltaMem, y);
                    NonzeroRowFlag[y] = 1;
                }
            } else {
                for (yint y = 0; y < ySize; ++y) {
                    AddRow(DeltaAccum, hostDeltaMem, y);
                }
            }
        }
        ModelMatrix->SetOp(TModelMatrix::OP_NONE);

    } else if (AddToModel == GRADIENT_APPLY) {
        if (HasAccumulatedDelta) {
            auto hostDeltaMem = delta.GetHostPtr();
            if (ModelMatrix->HasRowDisp()) {
                const TVector<ui32> &nonzeroRows = ModelMatrix->GetSumNonzeroRows();
                for (yint y : nonzeroRows) {
                    NonzeroRowFlag[y] = 1;
                }
                TVector<ui32> newRows;
                for (yint y = 0; y < ySize; ++y) {
                    if (NonzeroRowFlag[y]) {
                        AddRow(hostDeltaMem, DeltaAccum, y);
                        ClearRow(DeltaAccum, y);
                        newRows.push_back(y);
                    }
                }
                ClearPodArray(&NonzeroRowFlag, ySize);
                ModelMatrix->SetSumNonzeroRows(newRows);
            } else {
                for (yint y = 0; y < ySize; ++y) {
                    AddRow(hostDeltaMem, DeltaAccum, y);
                }
            }
            HasAccumulatedDelta = false;
        }
        ModelMatrix->SetOp(TModelMatrix::OP_ADD_DELTA);

    } else {
        Y_VERIFY(0 && "unknown op");
    }
}

