#pragma once

///////////////////////////////////////////////////////////////////////////////////////////////////
// embedding

struct TLabelInverseIndex
{
    struct TLabelPos
    {
        TLabelIndex Label = INVALID_LABEL_INDEX;
        int Pos = -1;

        TLabelPos() {}
        TLabelPos(TLabelIndex label, int pos) : Label(label), Pos(pos) {}
    };

    struct TLabelPosCmp
    {
        bool operator()(const TLabelPos &a, const TLabelPos &b) const
        {
            if (a.Label == b.Label) {
                return a.Pos < b.Pos;
            }
            return a.Label < b.Label;
        }
    };

    TVector<TLabelPos> LabelPosArr;
    TVector<TLabelIndex> InvLabelArr;
    TVector<ui32> InvLabelPos;
    TVector<ui32> InvLabelPosPtr;

    // make list of references for each label
    // for labels with positions in previous iteration and no positions in current iteration create empty list
    void BuildInverseIndex(const TVector<TLabelIndex> &labelArr, const TVector<ui32> &labelPtr)
    {
        // add previous iteration labels with position -1
        yint dst = 0;
        TLabelIndex prevLabel = INVALID_LABEL_INDEX;
        for (yint k = 0; k < YSize(LabelPosArr); ++k) {
            const TLabelPos &labelPos = LabelPosArr[k];
            if (labelPos.Pos >= 0) {
                TLabelIndex label = labelPos.Label;
                if (label != prevLabel) {
                    LabelPosArr[dst++] = TLabelPos(label, -1);
                    prevLabel = label;
                }
            }
        }
        LabelPosArr.resize(dst);

        // add current iteration positions
        for (yint pos = 0; pos < YSize(labelPtr) - 1; ++pos) {
            for (yint k = labelPtr[pos]; k < labelPtr[pos + 1]; ++k) {
                LabelPosArr.push_back(TLabelPos(labelArr[k], pos));
            }
        }
        Sort(LabelPosArr.begin(), LabelPosArr.end(), TLabelPosCmp());

        // make lists
        InvLabelArr.resize(0);
        InvLabelPos.resize(0);
        InvLabelPosPtr.resize(0);
        prevLabel = INVALID_LABEL_INDEX;
        for (yint k = 0; k < YSize(LabelPosArr); ++k) {
            const TLabelPos &labelPos = LabelPosArr[k];
            TLabelIndex label = labelPos.Label;
            if (label != prevLabel) {
                InvLabelPosPtr.push_back(YSize(InvLabelPos));
                InvLabelArr.push_back(label);
                prevLabel = label;
            }
            if (labelPos.Pos >= 0) {
                InvLabelPos.push_back(labelPos.Pos);
            }
        }
        InvLabelPosPtr.push_back(YSize(InvLabelPos));
    }
};


template <int STATE_DIM>
__global__ void AddEmbeddings(
    TLabelIndex *labelArr, ui32 *labelPtr, TCuda2DPtr<TFastMatrixFloat> labelEmbedding, float *labelEmbeddingScale,
    TCuda2DPtr<TStateFloat> state
)
{
    int t = blockIdx.x;

    int start = labelPtr[t];
    int finish = labelPtr[t + 1];
    if (start == finish) {
        StoreZeroVec<STATE_DIM>(state[t]);
    } else {
        float v[WCOUNT];
        LoadVec<STATE_DIM>(v, labelEmbedding[labelArr[start]]);
        for (int z = start + 1; z < finish; ++z) {
            LoadVecAdd<STATE_DIM>(v, v, labelEmbedding[labelArr[z]]);
        }
        ScaleVec<STATE_DIM>(v, v, *labelEmbeddingScale);
        StoreVec<STATE_DIM>(state[t], v);
    }
}


template <int STATE_DIM>
__global__ void BackpropEmbeddings(
    TLabelIndex *invLabelArr, ui32 *invLabelPos, ui32 *invLabelPosPtr,
    TCuda2DPtr<half> stateGrad, float *stateGradScale,
    TCuda2DPtr<float> deltaLabelEmbedding
)
{
    int labelId = blockIdx.x;
    int start = invLabelPosPtr[labelId];
    int finish = invLabelPosPtr[labelId + 1];
    float delta[WCOUNT];
    LoadZeroVec<STATE_DIM>(delta);
    for (int z = start; z < finish; ++z) {
        LoadVecAdd<STATE_DIM>(delta, delta, stateGrad[invLabelPos[z]]);
    }
    ScaleVec<STATE_DIM>(delta, delta, *stateGradScale);
    StoreVec<STATE_DIM>(deltaLabelEmbedding[invLabelArr[labelId]], delta);
    __threadfence_system(); // neccessary?
}
