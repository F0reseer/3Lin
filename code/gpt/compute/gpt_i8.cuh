#pragma once

///////////////////////////////////////////////////////////////////////////////////////////////////
//

template <int STATE_DIM, class TDst>
__global__ void ConvertHalfToVecFloat(TCuda2DPtr<half> src, float staticScale, float *pSrcScale, TCuda2DPtr<TDst> dst8, float *scaleArr)
{
    int h = threadIdx.x;
    int t = blockIdx.x;
    // each vector has its own scale
    constexpr int WSZ = STATE_DIM / WARP_SIZE;
    float v[WSZ];
    float sum2 = 0;
    for (int k = 0; k < WSZ; ++k) {
        int d = k * WARP_SIZE + h;
        float val = src[t][d];
        v[k] = val;
        sum2 += val * val;
    }
    sum2 = WarpSum(sum2);
    if (sum2 > 0) {
        float sko = sqrtf(sum2 / STATE_DIM);
        float discrScale = sko * VEC_SCALE;
        for (int k = 0; k < WSZ; ++k) {
            TVecFloat res;
            CvtToVecFloat(&res, v[k] / discrScale);
            int d = k * WARP_SIZE + h;
            dst8[t][d] = res;
        }
        if (scaleArr && h == 0) {
            if (pSrcScale) {
                scaleArr[t] = discrScale * *pSrcScale * staticScale;
            } else {
                scaleArr[t] = discrScale * staticScale;
            }
        }
    } else {
        for (int k = 0; k < WSZ; ++k) {
            int d = k * WARP_SIZE + h;
            dst8[t][d] = 0;
        }
        if (scaleArr && h == 0) {
            scaleArr[t] = 0;
        }
    }
}


template <int STATE_DIM, class TElemType>
__global__ void BackpropNormalizeVecs8(TCuda2DPtr<TElemType> srcNorm8, float *srcScale, TCuda2DPtr<half> grad, TCuda2DPtr<half> dst)
{
    int t = blockIdx.x;
    constexpr int WSZ = STATE_DIM / WARP_SIZE;

    // normalize
    float v[WSZ];
    LoadWarpVec<WSZ>(v, srcNorm8[t]);
    ScaleWarpVec<WSZ>(v, VEC_SCALE * srcScale[t]);
    float vGrad[WSZ];
    LoadWarpVec<WSZ>(vGrad, grad[t]);
    float stateGrad[WSZ];
    StateNormalizeBackpropWarpVec<WSZ>(v, vGrad, stateGrad);
    StoreWarpVec<WSZ>(dst[t], stateGrad);
}
