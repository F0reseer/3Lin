#pragma once


///////////////////////////////////////////////////////////////////////////////////////////////////
template <int WSZ>
__device__ void KVProductImpl(int blk, const float *key, const float *value, i8 *dst)
{
    int h = threadIdx.x;
    for (int k = 0; k < WSZ; ++k) {
        int d = k * WARP_SIZE + h;
        float keyShfl = __shfl_xor_sync(0xffffffff, key[k], blk);
        dst[d] = CvtToI8(keyShfl * value[k] * (VEC_SCALE * VEC_SCALE / VEC_SCALE));
    }
}

template <int WSZ>
__device__ void KVProductImpl(int blk, const float *key, const float *value, half *dst)
{
    int h = threadIdx.x;
    for (int k = 0; k < WSZ; ++k) {
        int d = k * WARP_SIZE + h;
        float keyShfl = __shfl_xor_sync(0xffffffff, key[k], blk);
        dst[d] = keyShfl * value[k] * (VEC_SCALE * VEC_SCALE / VEC_SCALE);
    }
}

template <int STATE_DIM>
__global__ void KVProduct(
    TCuda2DPtr<TVecFloat> kVecArr, TCuda2DPtr<TValueVecFloat> vVecArr,
    TCuda2DPtr<TKVFloat> kvVecArr
)
{
    int t = blockIdx.x;
    constexpr int WSZ = STATE_DIM / WARP_SIZE;
    float value[WSZ];
    LoadWarpVec<WSZ>(value, vVecArr[t]);
    float key[WSZ];
    LoadWarpVec<WSZ>(key, kVecArr[t]);
    for (int blk = 0; blk < COMBINER_REP; ++blk) {
        KVProductImpl<WSZ>(blk, key, value, kvVecArr[t] + blk * STATE_DIM);
    }
}


template <int STATE_DIM>
__global__ void KVProductBackprop(
    TCuda2DPtr<TVecFloat> kVecArr, TCuda2DPtr<TValueVecFloat> vVecArr, TCuda2DPtr<half> dkvVecArr,
    TCuda2DPtr<half> dkVecArr, TCuda2DPtr<half> dvVecArr, float *dScaleArr
)
{
    int h = threadIdx.x;
    int t = blockIdx.x;
    constexpr int WSZ = STATE_DIM / WARP_SIZE;
    float value[WSZ];
    LoadWarpVec<WSZ>(value, vVecArr[t]);
    float dValue[WSZ];
    LoadZeroWarpVec<WSZ>(dValue);
    float key[WSZ];
    LoadWarpVec<WSZ>(key, kVecArr[t]);
    float dKey[WSZ];
    LoadZeroWarpVec<WSZ>(dKey);
    float dScale = 0;
    for (int blk = 0; blk < COMBINER_REP; ++blk) {
        float dkv[WSZ];
        LoadWarpVec<WSZ>(dkv, dkvVecArr[t] + blk * STATE_DIM);
        // forward: kv = shfl(key) * value
        for (int k = 0; k < WSZ; ++k) {
            float keyShfl = __shfl_xor_sync(0xffffffff, key[k] * VEC_SCALE, blk);
            float dKeyShfl = __shfl_xor_sync(0xffffffff, dkv[k] * value[k] * VEC_SCALE, blk); // reverse shuffle
            dKey[k] += dKeyShfl;
            dValue[k] += dkv[k] * keyShfl;
            dScale += dkv[k] * keyShfl * value[k] * VEC_SCALE;
        }
    }
    StoreWarpVec<WSZ>(dvVecArr[t], dValue);
    StoreWarpVec<WSZ>(dkVecArr[t], dKey);
    dScale = WarpSum(dScale);
    if (h == 0) {
        dScaleArr[t] = dScale;
    }
}
