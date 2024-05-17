#pragma once


///////////////////////////////////////////////////////////////////////////////////////////////////
template <int STATE_DIM>
__device__ void KVProductImpl(int blk, const float *key, const float *value, i8 *dst)
{
    int h = threadIdx.x;
    for (int k = 0; k < WCOUNT; ++k) {
        int d = k * WARP_SIZE + h;
        float keyShfl = __shfl_xor_sync(0xffffffff, key[k], blk);
        dst[d] = CvtToI8(keyShfl * value[k] * (VEC_SCALE * VEC_SCALE / VEC_SCALE));
    }
}

template <int STATE_DIM>
__device__ void KVProductImpl(int blk, const float *key, const float *value, half *dst)
{
    int h = threadIdx.x;
    for (int k = 0; k < WCOUNT; ++k) {
        int d = k * WARP_SIZE + h;
        float keyShfl = __shfl_xor_sync(0xffffffff, key[k], blk);
        dst[d] = keyShfl * value[k] * (VEC_SCALE * VEC_SCALE / VEC_SCALE);
    }
}

template <int STATE_DIM>
__global__ void KVProduct(
    int len, TCuda2DPtr<TVecFloat> kVecArr, TCuda2DPtr<TValueVecFloat> vVecArr,
    TCuda2DPtr<TKVFloat> kvVecArr
)
{
    int t = blockIdx.x * SAMPLE_BLOCK + threadIdx.y;
    if (t < len) {
        float value[WCOUNT];
        LoadVec<STATE_DIM>(value, vVecArr[t]);
        float key[WCOUNT];
        LoadVec<STATE_DIM>(key, kVecArr[t]);
        for (int blk = 0; blk < COMBINER_REP; ++blk) {
            KVProductImpl<STATE_DIM>(blk, key, value, kvVecArr[t] + blk * STATE_DIM);
        }
    }
}
KERNEL_BLOCK_SIZE(KVProduct, WARP_SIZE, SAMPLE_BLOCK);


template <int STATE_DIM>
__global__ void KVProductBackprop(
    int len, TCuda2DPtr<TVecFloat> kVecArr, TCuda2DPtr<TValueVecFloat> vVecArr, TCuda2DPtr<half> dkvVecArr,
    TCuda2DPtr<half> dkVecArr, TCuda2DPtr<half> dvVecArr, float *dScaleArr
)
{
    int h = threadIdx.x;
    int t = blockIdx.x * SAMPLE_BLOCK + threadIdx.y;
    if (t < len) {
        float value[WCOUNT];
        LoadVec<STATE_DIM>(value, vVecArr[t]);
        float dValue[WCOUNT];
        LoadZeroVec<STATE_DIM>(dValue);
        float key[WCOUNT];
        LoadVec<STATE_DIM>(key, kVecArr[t]);
        float dKey[WCOUNT];
        LoadZeroVec<STATE_DIM>(dKey);
        float dScale = 0;
        for (int blk = 0; blk < COMBINER_REP; ++blk) {
            float dkv[WCOUNT];
            LoadVec<STATE_DIM>(dkv, dkvVecArr[t] + blk * STATE_DIM);
            // forward: kv = shfl(key) * value
            for (int k = 0; k < WCOUNT; ++k) {
                float keyShfl = __shfl_xor_sync(0xffffffff, key[k] * VEC_SCALE, blk);
                float dKeyShfl = __shfl_xor_sync(0xffffffff, dkv[k] * value[k] * VEC_SCALE, blk); // reverse shuffle
                dKey[k] += dKeyShfl;
                dValue[k] += dkv[k] * keyShfl;
                dScale += dkv[k] * keyShfl * value[k] * VEC_SCALE;
            }
        }
        StoreVec<STATE_DIM>(dvVecArr[t], dValue);
        StoreVec<STATE_DIM>(dkVecArr[t], dKey);
        dScale = WarpSum(dScale);
        if (h == 0) {
            dScaleArr[t] = dScale;
        }
    }
}
KERNEL_BLOCK_SIZE(KVProductBackprop, WARP_SIZE, SAMPLE_BLOCK);
