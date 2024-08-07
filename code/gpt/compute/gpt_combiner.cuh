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


constexpr int KV_PRODUCT_TRANSP_BLOCK_SIZE = 8;
template <int STATE_DIM>
__global__ void KVProductShuffleTranspose(
    int len, TSortNode *sortNode,
    TCuda2DPtr<TVecFloat> kVecArr, TCuda2DPtr<TValueVecFloat> vVecArr,
    TCuda2DPtr<i8> kvVecTArr
)
{
    // .Grid(STATE_DIM / 32, len / 128)
    int h = threadIdx.x;
    int warpId = threadIdx.y;
    constexpr int SAMPLE_PER_WARP = 128 / KV_PRODUCT_TRANSP_BLOCK_SIZE;
    int blockDimBase = blockIdx.x * WARP_SIZE;
    int blockTimeBase = blockIdx.y * 128;
    int thrTimeBase = warpId * SAMPLE_PER_WARP;

    float key[SAMPLE_PER_WARP];
    float value[SAMPLE_PER_WARP];
    for (int k = 0; k < SAMPLE_PER_WARP; ++k) {
        int t = blockTimeBase + thrTimeBase + k;
        if (t < len) {
            int nodeId = sortNode[t].NodeId;
            key[k] = kVecArr[nodeId][blockDimBase + h];
            value[k] = vVecArr[nodeId][blockDimBase + h];
        } else {
            key[k] = 0;
            value[k] = 0;
        }
    }

    __shared__ i8 buf[32][128 + 4]; // avoid smem bank conflict
    for (int blk = 0; blk < COMBINER_REP; ++blk) {
        for (int k = 0; k < SAMPLE_PER_WARP; ++k) {
            float keyShfl = __shfl_xor_sync(0xffffffff, key[k], blk);
            buf[h][thrTimeBase + k] = CvtToI8(keyShfl * value[k] * (VEC_SCALE * VEC_SCALE / VEC_SCALE));
        }
        __syncthreads();
        for (int y = 0; y < 32; y += KV_PRODUCT_TRANSP_BLOCK_SIZE) {
            int *src = (int *)&buf[y + warpId][h * 4];
            int *dst = (int *)&kvVecTArr[blockDimBase + blk * STATE_DIM + y + warpId][blockTimeBase + h * 4];
            *dst = *src;
        }
        __syncthreads();
    }
}
KERNEL_BLOCK_SIZE(KVProductShuffleTranspose, WARP_SIZE, KV_PRODUCT_TRANSP_BLOCK_SIZE);


template <int STATE_DIM, int HEAD_COUNT>
__global__ void KVProductBackprop(
    TCuda2DPtr<TVecFloat> kVecArr, TCuda2DPtr<TValueVecFloat> vVecArr, TCuda2DPtr<half> dkvVecArr, float *vecScaleArr,
    TCuda2DPtr<half> dkVecArr, TCuda2DPtr<half> dvVecArr, TCuda2DPtr<float> dScaleArr
)
{
    int h = threadIdx.x;
    int t = blockIdx.x;
    constexpr int WSZ = STATE_DIM / WARP_SIZE;
    constexpr int HEAD_DIM = STATE_DIM / HEAD_COUNT;
    float value[WSZ];
    LoadWarpVec<WSZ>(value, vVecArr[t]);
    float dValue[WSZ];
    LoadZeroWarpVec<WSZ>(dValue);
    float key[WSZ];
    LoadWarpVec<WSZ>(key, kVecArr[t]);
    float dKey[WSZ];
    LoadZeroWarpVec<WSZ>(dKey);
    float dScale[HEAD_COUNT];
    for (int head = 0; head < HEAD_COUNT; ++head) {
        dScale[head] = 0;
    }
    float vecScale = vecScaleArr ? vecScaleArr[t] : 1;
    for (int blk = 0; blk < COMBINER_REP; ++blk) {
        float dkv[WSZ];
        LoadWarpVec<WSZ>(dkv, dkvVecArr[t] + blk * STATE_DIM);
        ScaleWarpVec<WSZ>(dkv, vecScale);
        // forward: kv = shfl(key) * value
        for (int k = 0; k < WSZ; ++k) {
            float keyShfl = __shfl_xor_sync(0xffffffff, key[k] * VEC_SCALE, blk);
            float dKeyShfl = __shfl_xor_sync(0xffffffff, dkv[k] * value[k] * VEC_SCALE, blk); // reverse shuffle
            dKey[k] += dKeyShfl;
            dValue[k] += dkv[k] * keyShfl;
            int head = (k * WARP_SIZE + h) / HEAD_DIM;
            dScale[head] += dkv[k] * keyShfl * value[k] * VEC_SCALE;
        }
    }
    StoreWarpVec<WSZ>(dvVecArr[t], dValue);
    StoreWarpVec<WSZ>(dkVecArr[t], dKey);
    for (int head = 0; head < HEAD_COUNT; ++head) {
        dScaleArr[head][t] = WarpSum(dScale[head]);
    }
}
