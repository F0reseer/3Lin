#pragma once

///////////////////////////////////////////////////////////////////////////////////////////////////
// final layer kernels

__global__ void Softmax(
    TCuda2DPtr<half> logProbArr, float *matrScale, float globalScale,
    int vocabSize, float *biasArr,
    TCuda2DPtr<float> predArr, float *predArrScale
)
{
    int h = threadIdx.x;
    int hh = threadIdx.y;
    int t = blockIdx.x;
    float srcScale = *matrScale * globalScale;
    half *src = logProbArr[t];
    float *dst = predArr[t];
    float maxVal = -1e38f; // normalize exp
    for (int blk = 0; blk < vocabSize; blk += WARP_SIZE * VEC_BLOCK) {
        int c = blk + hh * WARP_SIZE + h;
        if (c < vocabSize) {
            float val = float(src[c]) * srcScale;
            maxVal = fmaxf(val, maxVal);
        }
    }
    maxVal = VecBlockMax(maxVal);
    float sum = 0;
    for (int blk = 0; blk < vocabSize; blk += WARP_SIZE * VEC_BLOCK) {
        int c = blk + hh * WARP_SIZE + h;
        if (c < vocabSize) {
            float val = float(src[c]) * srcScale;
            float w = exp2f(val + biasArr[c] - maxVal);
            dst[c] = w;
            sum += w;
        }
    }
    sum = VecBlockSum(sum);
    if (IsMainVecBlockThread()) {
        predArrScale[t] = 1.0f / sum;
    }
}
KERNEL_BLOCK_SIZE(Softmax, WARP_SIZE, VEC_BLOCK);


__global__ void ComputeGradient(
    int len,
    int targetOffset, int vocabSize, int vocabRoundSize,
    TCuda2DPtr<float> predArr, float *predArrScale, int *targetArr,
    TCuda2DPtr<half> gradArr, float *sumTrainErr
)
{
    int t = blockIdx.x;
    int h = threadIdx.x;
    int cc = -1;
    if (t < len) {
        cc = targetArr[targetOffset + t];
    }
    if (cc >= 0) {
        float scale = predArrScale[t];
        for (int c = h; c < vocabRoundSize; c += WARP_SIZE) {
            if (c < vocabSize) {
                float pred = predArr[t][c] * scale;
                // omit scale gradient by log2, constant scale does not change anything
                if (c == cc) {
                    gradArr[t][c] = 1 - pred;
                    atomicAdd(&sumTrainErr[0], 1);
                    atomicAdd(&sumTrainErr[1], log(pred));
                } else {
                    gradArr[t][c] = -pred;
                }
            } else {
                gradArr[t][c] = 0;
            }
        }
    } else {
        for (int c = h; c < vocabRoundSize; c += WARP_SIZE) {
            gradArr[t][c] = 0;
        }
    }
}


__global__ void CollectSumTrainErr(float *sumTrainErr)
{
    if (threadIdx.x == 0) {
        sumTrainErr[2] += sumTrainErr[0];
        sumTrainErr[3] += sumTrainErr[1];
        sumTrainErr[0] = 0;
        sumTrainErr[1] = 0;
    }
}
