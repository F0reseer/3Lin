#pragma once

///////////////////////////////////////////////////////////////////////////////////////////////////
// final layer kernels

__global__ void Softmax(
    int len, TCuda2DPtr<half> logProbArr, float *matrScale, float globalScale,
    int vocabSize, float *biasArr,
    TCuda2DPtr<float> predArr, float *predArrScale
)
{
    int h = threadIdx.x;
    int t = blockIdx.x * SAMPLE_BLOCK + threadIdx.y;
    if (t < len) {
        float srcScale = *matrScale * globalScale;
        half *src = logProbArr[t];
        float *dst = predArr[t];
        float maxVal = -1e38f; // normalize exp
        for (int blk = 0; blk < vocabSize; blk += WARP_SIZE) {
            int c = blk + h;
            if (c < vocabSize) {
                float val = float(src[c]) * srcScale;
                maxVal = fmaxf(val, maxVal);
            }
        }
        maxVal = WarpMax(maxVal);
        float sum = 0;
        for (int blk = 0; blk < vocabSize; blk += WARP_SIZE) {
            int c = blk + h;
            if (c < vocabSize) {
                float val = float(src[c]) * srcScale;
                float w = exp2f(val + biasArr[c] - maxVal);
                dst[c] = w;
                sum += w;
            }
        }
        sum = WarpSum(sum);
        predArrScale[t] = 1.0f / sum;
    }
}
KERNEL_BLOCK_SIZE(Softmax, WARP_SIZE, SAMPLE_BLOCK);


__global__ void ComputeGradient(
    int targetOffset, int vocabSize, int vocabRoundSize,
    TCuda2DPtr<float> predArr, float *predArrScale, int *targetArr,
    TCuda2DPtr<half> gradArr
)
{
    int t = blockIdx.x;
    int h = threadIdx.x;

    int cc = targetArr[targetOffset + t];
    if (cc < 0) {
        for (int c = h; c < vocabRoundSize; c += WARP_SIZE) {
            gradArr[t][c] = 0;
        }
    } else {
        float scale = predArrScale[t];
        for (int c = h; c < vocabRoundSize; c += WARP_SIZE) {
            if (c < vocabSize) {
                float pred = predArr[t][c] * scale;
                // omit scale gradient by log2, constant scale does not change anything
                if (c == cc) {
                    gradArr[t][c] = 1 - pred;
                } else {
                    gradArr[t][c] = -pred;
                }
            } else {
                gradArr[t][c] = 0;
            }
        }
    }
}
