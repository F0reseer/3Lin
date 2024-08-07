#pragma once

///////////////////////////////////////////////////////////////////////////////////////////////////
// attention utils
// each block process 16 samples, first warp computes weights, other accumulate results

struct TDotProductData
{
    union {
        struct {
            T8SMemI8Tile FromFrag8;
            T8SMemI8Tile ToFrag8;
        };
        struct {
            T4SMemHalfTile FromFrag;
            T4SMemHalfTile ToFrag;
        };
    };
};

// res[from][to]
template <int DP_DIM, class T1, class T2>
__forceinline __device__ void ComputeDotProducts(
    TDotProductData &data,
    int headOffset,
    TCuda2DPtr<T1> fromVecs, int fromBase, TCuda2DPtr<T2> toVecs, int toBase,
    TRegTile<float> *pSum
)
{
    CUDA_ASSERT(TILE_GROUP_SIZE == 64);
    pSum->Clear();
    __syncwarp();
    int fragmentOffset = headOffset & ~(TILE_GROUP_SIZE - 1);
    for (int x = 0; x < DP_DIM; x += TILE_GROUP_SIZE) {
        //__syncwarp(); // MMA() is implicit sync warp
        Copy4Tile(&data.FromFrag, fromVecs.Fragment(fragmentOffset + x, fromBase));
        Copy4Tile(&data.ToFrag, toVecs.Fragment(fragmentOffset + x, toBase));
        __syncwarp();
        int firstTile = (DP_DIM >= TILE_GROUP_SIZE) ? 0 : (headOffset >> 4) & 3;
        constexpr int SUM_TILE_COUNT = (DP_DIM >= TILE_GROUP_SIZE) ? TILE_GROUP : DP_DIM / TILE;
        for (int kTile = 0; kTile < SUM_TILE_COUNT; ++kTile) {
            MMA(pSum,
                TMmaRowMajor::FragA(data.FromFrag, firstTile + kTile),
                TMmaColMajor::FragB(data.ToFrag, firstTile + kTile));
        }
    }
}


template <int DP_DIM>
__forceinline __device__ void ComputeDotProducts(
    TDotProductData &data,
    int headOffset,
    TCuda2DPtr<i8> fromVecs, int fromBase, TCuda2DPtr<i8> toVecs, int toBase,
    TRegTile<int> *pSum
)
{
    CUDA_ASSERT(I8_TILE_GROUP_SIZE == 128);
    pSum->Clear();
    __syncwarp();
    int fragmentOffset = headOffset & ~(I8_TILE_GROUP_SIZE - 1);
    for (int x = 0; x < DP_DIM; x += I8_TILE_GROUP_SIZE) {
        //__syncwarp(); // MMA() is implicit sync warp
        Copy8Tile(&data.FromFrag8, fromVecs.Fragment(fragmentOffset + x, fromBase));
        Copy8Tile(&data.ToFrag8, toVecs.Fragment(fragmentOffset + x, toBase));
        __syncwarp();
        int firstTile = (DP_DIM >= I8_TILE_GROUP_SIZE) ? 0 : (headOffset >> 4) & 7;
        constexpr int SUM_TILE_COUNT = (DP_DIM >= I8_TILE_GROUP_SIZE) ? I8_TILE_GROUP : DP_DIM / TILE;
        for (int kTile = 0; kTile < SUM_TILE_COUNT; ++kTile) {
            MMA(pSum,
                TMmaRowMajor::FragA(data.FromFrag8, firstTile + kTile),
                TMmaColMajor::FragB(data.ToFrag8, firstTile + kTile));
        }
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// attention kernels
const int ATT_GROUP = 16;

// scaling qk vector has same effect as attention dot product scale
constexpr float ATT_QK_SCALE = 1;

// compute several attention quads in parallel
const int ATT_LOOKUP_BATCH = 4;
//const int ATT_GRAD_BATCH = 6; // template arg


template <int TT_GROUPS>
struct TAttentionLookupData
{
    TDotProductData DotData[ATT_LOOKUP_BATCH];
    TSwizzledSmemHalfTile wTile[ATT_LOOKUP_BATCH];
    T4SMemHalfTile vFrag[TT_GROUPS];
};

template <int HEAD_QDIM, int HEAD_TTDIM>
__global__ void ComputeAttentionValLookup(
    TCuda2DPtr<TVecFloat> qkState8, float *qkStateScale, TCuda2DPtr<TVecFloat> qvState8, float *qvStateScale, TCuda2DPtr<TValueVecFloat> vState8,
    TAttentionSpanGroup<ATT_GROUP> *attSpans2, int *attSpanPtr, float alibiSlope, float alibiHyper,
    TCuda2DPtr<float> sumWeightLog,
    TCuda2DPtr<half> valLookup8
)
{
    constexpr int TT_GROUPS = (HEAD_TTDIM > TILE_GROUP_SIZE) ? HEAD_TTDIM / TILE_GROUP_SIZE : 1;
    int h = threadIdx.x;

    int attBlock = blockIdx.x;
    int head = blockIdx.y;
    int fromBase = attBlock * ATT_GROUP;
    int headQoffset = head * HEAD_QDIM;
    int headTToffset = head * HEAD_TTDIM;

    TTileCoord tc;

    __shared__ TAttentionLookupData<TT_GROUPS> data;

    __shared__ int maxDP[16];
    __shared__ int tileMaxDP[ATT_LOOKUP_BATCH][16];
    __shared__ float sumWeightArr[ATT_LOOKUP_BATCH][16];
    __shared__ float resultScale[16];

    __shared__ float sumScale[16];

    if (threadIdx.y < ATT_LOOKUP_BATCH) {
        // wTile compute block
        int attBatchId = threadIdx.y;
        float attDotScale = CalcDotScale(HEAD_QDIM) * CalcAttentionMult();

        int ggFrom = h;
        if (ggFrom < 16) {
            if (attBatchId == 0) {
                maxDP[ggFrom] = 0;
            }
            sumWeightArr[attBatchId][ggFrom] = 1.0f / ATT_LOOKUP_BATCH; // start with some attention masks to avoid nans
            //sumWeightArr[attBatchId][ggFrom] = (attBatchId == 0) ? 1 : 0;
            tileMaxDP[attBatchId][ggFrom] = 0;
        }
        __syncwarp();

        for (int attIndex = attSpanPtr[attBlock]; attIndex < attSpanPtr[attBlock + 1]; ++attIndex) {
            const TAttentionSpanGroup<ATT_GROUP> &gg = attSpans2[attIndex];
            for (int toBatchBase = gg.Start; toBatchBase <= gg.Finish; toBatchBase += ATT_GROUP * ATT_LOOKUP_BATCH) {
                int toBase = toBatchBase + attBatchId * ATT_GROUP;
                TRegTile<float> dpTile;
                if (toBase <= gg.Finish) {
                    // res[from][to] = dot(qkState[from], qvState[to])
                    TRegTile<TVecFloatMMAResult> qProduct;
                    ComputeDotProducts<HEAD_QDIM>(data.DotData[attBatchId], headQoffset, qkState8, fromBase, qvState8, toBase, &qProduct);

                    // compute weight log (called dp)
                    for (int elem = 0; elem < tc.num_elements; ++elem) {
                        int from = tc.GetY(elem);
                        int to = tc.GetX(elem);
                        int glFrom = fromBase + from;
                        int glTo = toBase + to;
                        const TAttentionSpan &span = gg.Spans[from];
                        float dp = -10000;
                        if (glTo >= span.Start && glTo <= span.Finish) {
                            //dp = qProduct.x[elem] * attDotScale * VEC_SCALE * VEC_SCALE * qkStateScale[glFrom] * qvStateScale[glTo];
                            dp = qProduct.x[elem] * attDotScale * (VEC_SCALE * ATT_QK_SCALE) * (VEC_SCALE * qvStateScale[glTo]);
                            dp += GetAttentionDecay(glFrom - glTo, alibiSlope, alibiHyper);
                        }
                        dpTile.x[elem] = dp;
                    }
                    dpTile.StoreMaxByRow(tc, tileMaxDP[attBatchId]);
                }
                __syncthreads();

                // compute new maxDP, compute sumScale[]
                if (attBatchId == 0) {
                    if (ggFrom < 16) {
                        int oldMaxDP = maxDP[ggFrom];
                        int newMaxDP = oldMaxDP;
                        for (int k = 0; k < ATT_LOOKUP_BATCH; ++k) {
                            newMaxDP = max(newMaxDP, tileMaxDP[k][ggFrom]);
                        }
                        float dpScale = exp2f(oldMaxDP - newMaxDP);
                        maxDP[ggFrom] = newMaxDP;
                        sumScale[ggFrom] = dpScale;
                    }
                }
                __syncthreads();

                // compute weight
                if (ggFrom < 16) {
                    sumWeightArr[attBatchId][ggFrom] *= sumScale[ggFrom];
                }
                __syncwarp();
                if (toBase <= gg.Finish) {
                    TRegTile<half> wTile;
                    for (int elem = 0; elem < tc.num_elements; ++elem) {
                        int from = tc.GetY(elem);
                        wTile.x[elem] = exp2f(dpTile.x[elem] - maxDP[from]);
                    }
                    wTile.AddSumByRow(tc, sumWeightArr[attBatchId]);
                    wTile.Store(&data.wTile[attBatchId]);
                }
                __syncthreads();

                // accumulate in accumulate blocks
            }
        }
        // store sum weight, compute result scale
        if (attBatchId == 0) {
            if (ggFrom < 16) {
                float sumWeight = 0;
                for (int k = 0; k < ATT_LOOKUP_BATCH; ++k) {
                    sumWeight += sumWeightArr[k][ggFrom];
                }
                sumWeightLog[head][fromBase + ggFrom] = log2f(sumWeight) + maxDP[ggFrom];
                resultScale[ggFrom] = 1 / sumWeight; // sumWeight is guaranteed to be at least 1
            }
        }
        __syncthreads();
        // store results

    } else {
        // result accumulate blocks
        int ttGroup = threadIdx.y - ATT_LOOKUP_BATCH;
        int firstTile = (HEAD_TTDIM >= TILE_GROUP_SIZE) ? 0 : (headTToffset >> 4) & 3;
        int myTToffset = headTToffset + ttGroup * TILE_GROUP_SIZE;
        constexpr int SUM_TILE_COUNT = (HEAD_TTDIM >= TILE_GROUP_SIZE) ? TILE_GROUP : HEAD_TTDIM / TILE;

        TRegTile<float> vlSum[SUM_TILE_COUNT];
        for (int b = 0; b < SUM_TILE_COUNT; ++b) {
            vlSum[b].Clear();
        }

        for (int attIndex = attSpanPtr[attBlock]; attIndex < attSpanPtr[attBlock + 1]; ++attIndex) {
            const TAttentionSpanGroup<ATT_GROUP> &gg = attSpans2[attIndex];
            for (int toBatchBase = gg.Start; toBatchBase <= gg.Finish; toBatchBase += ATT_GROUP * ATT_LOOKUP_BATCH) {
                // compute dpTile
                __syncthreads();

                // compute maxDP and sumScale[]
                __syncthreads();

                // compute wTile, scale results
                { // scale result
                    TRegTile<float> tileSumScale;
                    tileSumScale.LoadRows(tc, sumScale);
                    for (int b = 0; b < SUM_TILE_COUNT; ++b) {
                        vlSum[b].Hadamard(tileSumScale);
                    }
                }
                __syncthreads();

                for (int attBatchId = 0; attBatchId < ATT_LOOKUP_BATCH; ++attBatchId) {
                    int toBase = toBatchBase + attBatchId * ATT_GROUP;
                    if (toBase <= gg.Finish) {
                        // add vectors to result
                        Copy4Tile(&data.vFrag[ttGroup], vState8.Fragment(myTToffset & ~(TILE_GROUP_SIZE - 1), toBase));
                        __syncwarp();
                        TRegTile<half> wTile;
                        wTile.Load(data.wTile[attBatchId]);

                        // accumulate results
                        for (int b = 0; b < SUM_TILE_COUNT; ++b) {
                            MMA(&vlSum[b],
                                TMmaRowMajor::FragA(wTile),
                                TMmaRowMajor::FragB(data.vFrag[ttGroup], firstTile + b));
                        }
                    }
                }
            }
        }
        // compute resultScale in wTile block
        __syncthreads();
        // store result
        TRegTile<float> tileResultScale;
        tileResultScale.LoadRows(tc, resultScale);
        for (int b = 0; b < SUM_TILE_COUNT; ++b) {
            vlSum[b].Hadamard(tileResultScale);
            vlSum[b].Store(tc, valLookup8.Fragment(myTToffset + b * TILE, fromBase));
        }
    }
}
//KERNEL_BLOCK_SIZE(ComputeAttentionValLookup, WARP_SIZE, ATT_LOOKUP_BATCH + TT_GROUPS);


template <int Q_GROUPS, int ATT_GRAD_BATCH>
struct TAttentionGradQKData
{
    TDotProductData DotData[ATT_GRAD_BATCH];
    TSwizzledSmemHalfTile dDot[2][ATT_GRAD_BATCH];
    T4SMemHalfTile vFrag[Q_GROUPS];
};


template <int HEAD_QDIM, int HEAD_TTDIM, int ATT_GRAD_BATCH>
__global__ void ComputeAttentionGradQK(
    TCuda2DPtr<TVecFloat> qkState8, float *qkStateScale, TCuda2DPtr<TVecFloat> qvState8, float *qvStateScale, TCuda2DPtr<TValueVecFloat> vState8,
    TAttentionSpanGroup<ATT_GROUP> *attSpans, int *attSpanPtr, float alibiSlope, float alibiHyper,
    TCuda2DPtr<half> dValLookup, TCuda2DPtr<float> dScaleArr,
    TCuda2DPtr<float> sumWeightLog,
    TCuda2DPtr<half> dQKState
)
{
    constexpr int Q_GROUPS = (HEAD_QDIM > TILE_GROUP_SIZE) ? HEAD_QDIM / TILE_GROUP_SIZE : 1;

    int attBlock = blockIdx.x;
    int head = blockIdx.y;
    int fromBase = attBlock * ATT_GROUP;
    int headQoffset = head * HEAD_QDIM;
    int headTToffset = head * HEAD_TTDIM;

    int dotBufId = 0;

    TTileCoord tc;

    __shared__ TAttentionGradQKData<Q_GROUPS, ATT_GRAD_BATCH> data;

    if (threadIdx.y < ATT_GRAD_BATCH) {
        int attBatchId = threadIdx.y;
        float attDotScale = CalcDotScale(HEAD_QDIM) * CalcAttentionMult();
        for (int attIndex = attSpanPtr[attBlock]; attIndex < attSpanPtr[attBlock + 1]; ++attIndex) {
            const TAttentionSpanGroup<ATT_GROUP> &gg = attSpans[attIndex];
            for (int toBatchBase = gg.Start; toBatchBase <= gg.Finish; toBatchBase += ATT_GROUP * ATT_GRAD_BATCH) {
                int toBase = toBatchBase + attBatchId * ATT_GROUP;
                if (toBase <= gg.Finish) {
                    // res[from][to] = dot(qkState[from], qvState[to])
                    TRegTile<TVecFloatMMAResult> qProduct;
                    ComputeDotProducts<HEAD_QDIM>(data.DotData[attBatchId], headQoffset, qkState8, fromBase, qvState8, toBase, &qProduct);

                    // dW[from][to] = dot(dValLookup[from], vState[to])
                    TRegTile<float> dW;
                    ComputeDotProducts<HEAD_TTDIM>(data.DotData[attBatchId], headTToffset, dValLookup, fromBase, vState8, toBase, &dW);

                    TRegTile<half> dDot;
                    for (int elem = 0; elem < tc.num_elements; ++elem) {
                        int from = tc.GetY(elem);
                        int to = tc.GetX(elem);
                        int glFrom = fromBase + from;
                        int glTo = toBase + to;
                        float w = 0;
                        const TAttentionSpan &span = gg.Spans[from];
                        if (glTo >= span.Start && glTo <= span.Finish) {
                            //float dp = qProduct.x[elem] * attDotScale * VEC_SCALE * VEC_SCALE * qkStateScale[glFrom] * qvStateScale[glTo];
                            float dp = qProduct.x[elem] * attDotScale * (VEC_SCALE * ATT_QK_SCALE) * (VEC_SCALE * qvStateScale[glTo]);
                            float attDecay = GetAttentionDecay(glFrom - glTo, alibiSlope, alibiHyper);
                            w = exp2f(dp + attDecay - sumWeightLog[head][glFrom]);
                        }
                        dDot.x[elem] = w * (dW.x[elem] * VEC_SCALE - dScaleArr[head][glFrom]) * attDotScale * LOG2; // log(2) from using exp2() instread of exp()
                    }
                    dDot.Store(&data.dDot[dotBufId][attBatchId]);
                }
                __syncthreads();
                // accumulate in accumulate blocks
                dotBufId = dotBufId ^ 1;
            }
        }

    } else {
        int qGroup = threadIdx.y - ATT_GRAD_BATCH;
        int firstTile = (HEAD_QDIM >= TILE_GROUP_SIZE) ? 0 : (headQoffset >> 4) & 3;
        int myQoffset = headQoffset + qGroup * TILE_GROUP_SIZE;
        constexpr int SUM_TILE_COUNT = (HEAD_QDIM >= TILE_GROUP_SIZE) ? TILE_GROUP : HEAD_QDIM / TILE;

        TRegTile<float> dqkStateSum[SUM_TILE_COUNT];
        for (int b = 0; b < SUM_TILE_COUNT; ++b) {
            dqkStateSum[b].Clear();
        }

        for (int attIndex = attSpanPtr[attBlock]; attIndex < attSpanPtr[attBlock + 1]; ++attIndex) {
            const TAttentionSpanGroup<ATT_GROUP> &gg = attSpans[attIndex];
            for (int toBatchBase = gg.Start; toBatchBase <= gg.Finish; toBatchBase += ATT_GROUP * ATT_GRAD_BATCH) {
                // compute dDot
                __syncthreads();

                for (int attBatchId = 0; attBatchId < ATT_GRAD_BATCH; ++attBatchId) {
                    int toBase = toBatchBase + attBatchId * ATT_GROUP;
                    if (toBase <= gg.Finish) {
                        TRegTile<half> dDot;
                        dDot.Load(data.dDot[dotBufId][attBatchId]);

                        // dQKState[from][x] += dDot[from][to] @ qvState[to][x];
                        //__syncwarp(); // MMA() is implicit sync warp
                        Copy4Tile(&data.vFrag[qGroup], qvState8.Fragment(myQoffset & ~(TILE_GROUP_SIZE - 1), toBase));
                        __syncwarp();
                        TRegTile<half> qvTileScale;
                        qvTileScale.ReplicateColumn(qvStateScale + toBase);
                        qvTileScale.Scale(VEC_SCALE);
                        for (int b = 0; b < SUM_TILE_COUNT; ++b) {
                            TRegTile<half> qvTile = LoadTile(data.vFrag[qGroup], firstTile + b);
                            qvTile.Hadamard(qvTileScale);
                            MMA(&dqkStateSum[b],
                                TMmaRowMajor::FragA(dDot),
                                TMmaRowMajor::FragB(qvTile));
                        }
                    }
                }
                dotBufId = dotBufId ^ 1;
            }
        }

        for (int b = 0; b < SUM_TILE_COUNT; ++b) {
            dqkStateSum[b].Scale(ATT_QK_SCALE);
            dqkStateSum[b].Store(tc, dQKState.Fragment(myQoffset + b * TILE, fromBase));
        }
    }
}
//KERNEL_BLOCK_SIZE(ComputeAttentionGradQK, WARP_SIZE, ATT_GRAD_BATCH + Q_GROUPS);


template <int Q_GROUPS, int TT_GROUPS, int ATT_GRAD_BATCH>
struct TAttentionGradQVData
{
    TDotProductData DotData[ATT_GRAD_BATCH];
    TSwizzledSmemHalfTile dDot[2][ATT_GRAD_BATCH];
    TSwizzledSmemHalfTile wHalfTile[2][ATT_GRAD_BATCH];
    T4SMemHalfTile dValFrag[TT_GROUPS];
    T4SMemHalfTile qkFrag[Q_GROUPS];
};


template <int HEAD_QDIM, int HEAD_TTDIM, int ATT_GRAD_BATCH>
__global__ void ComputeAttentionGradQV(
    TCuda2DPtr<TVecFloat> qkState8, float *qkStateScale, TCuda2DPtr<TVecFloat> qvState8, float *qvStateScale, TCuda2DPtr<TValueVecFloat> vState8,
    TAttentionSpanGroup<ATT_GROUP> *attSpans, int *attSpanPtr, float alibiSlope, float alibiHyper,
    TCuda2DPtr<half> dValLookup, TCuda2DPtr<float> dScaleArr,
    TCuda2DPtr<float> sumWeightLog,
    TCuda2DPtr<half> dQVState, TCuda2DPtr<half> dVState
)
{
    constexpr int TT_GROUPS = (HEAD_TTDIM > TILE_GROUP_SIZE) ? HEAD_TTDIM / TILE_GROUP_SIZE : 1;
    constexpr int Q_GROUPS = (HEAD_QDIM > TILE_GROUP_SIZE) ? HEAD_QDIM / TILE_GROUP_SIZE : 1;

    int attBlock = blockIdx.x;
    int head = blockIdx.y;
    int toBase = attBlock * ATT_GROUP;
    int headQoffset = head * HEAD_QDIM;
    int headTToffset = head * HEAD_TTDIM;

    int dotBufId = 0;

    TTileCoord tc;

    __shared__ TAttentionGradQVData<Q_GROUPS, TT_GROUPS, ATT_GRAD_BATCH> data;

    if (threadIdx.y < ATT_GRAD_BATCH) {
        int attBatchId = threadIdx.y;
        float attDotScale = CalcDotScale(HEAD_QDIM) * CalcAttentionMult();
        for (int attIndex = attSpanPtr[attBlock]; attIndex < attSpanPtr[attBlock + 1]; ++attIndex) {
            const TAttentionSpanGroup<ATT_GROUP> &gg = attSpans[attIndex];
            for (int fromBatchBase = gg.Start; fromBatchBase <= gg.Finish; fromBatchBase += ATT_GROUP * ATT_GRAD_BATCH) {
                int fromBase = fromBatchBase + attBatchId * ATT_GROUP;
                if (fromBase <= gg.Finish) {
                    // res[from][to] = dot(qkState[from], qvState[to])
                    TRegTile<TVecFloatMMAResult> qProduct;
                    ComputeDotProducts<HEAD_QDIM>(data.DotData[attBatchId], headQoffset, qkState8, fromBase, qvState8, toBase, &qProduct);

                    // dW[from][to] = dot(dValLookup[from], vState[to])
                    TRegTile<float> dW;
                    ComputeDotProducts<HEAD_TTDIM>(data.DotData[attBatchId], headTToffset, dValLookup, fromBase, vState8, toBase, &dW);

                    TRegTile<half> dDot;
                    TRegTile<half> wHalfTile;
                    for (int elem = 0; elem < tc.num_elements; ++elem) {
                        int from = tc.GetY(elem);
                        int to = tc.GetX(elem);
                        int glFrom = fromBase + from;
                        int glTo = toBase + to;
                        float w = 0;
                        const TAttentionSpan &span = gg.Spans[to];
                        if (glFrom >= span.Start && glFrom <= span.Finish) {
                            //float dp = qProduct.x[elem] * attDotScale * VEC_SCALE * VEC_SCALE * qkStateScale[glFrom] * qvStateScale[glTo];
                            float dp = qProduct.x[elem] * attDotScale * (VEC_SCALE * ATT_QK_SCALE) * (VEC_SCALE * qvStateScale[glTo]);
                            float attDecay = GetAttentionDecay(glFrom - glTo, alibiSlope, alibiHyper);
                            w = exp2f(dp + attDecay - sumWeightLog[head][glFrom]);
                        }
                        wHalfTile.x[elem] = w;
                        dDot.x[elem] = w * (dW.x[elem] * VEC_SCALE - dScaleArr[head][glFrom]) * attDotScale * LOG2; // log(2) from using exp2() instread of exp()
                    }

                    wHalfTile.Store(&data.wHalfTile[dotBufId][attBatchId]);
                    dDot.Store(&data.dDot[dotBufId][attBatchId]);
                }
                __syncthreads();
                // accumulate in accumulate blocks
                dotBufId = dotBufId ^ 1;
            }
        }

    } else if (threadIdx.y < ATT_GRAD_BATCH + TT_GROUPS) {
        int ttGroup = threadIdx.y - ATT_GRAD_BATCH;
        int firstTile = (HEAD_TTDIM >= TILE_GROUP_SIZE) ? 0 : (headTToffset >> 4) & 3;
        int myTToffset = headTToffset + ttGroup * TILE_GROUP_SIZE;
        constexpr int SUM_TILE_COUNT = (HEAD_TTDIM >= TILE_GROUP_SIZE) ? TILE_GROUP : HEAD_TTDIM / TILE;

        TRegTile<float> dVal2[SUM_TILE_COUNT];
        for (int b = 0; b < SUM_TILE_COUNT; ++b) {
            dVal2[b].Clear();
        }

        for (int attIndex = attSpanPtr[attBlock]; attIndex < attSpanPtr[attBlock + 1]; ++attIndex) {
            const TAttentionSpanGroup<ATT_GROUP> &gg = attSpans[attIndex];
            for (int fromBatchBase = gg.Start; fromBatchBase <= gg.Finish; fromBatchBase += ATT_GROUP * ATT_GRAD_BATCH) {
                // compute dDot
                __syncthreads();

                for (int attBatchId = 0; attBatchId < ATT_GRAD_BATCH; ++attBatchId) {
                    int fromBase = fromBatchBase + attBatchId * ATT_GROUP;
                    if (fromBase <= gg.Finish) {
                        TRegTile<half> wHalfTile;
                        wHalfTile.Load(data.wHalfTile[dotBufId][attBatchId]);

                        // (*pDVState)[to][x] += w[from][to] @ dValLookup[from][x];
                         //__syncwarp(); // MMA() is implicit sync warp
                        Copy4Tile(&data.dValFrag[ttGroup], dValLookup.Fragment(myTToffset & ~(TILE_GROUP_SIZE - 1), fromBase));
                        __syncwarp();
                        for (int b = 0; b < SUM_TILE_COUNT; ++b) {
                            MMA(&dVal2[b],
                                TMmaColMajor::FragA(wHalfTile),
                                TMmaRowMajor::FragB(data.dValFrag[ttGroup], firstTile + b));
                        }
                    }
                }
                dotBufId = dotBufId ^ 1;
            }
        }
        for (int b = 0; b < SUM_TILE_COUNT; ++b) {
            dVal2[b].Store(tc, dVState.Fragment(myTToffset + b * TILE, toBase));
        }

    } else {
        int qGroup = threadIdx.y - ATT_GRAD_BATCH - TT_GROUPS;
        int firstTile = (HEAD_QDIM >= TILE_GROUP_SIZE) ? 0 : (headQoffset >> 4) & 3;
        int myQoffset = headQoffset + qGroup * TILE_GROUP_SIZE;
        constexpr int SUM_TILE_COUNT = (HEAD_QDIM >= TILE_GROUP_SIZE) ? TILE_GROUP : HEAD_QDIM / TILE;

        TRegTile<float> dqvStateSum[SUM_TILE_COUNT];
        for (int b = 0; b < SUM_TILE_COUNT; ++b) {
            dqvStateSum[b].Clear();
        }
        for (int attIndex = attSpanPtr[attBlock]; attIndex < attSpanPtr[attBlock + 1]; ++attIndex) {
            const TAttentionSpanGroup<ATT_GROUP> &gg = attSpans[attIndex];
            for (int fromBatchBase = gg.Start; fromBatchBase <= gg.Finish; fromBatchBase += ATT_GROUP * ATT_GRAD_BATCH) {
                // compute dDot
                __syncthreads();

                for (int attBatchId = 0; attBatchId < ATT_GRAD_BATCH; ++attBatchId) {
                    int fromBase = fromBatchBase + attBatchId * ATT_GROUP;
                    if (fromBase <= gg.Finish) {
                        TRegTile<half> dDot;
                        dDot.Load(data.dDot[dotBufId][attBatchId]);

                        // dQVState[to][x] += dDot[from][to] @ qkState[from][x];
                        //__syncwarp(); // MMA() is implicit sync warp
                        Copy4Tile(&data.qkFrag[qGroup], qkState8.Fragment(myQoffset & ~(TILE_GROUP_SIZE - 1), fromBase));
                        __syncwarp();
                        TRegTile<half> qkTileScale;
                        //qkTileScale.ReplicateColumn(qkStateScale + fromBase + fromOffset * TILE);
                        //qkTileScale.FillEvery(VEC_SCALE);
                        qkTileScale.FillEvery(VEC_SCALE * ATT_QK_SCALE);
                        for (int b = 0; b < SUM_TILE_COUNT; ++b) {
                            TRegTile<half> qkTile = LoadTile(data.qkFrag[qGroup], firstTile + b);
                            qkTile.Hadamard(qkTileScale);
                            MMA(&dqvStateSum[b],
                                TMmaColMajor::FragA(dDot),
                                TMmaRowMajor::FragB(qkTile));
                        }
                    }
                }
                dotBufId = dotBufId ^ 1;
            }
        }

        for (int b = 0; b < SUM_TILE_COUNT; ++b) {
            dqvStateSum[b].Store(tc, dQVState.Fragment(myQoffset + b * TILE, toBase));
        }
    }
}
//KERNEL_BLOCK_SIZE(ComputeAttentionGradQV, WARP_SIZE, ATT_GRAD_BATCH + Q_GROUPS + TT_GROUPS);
