#include "stdafx.h"
#include <gpt/fed_lib/fed_lib.h>
#include <gpt/att/sliding_window.h>
#include <gpt/compute/gpt_cuda.cuh>
#include <gpt/data/data.h>
#include <lib/net/tcp_net.h>
#include <lib/file/dir.h>
#include <lib/config/config.h>
#include <lib/config/cfg_file.h>
#include <lib/hp_timer/hp_timer.h>


using namespace NNet;

///////////////////////////////////////////////////////////////////////////////////////////////////
static void ParseUserConfig(const TString &filename, TFedLogin *p)
{
    TVector<char> cfgText;
    Y_VERIFY(ReadWholeFile(filename, &cfgText));
    TConfigFile cfg;
    ParseConfig(&cfg, cfgText.data());
    for (TConfigFile::TOp &op : cfg.OpArr) {
        if (op.Op == CFG_OP_ASSIGNMENT) {
            if (op.Dst == "UserName") {
                p->UserName = op.Args[0];
            } else if (op.Dst == "UserId") {
                p->UserId = GetGuid(op.Args[0]);
            } else {
                DebugPrintf("Ignoring unknown variable %s\n", op.Dst.c_str());
            }
        } else {
            DebugPrintf("Ignoring unknown op %s\n", op.Dst.c_str());
        }
    }
}

static void PrintUserConfig(const TString &filename, const TFedLogin &login)
{
    TOFStream f(filename.c_str());
    f << "UserName = " << login.UserName.c_str() << "\n";
    f << "UserId = '" << GetGuidAsString(login.UserId).c_str() << "'\n";
}


///////////////////////////////////////////////////////////////////////////////////////////////////
template <class T>
static void SendData(TIntrusivePtr<ITcpSendRecv> net, TIntrusivePtr<ITcpConnection> conn, T &data)
{
    TIntrusivePtr<TTcpPacket> pkt = new TTcpPacket;
    SerializeMem(false, &pkt->Data, data);
    net->Send(conn, pkt);
}


static TIntrusivePtr<TTcpPacketReceived> RecvPacket(TIntrusivePtr<TTcpRecvQueue> q)
{
    TIntrusivePtr<TTcpPacketReceived> pkt;
    while (!q->RecvList.DequeueFirst(&pkt)) {
        SchedYield();
    }
    return pkt;
}


template <class T>
void RecvData(TIntrusivePtr<TTcpRecvQueue> net, T *p)
{
    TIntrusivePtr<TTcpPacketReceived> pkt;
    while (!net->RecvList.DequeueFirst(&pkt)) {
        SchedYield();
    }
    SerializeMem(true, &pkt->Data, *p);
}


void RecvWeightedModelParams(TIntrusivePtr<TTcpRecvQueue> net, TWeightedModelParamsPkt *p)
{
    TIntrusivePtr<TTcpPacketReceived> pkt;
    while (!net->RecvList.DequeueFirst(&pkt)) {
        SchedYield();
    }
    p->Swap(&pkt->Data);
}


///////////////////////////////////////////////////////////////////////////////////////////////////
extern yint MatrixAddWorkerThreadCount;

int main(int argc, char **argv)
{
    TString masterAddr = "185.137.233.184";
    TString configFilename = "user.cfg";
    TFedLogin login;
    yint batchSize = 16384;
    yint deviceCount = 1;

    if (DoesFileExist(configFilename)) {
        ParseUserConfig(configFilename, &login);
        DebugPrintf("user %s key loaded from %s\n", login.UserName.c_str(), configFilename.c_str());
    }

    TOpt cmdline("c:b:d:u:t:", argc, argv);
    for (const TOpt::TParam &param : cmdline.Params) {
        if (param.Name == "c") {
            masterAddr = param.Args[0];
        } else if (param.Name == "b") {
            batchSize = atoi(param.Args[0].c_str());
        } else if (param.Name == "d") {
            deviceCount = atoi(param.Args[0].c_str());
        } else if (param.Name == "u") {
            login.UserName = param.Args[0];
        } else if (param.Name == "t") {
            MatrixAddWorkerThreadCount = atoi(param.Args[0].c_str());
        }
    }

    if (!login.UserName.empty() && !IsValidUsername(login.UserName)) {
        DebugPrintf("illigal username %s\n", login.UserName.c_str());
        return 0;
    }

    if (!login.UserName.empty() && login.UserId.IsEmpty()) {
        CreateGuid(&login.UserId);
        PrintUserConfig(configFilename, login);
        DebugPrintf("user id saved to %s\n", configFilename.c_str());
    }

    TIntrusivePtr<ITcpSendRecv> net = CreateTcpSendRecv();
    TIntrusivePtr<TTcpRecvQueue> gradQueue = new TTcpRecvQueue();
    TIntrusivePtr<TTcpRecvQueue> dataQueue = new TTcpRecvQueue();
    TIntrusivePtr<ITcpConnection> gradConn = Connect(masterAddr, FED_GRAD_PORT, FedToken);
    TIntrusivePtr<ITcpConnection> dataConn = Connect(masterAddr, FED_DATA_PORT, FedToken);
    net->StartSendRecv(gradConn, gradQueue);
    net->StartSendRecv(dataConn, dataQueue);

    // send login data
    SendData(net, gradConn, login);

    // get config
    TFedParams fedParams;
    RecvData(gradQueue, &fedParams);
    TTrainConfig &config = fedParams.Config;
    DebugPrintf("received fed params\n"); fflush(0);

    // compute fragment count per batch
    if (config.TrainFragLen > batchSize) {
        DebugPrintf("train frag length %g is longer then worker batch size %g\n", config.TrainFragLen * 1., batchSize * 1.); fflush(0);
        return 0;
    }
    yint trainFragPerDevice = config.TrainBatchSize / deviceCount;
    if (trainFragPerDevice == 0 || trainFragPerDevice * deviceCount != config.TrainBatchSize) {
        DebugPrintf("suboptimal configuration, %g fragments per device (%g fragments, %g devices)\n",
            trainFragPerDevice * 1., config.TrainBatchSize * 1., deviceCount * 1.); fflush(0);
        return 0;
    }
    yint maxFragPerBatch = batchSize / (config.TrainFragLen + 1);
    Y_VERIFY(maxFragPerBatch > 0);
    yint accumulateSteps = (trainFragPerDevice + maxFragPerBatch - 1) / maxFragPerBatch;
    Y_VERIFY(accumulateSteps > 0);
    maxFragPerBatch = (trainFragPerDevice + accumulateSteps - 1) / accumulateSteps;
    yint maxNodeCount = maxFragPerBatch * (config.TrainFragLen + 1);
    DebugPrintf("%g devices, %g gradient accumulation steps, %g fragments per step, %g fragment legnth\n",
        deviceCount * 1., accumulateSteps * 1., maxFragPerBatch * 1., config.TrainFragLen * 1.); fflush(0);

    TWeightedModelParamsPkt basepoint;
    RecvWeightedModelParams(gradQueue, &basepoint);
    DebugPrintf("received base point\n"); fflush(0);

    // create model
    TIntrusivePtr<IModel> pModel;
    {
        TModelParams params;
        UnpackModelParams(&params, basepoint);
        pModel = CreateModel(deviceCount, params);
    }
    TIntrusivePtr<IComputeContext> pCtx = NCUDA_GPT::CreateContext(pModel, maxNodeCount);

    // our delta contribution
    TIntrusivePtr<TTcpPacket> myDeltaPkt = new TTcpPacket;
    // current epoch weight
    float currentWeight = 0;

    TIntrusivePtr<TTcpPacket> dataQuery = new TTcpPacket;
    net->Send(dataConn, dataQuery);

    NHPTimer::STime tLastDelta;
    NHPTimer::GetTime(&tLastDelta);
    
    TXRng rng(GetCycleCount());
    for (yint iter = 0;; ++iter) {
        if (iter > 0 && (iter % 100) == 0) {
            double score = pCtx->GetAvrgTrainErr();
            DebugPrintf("iter %gk, avrg train score %g\n", iter / 1000., score * fedParams.Compression); fflush(0);
        }

        // receive batch
        TIntrusivePtr<TTcpPacketReceived> fragPkt = RecvPacket(dataQueue);
        TVector<TFragment> fragArr;
        SerializeMem(true, &fragPkt->Data, fragArr);
        Y_VERIFY(YSize(fragArr) == config.TrainBatchSize);
        //DebugPrintf("got %g fragments\n", YSize(fragArr) * 1.); fflush(0);
        net->Send(dataConn, dataQuery);

        for (yint accStep = 0; accStep < accumulateSteps; ++accStep) {
            EAddToModel addToModel = GRADIENT_ACCUMULATE;
            yint base = accStep * maxFragPerBatch * deviceCount;
            yint fragCount = maxFragPerBatch;
            if (accStep == accumulateSteps - 1) {
                // last accumulate step
                addToModel = GRADIENT_APPLY;
                fragCount = (YSize(fragArr) - base) / deviceCount;
                Y_VERIFY(YSize(fragArr) == base + fragCount * deviceCount && "batch should be fully processed");
            }
            // provide train data to devices
            for (yint deviceId = 0; deviceId < deviceCount; ++deviceId) {
                TVector<TFragment> devFrags;
                for (yint k = 0; k < fragCount; ++k) {
                    devFrags.push_back(fragArr[base + deviceId * fragCount + k]);
                }
                MakeTrain(rng, devFrags, config.TokenDrop, config.ChannelDrop, pCtx.Get(), deviceId);
            }
            // backprop
            TTrainingStep step = config.GetStep(0, 100); // always take step from the start
            pCtx->Backprop(step, addToModel);
        }
        currentWeight += 1;

        // process new basepoint if it has arrived
        TIntrusivePtr<TTcpPacketReceived> newBasepointPkt;
        if (gradQueue->RecvList.DequeueFirst(&newBasepointPkt)) {
            double iterTime = NHPTimer::GetTimePassed(&tLastDelta);
            float tokensPerMin = currentWeight / iterTime * 60;
            DebugPrintf("%g tokens per minute, replacing basepoint, sz = %gmb\n", tokensPerMin, YSize(newBasepointPkt->Data) / 1000000.); fflush(0);
            // these operations can be done inplace without creating TModelParams object

            // current delta from basepoint
            TModelParams tune;
            pCtx->GetParams(&tune);
            AddPackedModelParamsScaled(&tune, basepoint, -1, 0);

            // keep current delta
            TWeightedModelParamsPkt newMyDelta;
            PackModelParams(&newMyDelta, tune, currentWeight);
            currentWeight = 0;

            // update basepoint from master
            basepoint.Swap(&newBasepointPkt->Data);

            // compute new params
            TModelParams &newParams = tune;
            TWeightedModelParamsPkt myDelta(&myDeltaPkt->Data);
            AddPackedModelParamsScaled(&newParams, basepoint, 1, 0);
            float myDeltaWeight = GetWeight(myDelta);
            // subtract our delta (our contribution to received basepoint)
            if (myDeltaWeight > 0) {
                float globalWeight = GetWeight(basepoint);
                Y_ASSERT(globalWeight >= myDeltaWeight);
                AddPackedModelParamsScaled(&newParams, myDelta, -myDeltaWeight / globalWeight * FED_WEIGHT_SCALE, 0);
            }

            // assign new myDelta
            newMyDelta.Swap(&myDeltaPkt->Data);

            // send delta
            net->Send(gradConn, myDeltaPkt);

            // update current params
            pCtx->SetParams(newParams);
            DebugPrintf("continue mining\n"); fflush(0);
            NHPTimer::GetTime(&tLastDelta);
        }
    }

    return 0;
}
