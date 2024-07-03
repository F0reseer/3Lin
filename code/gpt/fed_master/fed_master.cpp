#include "stdafx.h"
#include <lib/net/tcp_net.h>
#include <gpt/fed_lib/fed_lib.h>
#include <gpt/data_config/data_config.h>
#include "gpt/train_config/train_config.h"
#include <lib/hp_timer/hp_timer.h>
#include <lib/net/http_server.h>
#include <lib/net/http_request.h>
#include <lib/net/html_compose.h>
#include <lib/file/dir.h>


TString FED_SCRIPT =
    " DELTA_COLLECT_TIMEOUT = 10"
    " DELTA_COLLECT_MIN_INTERVAL = 10"
    " TRAIN_CONFIG = 'b64f64'"
    " DROP_CONFIG = 'drop0.9ch0.9reg2000'"
    " MODEL_DIMS = 'e256d65'" // 25M, default
    // load data, create model, train
    " make_char_dataset('D:/111enwiki9/wiki7_filter.txt')"
    " create_model(MPF_TAIL_LOSS, MPF_TUNE_FINAL_LAYER, MPF_TUNE_EMBED)"
    " run_fed_master('D:/models/fed_small')\n"
    ;

//TString FED_SCRIPT =
//    " KEEP_MODEL_COUNT = 100"
//    " DELTA_COLLECT_TIMEOUT = 300"
//    " DELTA_COLLECT_MIN_INTERVAL = 100"
//    " TEST_FRACTION = 0"
//    " TRAIN_CONFIG = 'b64f1024'"
//    " DROP_CONFIG = 'drop1ch1'"
//    " MODEL_DIMS = 'e1024tt256d65w1024'" // 420M
//    // load data, create model, train
//    " load_tokenizer('d:/tokenizers/50k.bin')"
//    " load_indexed_docset_folder('D:/text/Gutenberg/', 1)"
//    " load_indexed_docset_folder('D:/text/open_web_text/', 1)"
//    " load_indexed_docset_folder('D:/text/librusec/', 1)"
//    " load_indexed_docset_folder('D:/text/cultura_y/', 1)"
//    " create_model(MPF_TAIL_LOSS, MPF_TUNE_FINAL_LAYER, MPF_TUNE_EMBED)"
//    " run_fed_master('D:/models/fed')\n"
//    ;


const TString ModelFileExtension = ".m8";
using namespace NNet;


///////////////////////////////////////////////////////////////////////////////////////////////////
struct TFedState
{
    struct TWorker
    {
        TString Addr;
        float SumWeight = 0;
        float SumCount = 0;
        float CurrentWeight = 0;
    };
    TVector<TWorker> WorkerArr;
    float TimeSinceLastDelta = 0;
};

static void RenderRootPage(const TFedState &fs, TString *pRes)
{
    TString css = NNet::DefaultTableCSS("center");
    NNet::THtmlPage page("Fed server", css, "");

    // server state
    page += Sprintf("Time since last data: %g sec<br><br>", fs.TimeSinceLastDelta);

    // workers
    page += "<table><tr><th>Worker addr<th>Avrg weight<th>Current weight\n";
    for (const TFedState::TWorker &worker : fs.WorkerArr) {
        page += Sprintf("<tr><td>%s", worker.Addr.c_str());
        page += (worker.SumCount > 0) ? Sprintf("<td>%g\n", worker.SumWeight / worker.SumCount) : "<td>?";
        page += Sprintf("<td>%g\n", worker.CurrentWeight);
    }
    page += "</table><br>\n";

    // render result
    page.MakeHtml(pRes);
}



///////////////////////////////////////////////////////////////////////////////////////////////////
static void MakeBatches(TXRng &rng, const TTrainConfig &config, TDataset &data, TDataset::ETrainTest trt, TVector<TFragment> *pRes)
{
    TVector<TFragment> &fragArr = *pRes;
    for (yint k = 0; k < config.TrainBatchSize; ++k) {
        yint len = config.TrainFragLen;
        TFragment frag;
        data.MakeFragment(trt, rng, len, &frag);
        fragArr.push_back(frag);
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
template <class T>
TIntrusivePtr<TTcpPacket> MakePacket(T &data)
{
    TIntrusivePtr<TTcpPacket> pkt = new TTcpPacket;
    SerializeMem(false, &pkt->Data, data);
    return pkt;
}


class TFedMasterScriptParser : public TTrainDataConfigParser
{
    float DeltaCollectTimeout = 10 * 60; // in seconds
    float DeltaCollectMinInterval = 10; // in seconds
    yint KeepModelCount = 50;
    yint Version = 1;

private:
    // data queries handling
    struct TDataQueriesCtx
    {
        TDataset &Data;
        TIntrusivePtr<ITcpSendRecv> Net;
        TTrainConfig Config;

        TDataQueriesCtx(TDataset &data, TIntrusivePtr<ITcpSendRecv> net, const TTrainConfig &config) : Data(data), Net(net), Config(config) {}
    };

    static void ServeDataQueries(void *p)
    {
        TDataQueriesCtx &ctx = *(TDataQueriesCtx *)p;
        TIntrusivePtr<ITcpAccept> dataAccept = ctx.Net->StartAccept(FED_DATA_PORT, FedToken);
        TIntrusivePtr<TTcpRecvQueue> dataQueue = new TTcpRecvQueue();

        TXRng rng(GetCycleCount());
        for (;;) {
            // accept new data connections
            TIntrusivePtr<ITcpConnection> conn;
            while (dataAccept->GetNewConnection(&conn)) {
                conn->SetExitOnError(false);
                ctx.Net->StartSendRecv(conn, dataQueue);
            }

            // process data requests
            TIntrusivePtr<TTcpPacketReceived> recvPkt;
            while (dataQueue->RecvList.DequeueFirst(&recvPkt)) {
                //DebugPrintf("got fragment request from %p\n", recvPkt->Conn.Get());
                TVector<TFragment> fragArr;
                MakeBatches(rng, ctx.Config, ctx.Data, TDataset::TRAIN, &fragArr);
                ctx.Net->Send(recvPkt->Conn, MakePacket(fragArr));
            }
        }
    }


private:
    void LoadLastCheckpoint(const TString &folder)
    {
        TVector<TFindFileResult> dir;
        FindAllFiles(folder, &dir);
        TString modelFile;
        THashMap<int, TString> allModels;
        for (const TFindFileResult &ff : dir) {
            if (ff.IsDir) {
                continue;
            }
            if (EndsWith(ff.Name, ".tmp")) {
                EraseFile(ff.Name);
            }
            if (EndsWith(ff.Name, ModelFileExtension)) {
                TString sz = ff.Name.substr(0, YSize(ff.Name) - YSize(ModelFileExtension));
                yint numLen = 0;
                while (numLen < YSize(sz) && isdigit(sz[YSize(sz) - numLen - 1])) {
                    ++numLen;
                }
                yint version = atol(sz.substr(YSize(sz) - numLen).c_str());
                allModels[version] = folder + ff.Name;
                if (version >= Version) {
                    Version = version;
                    modelFile = folder + ff.Name;
                }
            }
        }
        if (!modelFile.empty()) {
            DebugPrintf("load model version %d\n", (int)Version);
            TWeightedModelParamsPkt wbp;
            wbp.Read(modelFile);
            Data.StartParams = new TModelParamsHolder();
            UnpackModelParams(&Data.StartParams->Params, wbp);
            ++Version;
            if (KeepModelCount > 0) {
                for (auto it = allModels.begin(); it != allModels.end(); ++it) {
                    if (it->first < Version - KeepModelCount) {
                        DebugPrintf("erase obsolete model %s\n", it->second.c_str());
                        EraseFile(it->second);
                    }
                }
            }
        }
    }

    void SaveModel(TWeightedModelParamsPkt &wbp, const TString &folder)
    {
        TString tmpName = folder + "model.tmp";
        wbp.Write(tmpName);
        RenameFile(tmpName, Sprintf("%smodel_%d%s", folder.c_str(), (int)Version++, ModelFileExtension.c_str()));
        DebugPrintf("model saved\n");
        if (KeepModelCount > 0) {
            TString oldFile = Sprintf("%smodel_%d%s", folder.c_str(), (int)(Version - KeepModelCount - 1), ModelFileExtension.c_str());
            if (DoesFileExist(oldFile)) {
                EraseFile(oldFile);
            }
        }
    }


private:
    struct TWorker
    {
        TFedState::TWorker Stat;
        bool GotDelta = true;
        bool FirstDelta = true;

        TWorker() {}
        TWorker(TIntrusivePtr<ITcpConnection> conn)
        {
            Stat.Addr = conn->GetPeerAddress();
        }
    };

    void RunFedMaster(const TString &folder)
    {
        TTrainConfig config(TrainConfig, DropConfig);

        // accept connections
        TIntrusivePtr<ITcpSendRecv> net = CreateTcpSendRecv();
        TIntrusivePtr<ITcpAccept> gradAccept = net->StartAccept(FED_GRAD_PORT, FedToken);
        TIntrusivePtr<TTcpRecvQueue> gradQueue = new TTcpRecvQueue();

        // http server
        THttpServer srv(FED_HTTP_PORT);

        // checkpoint
        LoadLastCheckpoint(folder);

        // verify model params
        Y_VERIFY(!Data.StartParams->Params.IsEmpty());
        Data.StartParams->Params.ModelDim.FragLen = config.TrainFragLen;
        Data.VerifyVocabSize();

        // init basepoint and sum delta
        TIntrusivePtr<TModelParamsHolder> sumDelta = Data.StartParams.Release();
        TIntrusivePtr<TTcpPacket> basepointPkt = new TTcpPacket;
        {
            TWeightedModelParamsPkt wbp;
            PackModelParams(&wbp, sumDelta->Params, 0);
            wbp.Swap(&basepointPkt->Data);
        }
        Scale(&sumDelta->Params, 0, 0);
        float sumDeltaWeight = 0;

        // worker params
        TFedParams fedParams;
        fedParams.Config = config;
        fedParams.Compression = Data.Data.GetCompression();

        // workers
        THashMap<TIntrusivePtr<ITcpConnection>, TWorker> workerSet;

        // timeout
        NHPTimer::STime tCurrent;
        NHPTimer::GetTime(&tCurrent);
        double collectTime = 0;

        // run train fragment gen thread
        TThread dataThread;
        TDataQueriesCtx dataCtx(Data.Data, net, config);
        dataThread.Create(ServeDataQueries, &dataCtx);

        // collect updates and send basepoints
        for (;;) {
            double deltaT = NHPTimer::GetTimePassed(&tCurrent);

            // http commands
            while (srv.CanAccept(0)) {
                THttpRequest req;
                SOCKET s = srv.AcceptNonBlocking(&req);
                if (s != INVALID_SOCKET) {
                    DebugPrintf("Http query %s\n", req.Req.c_str());
                    if (req.Req == "") {
                        TFedState fs;
                        fs.TimeSinceLastDelta = collectTime;
                        for (auto it = workerSet.begin(); it != workerSet.end(); ++it) {
                            fs.WorkerArr.push_back(it->second.Stat);
                        }
                        TString html;
                        RenderRootPage(fs, &html);
                        HttpReplyHTML(s, html);
                    } else {
                        ReplyNotFound(s);
                    }
                }
            }

            // accept new gradient connections
            TIntrusivePtr<ITcpConnection> conn;
            while (gradAccept->GetNewConnection(&conn)) {
                DebugPrintf("got connection from %s\n", conn->GetPeerAddress().c_str());
                conn->SetExitOnError(false);
                net->StartSendRecv(conn, gradQueue);
                net->Send(conn, MakePacket(fedParams));
                net->Send(conn, basepointPkt);
                workerSet[conn] = TWorker(conn);
                DebugPrintf("added worker %s, send basepoint\n", conn->GetPeerAddress().c_str());
            }

            // process gradient requests
            TIntrusivePtr<TTcpPacketReceived> recvPkt;
            while (gradQueue->RecvList.DequeueFirst(&recvPkt)) {
                TWeightedModelParamsPkt delta;
                delta.Swap(&recvPkt->Data);
                // add delta
                float deltaWeight = GetWeight(delta);
                AddPackedModelParamsScaled(&sumDelta->Params, delta, deltaWeight, deltaWeight);
                sumDeltaWeight += deltaWeight;
                // update worker
                Y_VERIFY(workerSet.find(recvPkt->Conn) != workerSet.end());
                TWorker &worker = workerSet[recvPkt->Conn];
                DebugPrintf("got delta from %s\n", worker.Stat.Addr.c_str());
                Y_VERIFY(worker.GotDelta == false);
                worker.GotDelta = true;
                worker.Stat.CurrentWeight = deltaWeight;
            }

            // check if delta is collected from all workers
            bool deltaCollected = true;
            for (auto it = workerSet.begin(); it != workerSet.end();) {
                bool keep = false;
                if (it->first->IsValid()) {
                    keep = true;
                    TWorker &worker = it->second;
                    if (!worker.GotDelta) {
                        if (collectTime > DeltaCollectTimeout) {
                            DebugPrintf("disconnect worker %s on delta collect timeout\n", worker.Stat.Addr.c_str());
                            it->first->Stop();
                            keep = false;
                        } else {
                            deltaCollected = false;
                        }
                    }
                }
                if (keep) {
                    ++it;
                } else {
                    auto del = it++;
                    workerSet.erase(del);
                }
            }

            // send new base point
            collectTime += deltaT;
            if (workerSet.empty() && sumDeltaWeight == 0) {
                collectTime = 0;
            } else if (deltaCollected) {
                if (collectTime >= DeltaCollectMinInterval) {
                    collectTime = 0;
                    DebugPrintf("delta collected, model version %d\n", (int)Version);
                    {
                        // add sum delta to basepoint
                        TWeightedModelParamsPkt wbp;
                        wbp.Swap(&basepointPkt->Data);
                        if (sumDeltaWeight > 0) {
                            Scale(&sumDelta->Params, FED_WEIGHT_SCALE / sumDeltaWeight, 1 / sumDeltaWeight);
                            AddPackedModelParamsScaled(&sumDelta->Params, wbp, 1, 0);
                            PackModelParams(&wbp, sumDelta->Params, sumDeltaWeight);
                        } else {
                            SetWeight(wbp, sumDeltaWeight);
                        }
                        // save new model
                        SaveModel(wbp, folder);
                        // keep in basepoint packet
                        wbp.Swap(&basepointPkt->Data);
                    }
                    // clear sum delta
                    if (sumDeltaWeight > 0) {
                        Scale(&sumDelta->Params, 0, 0);
                        sumDeltaWeight = 0;
                    }
                    // send new basepoint
                    for (auto it = workerSet.begin(); it != workerSet.end(); ++it) {
                        TWorker &worker = it->second;
                        net->Send(it->first, basepointPkt);
                        Y_VERIFY(worker.GotDelta);
                        DebugPrintf("send basepoint to %s\n", worker.Stat.Addr.c_str());
                        if (!worker.FirstDelta) {
                            worker.Stat.SumWeight += worker.Stat.CurrentWeight;
                            worker.Stat.SumCount += 1;
                        }
                        worker.Stat.CurrentWeight = 0;
                        worker.GotDelta = false;
                        worker.FirstDelta = false;
                    }
                }
            }
        }
    }


    void ParseScriptOp(const TConfigFile::TOp &op) override
    {
        if (op.Op == CFG_OP_ASSIGNMENT) {
            if (op.Dst == "DELTA_COLLECT_TIMEOUT") {
                DeltaCollectTimeout = atof(op.Args[0].c_str());
            } else if (op.Dst == "DELTA_COLLECT_MIN_INTERVAL") {
                DeltaCollectMinInterval = atof(op.Args[0].c_str());
            } else if (op.Dst == "KEEP_MODEL_COUNT") {
                KeepModelCount = atof(op.Args[0].c_str());
            } else {
                DebugPrintf("unknown config variable %s\n", op.Dst.c_str());
            }

        } else if (op.Op == CFG_OP_CALL) {
            if (op.Dst == "run_fed_master") {
                Y_VERIFY(YSize(op.Args) == 1);
                Data.FinishDatasetBuild();
                TString folder = op.Args[0];
                if (!folder.empty() && !EndsWith(folder, "/")) {
                    folder += '/';
                }
                RunFedMaster(folder);
            } else {
                DebugPrintf("unknown function %s\n", op.Dst.c_str());
                abort();
            }
        }
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    TOpt cmdline("c:", argc, argv);
    for (const TOpt::TParam &param : cmdline.Params) {
        if (param.Name == "c") {
            DebugPrintf("Fed script %s\n", param.Args[0].c_str());
            TVector<char> cfg;
            ReadWholeFile(param.Args[0], &cfg);
            FED_SCRIPT = cfg.data();
        }
    }

    // execute config script
    TFedMasterScriptParser fed;
    fed.ParseScript(FED_SCRIPT);

    return 0;
}
