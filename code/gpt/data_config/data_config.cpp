#include "stdafx.h"
#include "data_config.h"


void TTrainDataConfigParser::ParseScript(const TString &configText)
{
    TConfigFile cfg;
    ParseConfig(&cfg, configText);

    for (const TConfigFile::TOp &op : cfg.OpArr) {
        if (op.Op == CFG_OP_ASSIGNMENT) {
            if (op.Dst == "TEST_FRACTION") {
                TestFraction = atof(op.Args[0].c_str());
            } else if (op.Dst == "USE_PPM") {
                UsePPM = (IsYes(op.Args[0]));
            } else if (op.Dst == "TRAIN_CONFIG") {
                TrainConfig = op.Args[0];
            } else if (op.Dst == "DROP_CONFIG") {
                DropConfig = op.Args[0];
            } else if (op.Dst == "MODEL_DIMS") {
                Y_VERIFY(Data.StartParams == nullptr && "model dimenstion are useless, model already created");
                ModelDimsString = op.Args[0];
            } else {
                ParseScriptOp(op);
            }

        } else if (op.Op == CFG_OP_CALL) {
        // model ops
            if (op.Dst == "create_model") {
                Data.CreateModel(op, ModelDimsString, UsePPM);

            } else if (op.Dst == "load_model") {
                Y_VERIFY(YSize(op.Args) == 1);
                DebugPrintf("Load model %s\n", op.Args[0].c_str());
                Data.StartParams = new TModelParamsHolder();
                Serialize(true, op.Args[0], Data.StartParams->Params);
                Y_VERIFY(!Data.StartParams->Params.IsEmpty());

        // tokenizer ops
            } else if (op.Dst == "set_vocab_size") {
                Y_VERIFY(YSize(op.Args) == 1);
                Data.VocabSize = atoi(op.Args[0].c_str());

            } else if (op.Dst == "set_doc_start_token") {
                Y_VERIFY(YSize(op.Args) == 1);
                Data.OverrideDocStartToken = atoi(op.Args[0].c_str());

            } else if (op.Dst == "load_tokenizer") {
                Y_VERIFY(YSize(op.Args) == 1);
                Serialize(true, op.Args[0], Data.Tokenizer);
                Data.VocabSize = Data.Tokenizer.GetVocabSize();

            } else if (op.Dst == "make_byte_tokenizer") {
                Data.Tokenizer.MakeByteEncoder(TTokenizer::TK_CHAR);
                Data.VocabSize = Data.Tokenizer.GetVocabSize();

        // dataset ops
            } else if (op.Dst == "make_char_dataset") {
                Y_VERIFY(Data.StartParams == nullptr);
                Y_VERIFY(Data.Tokenizer.IsEmpty());
                Y_VERIFY(Data.DataBuild.Get() == 0);
                TVector<char> text;
                LoadDocument(&text, op.Args[0]);
                MakeCharDataset(&Data.Data, &Data.Tokenizer, text, TestFraction, UsePPM);
                Data.VocabSize = Data.Tokenizer.GetVocabSize();

            } else if (op.Dst == "load_bert_train" || op.Dst == "load_bert_test") {
                Y_VERIFY(YSize(op.Args) == 1);
                Y_VERIFY(Data.StartParams == nullptr);
                TDataset::ETrainTest trt = (op.Dst == "load_bert_train") ? TDataset::TRAIN : TDataset::TEST;
                Data.Data.LoadBert(Data.VocabSize, op.Args[0], trt);

            } else if (op.Dst == "load_tokenized_train" || op.Dst == "load_tokenized_test") {
                Y_VERIFY(Data.StartParams == nullptr);
                yint tokenWidth = 2;
                if (YSize(op.Args) > 1) {
                    tokenWidth = atoi(op.Args[1].c_str());
                }
                TVector<TBPEToken> data;
                LoadTokenized(op.Args[0], tokenWidth, &data);
                Data.CreateDatasetBuilders(Data.VocabSize, UsePPM);
                float ltTestFraction = (op.Dst == "load_tokenized_train") ? 0 : 1;
                TDatasetParams params(Data.VocabSize);
                params.CountDocset(data, 0, YSize(data), ltTestFraction);
                float weight = 1;
                Data.DataBuild->AddTokenizedDocset(data, params, weight);

            } else if (op.Dst == "load_text" || op.Dst == "load_folder" || op.Dst == "load_docset") {
                Y_VERIFY(Data.StartParams == nullptr);
                Y_VERIFY(!Data.Tokenizer.IsEmpty());
                Y_VERIFY(YSize(op.Args) > 0);
                TVector<TVector<char>> docSet;
                if (op.Dst == "load_text") {
                    docSet.resize(1);
                    LoadDocument(&docSet[0], op.Args[0]);
                } else if (op.Dst == "load_folder") {
                    LoadDocumentSetFromFiles(&docSet, op.Args[0]);
                } else if (op.Dst == "load_docset") {
                    LoadDocumentSetFromBin(&docSet, op.Args[0]);
                }
                Data.CreateDatasetBuilders(UsePPM);
                float weight = (YSize(op.Args) > 1) ? atof(op.Args[1].c_str()) : 1;
                AddDocset(Data.DataBuild.Get(), Data.Tokenizer, docSet, weight, TestFraction);

            } else if (op.Dst == "load_indexed_docset_folder") {
                Y_VERIFY(YSize(op.Args) > 0);
                Y_VERIFY(!Data.Tokenizer.IsEmpty());
                Data.CreateDatasetBuilders(UsePPM);
                float weight = (YSize(op.Args) > 1) ? atof(op.Args[1].c_str()) : 1;
                AddIndexedDocset(Data.DataBuild.Get(), op.Args[0], weight);

            } else if (op.Dst == "index_docset_folder") {
                Y_VERIFY(YSize(op.Args) == 1);
                Y_VERIFY(!Data.Tokenizer.IsEmpty());
                IndexDocsetDir(op.Args[0], Data.Tokenizer, UsePPM, TestFraction);

            } else if (op.Dst == "save_dataset") {
                Y_VERIFY(YSize(op.Args) == 1);
                Data.FinishDatasetBuild();
                Serialize(false, op.Args[0], Data.Data);

            } else if (op.Dst == "load_dataset") {
                Y_VERIFY(YSize(op.Args) == 1);
                Y_VERIFY(Data.DataBuild.Get() == 0);
                Serialize(true, op.Args[0], Data.Data);

            } else {
                ParseScriptOp(op);
            }
        }
    }
}
