#pragma once
#include <gpt/data/data_load.h>
#include <lib/config/cfg_file.h>
#include <lib/config/config.h>


///////////////////////////////////////////////////////////////////////////////////////////////////
class TDataSourceConfigParser
{
    TTokenizer Tokenizer;
    TIntrusivePtr<TDatasetBuilder> DataBuild;
    bool UsePPM = false;
    float TestFraction = 0.05f;
    yint VocabSize = 0;
    yint DocStartToken = -1;
    TIntrusivePtr<IDataSource> Dataset;

private:
    void LoadTokenizerParams()
    {
        VocabSize = Tokenizer.GetVocabSize();
        DocStartToken = Tokenizer.HasDocStartToken() ? Tokenizer.GetDocStartToken() : -1;
    }

    void CreateDatasetBuilder()
    {
        if (!Tokenizer.IsEmpty()) {
            Y_VERIFY(VocabSize == Tokenizer.GetVocabSize());
            if (Tokenizer.HasDocStartToken()) {
                Y_VERIFY(DocStartToken == Tokenizer.GetDocStartToken());
            } else {
                Y_VERIFY(DocStartToken == -1);
            }
        }
        Y_VERIFY(VocabSize > 0);
        if (DataBuild.Get() == 0) {
            DataBuild = new TDatasetBuilder(UsePPM, VocabSize, DocStartToken);
        }
    }

    void MakeDataset()
    {
        if (DataBuild.Get()) {
            Dataset = DataBuild->MakeDataset();
            DataBuild = 0;
            const IDataSource::TDataStats &stats = Dataset->GetStats();
            Y_VERIFY(stats.VocabSize == VocabSize);
            Y_VERIFY(stats.DocStartToken == DocStartToken);
        }
    }

public:
    void ParseScript(const TVector<TConfigFile::TOp> &opArr, yint *pOpPtr);
    TIntrusivePtr<IDataSource> GetDataset() { return Dataset; }
};
