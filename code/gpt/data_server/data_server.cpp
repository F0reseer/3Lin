#include "stdafx.h"
#include <lib/net/tcp_net.h>
#include <gpt/data_config/data_config.h>
#include <gpt/data/net_data.h>
#include <lib/config/config.h>
#include <lib/config/cfg_file.h>


TString DATA_SCRIPT =
    " make_char_dataset('D:/111enwiki9/wiki7_filter.txt')"
    ;

//TString DATA_SCRIPT =
//    " load_tokenizer('d:/tokenizers/50k.bin')"
//    " load_indexed_docset_folder('D:/text/Gutenberg/', 1)"
//    " load_indexed_docset_folder('D:/text/open_web_text/', 1)"
//    " load_indexed_docset_folder('D:/text/librusec/', 1)"
//    " load_indexed_docset_folder('D:/text/cultura_y/', 1)"
//    " make_dataset()"
//    ;


using namespace NNet;

///////////////////////////////////////////////////////////////////////////////////////////////////
static TIntrusivePtr<IDataSource> MakeDataSource(const TString &dataScript)
{
    TConfigFile cfg;
    ParseConfig(&cfg, dataScript);
    yint ptr = 0;
    TDataSourceConfigParser dataCfg;
    dataCfg.ParseScript(cfg.OpArr, &ptr);
    return dataCfg.GetDataset();
}


int main(int argc, char **argv)
{
    TOpt cmdline("s:", argc, argv);
    for (const TOpt::TParam &param : cmdline.Params) {
        if (param.Name == "s") {
            DebugPrintf("Data source script %s", param.Args[0].c_str()); fflush(0);
            TVector<char> cfg;
            Y_VERIFY(ReadWholeFile(param.Args[0], &cfg));
            Y_VERIFY(!cfg.empty() && "empty config");
            DATA_SCRIPT = cfg.data();
        }
    }

    TIntrusivePtr<IDataSource> data = MakeDataSource(DATA_SCRIPT);
    TIntrusivePtr<ITcpSendRecv> net = CreateTcpSendRecv();
    RunDataServer(net, data);
    return 0;
}
