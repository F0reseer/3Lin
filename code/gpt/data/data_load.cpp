#include "stdafx.h"
#include "data_load.h"
#include <lib/random/rand_utils.h>
#include <lib/file/dir.h>
#include <lib/hp_timer/hp_timer.h>
#include <util/thread.h>


///////////////////////////////////////////////////////////////////////////////////////////////////
void GenerateArithmetic()
{
    TMersenne<ui32> rng(1313);
    TOFStream f("d:/arith.txt");
    for (yint k = 0; k < 100000000; ++k) {
        yint maxVal = 10;
        int rr = rng.Uniform(4);
        if (rr == 0) {
            maxVal = 100000;
        } else if (rr == 1) {
            maxVal = 10000;
        } else if (rr == 2) {
            maxVal = 1000;
        } else if (rr == 3) {
            maxVal = 100;
        }
        int op = rng.Uniform(4);
        if (op > 1) {
            maxVal = Min<int>(maxVal, 10000);
        }
        yint n1 = rng.Uniform(maxVal);
        yint n2 = rng.Uniform(maxVal);
        if (op == 0) {
            f << n1 << " + " << n2 << " = " << n1 + n2 << "\n";
        } else if (op == 1) {
            f << n1 << " - " << n2 << " = " << n1 - n2 << "\n";
        } else {
            f << n1 << " * " << n2 << " = " << n1 * n2 << "\n";
        }
    }
}


void GenerateArithmetic97()
{
    // Grokking, binary ops train https://arxiv.org/pdf/2201.02177v1.pdf
    TVector<TString> samples;
    const yint MOD = 97;
    for (yint x = 0; x < MOD; ++x) {
        for (yint y = 0; y < MOD; ++y) {
            //yint val = (x + y) % MOD;
            yint val = (x * x + x * y + y * y + x) % MOD;
            samples.push_back(Sprintf("%c * %c = %c", x + 128, y + 128, val + 128));
        }
    }
    TMersenne<ui32> rng(1313);
    Shuffle(samples.begin(), samples.end(), rng);
    TOFStream f("d:/arith97.txt");
    f << "\n";
    for (const TString &str : samples) {
        f << str.c_str() << "\n";
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
//
static void CollectFilesRecursive(const TString &prefix, TVector<TString> *pRes)
{
    TVector<TFindFileResult> dir;
    FindAllFiles(prefix, &dir);
    for (const TFindFileResult &ff : dir) {
        if (ff.IsDir) {
            CollectFilesRecursive(prefix + "/" + ff.Name, pRes);
        }
    }
    for (const TFindFileResult &ff : dir) {
        if (!ff.IsDir) {
            pRes->push_back(prefix + "/" + ff.Name);
        }
    }
}


void LoadDocument(TVector<char> *pRes, const TString &fileName)
{
    Y_VERIFY(ReadWholeFile(fileName, pRes));
}


void LoadDocumentSetFromFiles(TVector<TVector<char>> *pRes, const TString &dir)
{
    TVector<TString> files;
    CollectFilesRecursive(dir, &files);
    //printf("Load %g files\n", YSize(files) * 1.);
    for (const TString &ff : files) {
        //printf("Load %s\n", ff.c_str());
        TVector<char> &text = *pRes->insert(pRes->end());
        Y_VERIFY(ReadWholeFile(ff, &text));
    }
}


void LoadDocumentSetFromBin(TVector<TVector<char>> *pRes, const TString &fileName)
{
    TFileStream f(IO_READ, fileName);
    pRes->reserve(100000);
    while (f.IsValid()) {
        ui32 sz = 0;
        if (f.Read(&sz, sizeof(sz)) != sizeof(sz)) {
            break;
        }
        TVector<char> &dst = *pRes->insert(pRes->end());
        dst.resize(sz);
        yint chk = f.Read(dst.data(), sz);
        if (chk != sz) {
            DebugPrintf("file %s, expected to read %g bytes, get %g bytes \n", fileName.c_str(), sz * 1., chk * 1.);
            break;
        }
    }
}


template <int WIDTH>
void CopyInts(TVector<TBPEToken> *p, const ui8 *src, yint len)
{
    ClearPodArray(p, len);
    const ui8 *srcPtr = src;
    TBPEToken *dstPtr = p->data();
    for (yint t = 0; t < len; ++t) {
        ui64 x = 0;
        ui8 *xPtr = (ui8 *)&x;
        for (yint x = 0; x < WIDTH; ++x) {
            *xPtr++ = *srcPtr++;
        }
        *dstPtr++ = x;
    }
}

void LoadTokenized(const TString &fileName, yint tokenWidth, yint headerSize, TVector<TBPEToken> *p)
{
    TFileStream f1(IO_READ, fileName);
    Y_VERIFY(f1.IsValid());
    if (headerSize > 0) {
        TVector<ui8> header;
        header.resize(headerSize);
        f1.Read(header.data(), headerSize);
    }
    yint len = (f1.GetLength() - headerSize) / tokenWidth;
    TVector<ui8> buf;
    buf.resize(len * tokenWidth);
    f1.Read(buf.data(), YSize(buf));
    switch (tokenWidth) {
    case 1: CopyInts<1>(p, buf.data(), len); break;
    case 2: CopyInts<2>(p, buf.data(), len); break;
    case 3: CopyInts<3>(p, buf.data(), len); break;
    case 4: CopyInts<4>(p, buf.data(), len); break;
    default:
        Y_VERIFY(0 && "unexpected token width");
    }
}


void SaveDocumentSetToBin(const TVector<TVector<char>> &textArr, const TString &fileName)
{
    TFileStream f(IO_WRITE, fileName);
    for (const TVector<char> &text : textArr) {
        ui32 sz = YSize(text);
        f.Write(&sz, sizeof(sz));
        f.Write(text.data(), sz);
    }
}


//void Repack()
//{
//    TString prefix = "D:/text/cultura_y/";
//    TVector<TFindFileResult> dir;
//    FindAllFiles(prefix, &dir);
//    yint resId = 0;
//    TVector<TVector<char>> resDocs;
//    yint totalSize = 0;
//    for (const TFindFileResult &ff : dir) {
//        TVector<TVector<char>> docs;
//        LoadDocumentSetFromBin(&docs, prefix + ff.Name);
//        for (yint k = 0; k < YSize(docs); ++k) {
//            resDocs.push_back(docs[k]);
//            totalSize += YSize(docs[k]);
//            if (totalSize > 150 * 1000000) {
//                SaveDocumentSetToBin(resDocs, Sprintf("d:/%d.bin", (int)resId));
//                ++resId;
//                resDocs.resize(0);
//                totalSize = 0;
//            }
//        }
//    }
//    SaveDocumentSetToBin(resDocs, Sprintf("d:/%d.bin", (int)resId));
//}


///////////////////////////////////////////////////////////////////////////////////////////////////
TIntrusivePtr<TDataset> MakeCharDataset(TTokenizer *pTokenizer, const TVector<char> &text, float testFraction, bool usePPM)
{
    pTokenizer->MakeUsedLettersEncoder(text);

    TVector<TBPEToken> data;
    yint charLen = pTokenizer->GenWords(text, 0, YSize(text), &data);

    TDatasetParams params(pTokenizer->GetVocabSize());
    params.CountDocset(data, 0, charLen, testFraction);

    Y_VERIFY(!pTokenizer->HasDocStartToken());
    TIntrusivePtr<TDatasetBuilder> db = new TDatasetBuilder(usePPM, pTokenizer->GetVocabSize(), -1);
    float weight = 1;
    db->AddTokenizedDocset(data, params, weight);
    return db->MakeDataset();
}


void AddDocset(TIntrusivePtr<TDatasetBuilder> pBuilder, const TTokenizer &tokenizer, const TVector<TVector<char>> &docSet, float weight, float testFraction)
{
    TVector<TBPEToken> data;
    TBPEToken docStart = tokenizer.GetDocStartToken();
    data.push_back(docStart);
    yint totalLen = 0;
    for (const TVector<char> &text : docSet) {
        totalLen += tokenizer.GenWords(text, 0, YSize(text), &data);
        data.push_back(docStart);
    }
    TDatasetParams params(tokenizer.GetVocabSize());
    params.CountDocset(data, 0, totalLen, testFraction);

    pBuilder->AddTokenizedDocset(data, params, weight);
}



///////////////////////////////////////////////////////////////////////////////////////////////////
struct TIndexedDataset
{
    TDatasetParams Params;
    yint VocabSize = 0;
    SAVELOAD(Params, VocabSize);
};


void AddIndexedDocset(TDatasetBuilder *pBuilder, const TString &dir, float weight)
{
    TIndexedDataset hdr;
    Serialize(IO_READ, dir + "/index_hdr.bin", hdr);
    TString indexFname = dir + "/index.bin";
    TString ppmIndexFname = dir + "/index_ppm.bin";
    pBuilder->AddIndexedDocset(indexFname, ppmIndexFname, hdr.Params, hdr.VocabSize, weight);
}


struct TDocsetIndexContext
{
    struct TThreadHolder : public TThrRefBase
    {
        TThread Thr;
    };

    const TTokenizer &Tokenizer;
    yint DocStartToken = 0;
    TString Dir;
    bool UsePPM = false;
    float TestFraction = 0;
    TVector<TFindFileResult> AllFiles;
    std::atomic<yint> CurFileId;
    TVector<TIntrusivePtr<TThreadHolder>> Workers;
    TAtomic WriteLock;
    yint Offset = 0;
    TIntrusivePtr<TPackedBPETokenWriter> IndexFile;
    TIntrusivePtr<TPackedBPETokenWriter> IndexFilePPM;
    TDatasetParams Params;
    yint TokenWidth = 0;
    yint SrcHeaderSize = 0;

public:
    TDocsetIndexContext(const TTokenizer &tokenizer, yint vocabSize, yint docStartToken, const TString &dir, bool usePPM, float testFraction)
        : Tokenizer(tokenizer)
        , DocStartToken(docStartToken)
        , Dir(dir)
        , UsePPM(usePPM)
        , TestFraction(testFraction)
        , CurFileId(0)
        , WriteLock(0)
        , Params(vocabSize)
    {
        // erase old result
        EraseFile(dir + "/index.bin");
        EraseFile(dir + "/index_ppm.bin");
        EraseFile(dir + "/index_hdr.bin");
        // 
        FindAllFiles(dir, &AllFiles);
        //
        IndexFile = new TPackedBPETokenWriter(dir + "/index.bin", Params.BytesPerToken);
        if (UsePPM) {
            IndexFilePPM = new TPackedBPETokenWriter(dir + "/index_ppm.bin", Params.BytesPerToken);
        }
    }

    void SetLoadTokenizedParams(yint tokenWidth, yint srcHeaderSize)
    {
        TokenWidth = tokenWidth;
        SrcHeaderSize = srcHeaderSize;
    }

    void RunWorkers(yint workerCount)
    {
        for (yint k = 0; k < workerCount; ++k) {
            TThreadHolder *p = new TThreadHolder;
            p->Thr.Create(this);
            Workers.push_back(p);
        }
    }

    void WaitCompletion()
    {
        Workers.clear();
    }

    void WorkerThread()
    {
        for (;;) {
            yint fileId = CurFileId.fetch_add(1);
            if (fileId >= YSize(AllFiles)) {
                return;
            }
            const TFindFileResult &ff = AllFiles[fileId];
            if (ff.IsDir) {
                continue;
            }

            TVector<TBPEToken> data;
            TString srcFileName = Dir + "/" + ff.Name;
            yint utf8charCount = 0;
            if (Tokenizer.IsEmpty()) {
                // load tokenized
                LoadTokenized(srcFileName, TokenWidth, SrcHeaderSize, &data);
                utf8charCount += YSize(data);
            } else {
                TVector<TVector<char>> docSet;
                LoadDocumentSetFromBin(&docSet, srcFileName);

                //NHPTimer::STime tStart;
                //NHPTimer::GetTime(&tStart);
                TBPEToken docStart = DocStartToken;
                data.push_back(docStart);
                for (const TVector<char> &text : docSet) {
                    utf8charCount += Tokenizer.GenWords(text, 0, YSize(text), &data);
                    data.push_back(docStart);
                }
            }

            TVector<TBPEToken> ppm;
            if (UsePPM) {
                ComputeWindowPPM(data, &ppm, DocStartToken);
            }

            {
                TGuard<TAtomic> gg(WriteLock);
                Params.CountDocset(data, Offset, utf8charCount, TestFraction);
                IndexFile->Write(data);
                if (UsePPM) {
                    IndexFilePPM->Write(ppm);
                }
                Offset += YSize(data);
                DebugPrintf(".");
                //DebugPrintf("time passed %g\n", NHPTimer::GetTimePassed(&tStart));
            }
        }
    }
};


static void IndexDir(const TString &dir, TDocsetIndexContext &ctx, yint vocabSize)
{
    const yint WORKER_COUNT = 8;
    ctx.RunWorkers(WORKER_COUNT);
    ctx.WaitCompletion();

    TIndexedDataset hdr;
    hdr.Params = ctx.Params;
    hdr.VocabSize = vocabSize;
    Serialize(IO_WRITE, dir + "/index_hdr.bin", hdr);
    DebugPrintf("\n");
}


void IndexDocsetDir(const TString &dir, const TTokenizer &tokenizer, bool usePPM, float testFraction)
{
    DebugPrintf("Indexing %s folder\n", dir.c_str());
    yint vocabSize = tokenizer.GetVocabSize();
    yint docStartToken = tokenizer.GetDocStartToken();
    TDocsetIndexContext ctx(tokenizer, vocabSize, docStartToken, dir, usePPM, testFraction);
    IndexDir(dir, ctx, tokenizer.GetVocabSize());
}


void IndexTokenizedDir(const TString &dir, yint vocabSize, yint docStartToken, bool usePPM, float testFraction, yint tokenWidth, yint headerSize)
{
    DebugPrintf("Indexing tokenized data in %s folder\n", dir.c_str());
    TTokenizer emptyTokenizer;
    TDocsetIndexContext ctx(emptyTokenizer, vocabSize, docStartToken, dir, usePPM, testFraction);
    ctx.SetLoadTokenizedParams(tokenWidth, headerSize);
    IndexDir(dir, ctx, vocabSize);
}
