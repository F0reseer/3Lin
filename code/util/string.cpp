#include "stdafx.h"
#include "string.h"


static ui16 CharTable1251[128] = {
    1026, 1027, 8218, 1107,  8222, 8230, 8224, 8225,
    8364, 8240, 1033, 8249,  1034, 1036, 1035, 1039,
    1106, 8216, 8217, 8220,  8221, 8226, 8211, 8212,
    152, 8482, 1113, 8250,   1114, 1116, 1115, 1119,

    160, 1038, 1118, 1032,   164, 1168, 166, 167,
    1025, 169, 1028, 171,    172, 173, 174, 1031,
    176, 177, 1030, 1110,    1169, 181, 182, 183,
    1105, 8470, 1108, 187,   1112, 1029, 1109, 1111,

    1040, 1041, 1042, 1043,  1044, 1045, 1046, 1047,
    1048, 1049, 1050, 1051,  1052, 1053, 1054, 1055,
    1056, 1057, 1058, 1059,  1060, 1061, 1062, 1063,
    1064, 1065, 1066, 1067,  1068, 1069, 1070, 1071,

    1072, 1073, 1074, 1075,  1076, 1077, 1078, 1079,
    1080, 1081, 1082, 1083,  1084, 1085, 1086, 1087,
    1088, 1089, 1090, 1091,  1092, 1093, 1094, 1095,
    1096, 1097, 1098, 1099,  1100, 1101, 1102, 1103
};


const yint MAX_CODE = 9000;
static TAtomic TableReady, TableLock;
static ui8 UnicodeTo1251[MAX_CODE];

static void MakeTables()
{
    if (TableReady) {
        return;
    }
    TGuard<TAtomic> lock(TableLock);
    for (yint k = 0; k < ARRAY_SIZE(UnicodeTo1251); ++k) {
        if (k < 128) {
            UnicodeTo1251[k] = k;
        } else {
            UnicodeTo1251[k] = '?';
        }
    }
    for (yint k = 0; k < ARRAY_SIZE(CharTable1251); ++k) {
        UnicodeTo1251[CharTable1251[k]] = 128 + k;
    }
    TableReady = 1;
}


TString Utf2Win(const TString &utf8)
{
    MakeTables();
    TString res;
    yint sz = YSize(utf8);
    for (yint i = 0; i < sz; ++i) {
        ui8 f = utf8[i];
        ui32 code = 0;

        if (f < 128) {
            code = f;

        } else if ((f & 0xe0) == 0xc0) {
            if (i + 1 >= sz) {
                break;
            }
            ui32 c0 = utf8[i + 0];
            ui32 c1 = utf8[i + 1];
            code = (c1 & 0x3f) + ((c0 & 0x1f) << 6);

        } else if ((f & 0xf0) == 0xe0) {
            if (i + 2 >= sz) {
                break;
            }
            ui32 c0 = utf8[i + 0];
            ui32 c1 = utf8[i + 1];
            ui32 c2 = utf8[i + 2];
            code = (c2 & 0x3f) + ((c1 & 0x3f) << 6) + ((c0 & 0xf) << 12);

        } else if ((f & 0xf8) == 0xf0) {
            if (i + 3 >= sz) {
                break;
            }
            ui32 c0 = utf8[i + 0];
            ui32 c1 = utf8[i + 1];
            ui32 c2 = utf8[i + 2];
            ui32 c3 = utf8[i + 3];
            code = (c3 & 0x3f) + ((c2 & 0x3f) << 6) + ((c1 & 0x3f) << 12) + ((c0 & 0x7) << 18);

        } else {
            continue; // not first utf encoding char somehow
        }

        if (code < MAX_CODE) {
            res.push_back(UnicodeTo1251[code]);
        } else {
            res.push_back('?');
        }
    }
    return res;
}


TString Win2Utf(const TString &cp1251)
{
    MakeTables();
    TString res;
    for (ui8 c : cp1251) {
        if (c < 128) {
            res.push_back(c);
        } else {
            ui32 code = CharTable1251[c - 128];
            if (code < 0x800) {
                res.push_back(0xc0 + (code >> 6));
                res.push_back(0x80 + (code & 0x3f));
            } else if (code < 0x10000) {
                res.push_back(0xe0 + (code >> 12));
                res.push_back(0x80 + ((code >> 6) & 0x3f));
                res.push_back(0x80 + (code & 0x3f));
            } else {
                res.push_back(0xf0 + (code >> 18));
                res.push_back(0x80 + ((code >> 12) & 0x3f));
                res.push_back(0x80 + ((code >> 6) & 0x3f));
                res.push_back(0x80 + (code & 0x3f));
            }
        }
    }
    return res;
}


char Unicode2Win(yint key)
{
    MakeTables();
    if (key >= MAX_CODE) {
        return '?';
    }
    if (key < 128) {
        return key;
    }
    return UnicodeTo1251[key];
}


//#include <codecvt>
//#include <locale>
//void BuildTable()
//{
//    TVector<char> cp1251;
//    for (yint k = 128; k < 256; ++k) {
//        cp1251.push_back(k);
//    }
//    cp1251.push_back(0);
//    TVector<wchar_t> wkeyArr;
//    wkeyArr.resize(1000);
//    yint wkeySize = MultiByteToWideChar(1251, 0, cp1251.data(), -1, wkeyArr.data(), YSize(wkeyArr));
//    wkeySize = wkeySize;
//}
