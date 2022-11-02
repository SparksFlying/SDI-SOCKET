#pragma once
#include <iostream>
#include <map>
#include <string>
using std::map;
using std::string;

namespace config{
    const string workspaceFolder = "/home/zanglang/source/SDI/";
    constexpr uint32_t KEY_SIZE = 1024;
    constexpr size_t EPSILON = 64;
    constexpr uint64_t SIZE = 1000;
    constexpr size_t MAX_SEGMENT_SIZE = 300;
    const map<int,int> dim_bitperdim_table{
        {1, 64},
        {2, 32},
        {3, 21},
        {4, 16},
        {5, 12},
        {6, 10}
    };

    constexpr size_t FLOAT_EXP = 30;
    constexpr bool LOG = true;

    const string DO_IP = "127.0.0.1";
    const string DSP_IP = "127.0.0.1";
    const string DAP_IP = "127.0.0.1";
    const string CA_IP = "127.0.0.1";
    
    const int DO_PORT = 10001;
    const int DSP_PORT = 10002;
    const int DAP_PORT = 10003;
    const int CA_PORT = 10004;
};