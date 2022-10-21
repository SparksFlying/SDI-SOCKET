#pragma once
#include<iostream>
#include<map>
#include<string>
using std::map;
using std::string;
namespace config{
    constexpr uint32_t KEY_SIZE = 1024;
    constexpr size_t EPSILON = 32;
    constexpr uint64_t SIZE = 1000;
    const map<int,int> dim_bitperdim_table{
        {1, 64},
        {2, 32},
        {3, 21},
        {4, 16},
        {5, 12},
        {6, 10}
    };

    constexpr size_t FLOAT_EXP = 30;
    constexpr bool TRACK_CALC = false;
    constexpr bool INFO_NODE_PARAM = false;
    constexpr bool PADDING = true;
    constexpr bool PARAL = true;

    const string DO_IP = "127.0.0.1";
    const string DSP_IP = "127.0.0.1";
    const string DAP_IP = "127.0.0.1";
    const string CA_IP = "127.0.0.1";
    
    const int DO_PORT = 10001;
    const int DSP_PORT = 10002;
    const int DAP_PORT = 10003;
    const int CA_PORT = 10004;
};