#pragma once
#include<iostream>
#include<map>
using std::map;
namespace config{
    constexpr size_t EPSILON=32;
    constexpr uint64_t SIZE=1000;
    const map<int,int> dim_bitperdim_table{
        {1,64},
        {2,32},
        {3,21},
        {4,16},
        {5,12},
        {6,10}
    };
};