#include "entity.hpp"
#include <iostream>
#include <array>
using Entity::record;
using std::array;
using std::vector;
using std::string;

vector<record<2>> getData(){
    vector<record<2>> data(1000);
    for(size_t i = 0; i < 1000; ++i){
        data[i].id = i;
        data[i].key = array<uint32_t, 2>{uint32_t(i + 1), uint32_t(i + 2)};
        data[i].value  = nullptr;
    }
    return data;
}

int run(int argc, char* argv[]){
    if(argc < 2) return 0;
    string entityType(argv[1]);
    if(entityType == "CA" || entityType == "ca"){
        Entity::CA ca(argc > 2 ? std::stoi(string(argv[2])) : config::KEY_SIZE);
        ca.run();
    }else if(entityType == "DO" || entityType == "do"){
        Entity::DO<2> dataowner("test", getData());
        dataowner.outSource();
        dataowner.run();
    }else if(entityType == "DSP" || entityType == "dsp"){
        Entity::DSP dsp;
        dsp.run();
    }else if(entityType == "DAP" || entityType == "dap"){
        Entity::DAP dap;
        dap.run();
    }else if(entityType == "AU" || entityType == "au"){
        Entity::AU au;
        au.query(QueryRectangle<uint32_t>(std::vector<uint32_t>{10, 10, 100, 100}));
    }
    return 0;
}

int main(int argc, char* argv[]){
    py::scoped_interpreter guard{};
    
    run(argc, argv);
}