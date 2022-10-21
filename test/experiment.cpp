#include "entity.hpp"
#include <iostream>
#include <array>
using Entity::record;
using std::array;
using std::vector;
using std::string;

vector<record<2>> getData(){
    vector<record<2>> data(1);
    data[0].id = 0;
    data[0].key = array<uint32_t, 2>{1, 2};
    data[0].value  = nullptr;
    return data;
}


int main(int argc, char* argv[]){
    if(argc < 2) return 0;
    string entityType(argv[1]);
    if(entityType == "CA" || entityType == "ca"){
        Entity::CA ca(argc > 2 ? std::stoi(string(argv[2])) : config::KEY_SIZE);
        ca.run();
    }else if(entityType == "DO" || entityType == "do"){
        Entity::DO<2> dataowner(getData());
        dataowner.outSource();
        dataowner.run();
    }else if(entityType == "DSP" || entityType == "dsp"){
        Entity::DSP dsp;
        dsp.run();
    }else if(entityType == "DAP" || entityType == "dap"){
        Entity::DAP dap;
        std::cout << "recv\n";
        dap.run();
    }else if(entityType == "AU" || entityType == "au"){
        Entity::AU au;
    }
    
    // Entity::DO<2> DataOwner(data);
    // DataOwner.buildIndex();

}