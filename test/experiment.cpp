#include "entity.hpp"
#include <iostream>
#include <array>
using std::array;
using std::vector;
using std::string;

size_t dim = 2;
size_t dataSize = 100000;
string dataName = "car";
string selectivity = "0.01";

std::map<string, string> nametrans{
    {"car", "car"},
    {"CAR", "car"},
    {"us", "US"},
    {"US", "US"},
    {"syn", "SYN"},
    {"SYN", "SYN"}
};

vector<record<2>> getData(){
    size_t size = 100000;
    vector<record<2>> data(size);
    for(size_t i = 0; i < size; ++i){
        data[i].id = i;
        data[i].key = array<uint32_t, 2>{uint32_t(i + 1), uint32_t(i + 2)};
    }

    // QueryRectangle<uint32_t> QR(std::vector<uint32_t>{4500, 4500, 5000, 5000});
    // for(size_t i = 0; i < size; ++i){
    //     if(QR.isFallin<2>(data[i].key)){
    //         std::cout << i << " ";
    //     }
    // }
    return data;
}

void test_Integer(){
    mpz_class _mp;
    _mp.set_str("ffff", 16);
    Integer b(_mp);
    std::cout << b.get_str() << std::endl;
}

void test_packLimit(){
    PaillierFast crypto(1024);
    crypto.generate_keys();
    std::cout << Vector::pack_count(32, crypto);
}

void test_IntegerMod(){
    Integer a(-3);
    a %= Integer(1000);
    std::cout << a.get_str() << std::endl;
    std::cout << (Integer(-3) % Integer(1000)).get_str() << std::endl; 
}

void test_seri()
{
    string block;
    {
        Entity::stringstream ss;
        ss << "123" << " ";
        ss << "456" << " ";
        ss << "789" << " ";
        block = ss.str();
    }
    {
        Entity::stringstream ss(block);
        vector<string> nums(3);
        ss >> nums[0];
        ss >> nums[1];
        ss >> nums[2];
        for(string& num : nums){
            std::cout << num << " ";
        }
    }
}

void initCA(int argc, char* argv[]){
    Log::instance("CA").setLevel(logging::trivial::severity_level::debug);
    Entity::CA ca(config::keysFolder + "/" + string(argv[2]) + "-bit");
    //Entity::CA ca(argc > 2 ? std::stoi(string(argv[2])) : config::KEY_SIZE);
    ca.run();
}

void initDataOwner(int argc, char* argv[]){
    Log::instance("DO").setLevel(logging::trivial::severity_level::debug);
    vector<vector<uint32_t>> data;
    if(argc > 3){
        if(string(argv[2]) == "outsource"){
            dataName = argv[3];
            dim = std::stoul(argv[4]);
            dataSize = std::stoul(argv[5]);
            if(dataName == "car"){
                data = read_from_roadnet_txt(dataSize, dim, config::dataFolder + "/" + "roadNet-CA-unique.txt");
            }else if(dataName == "syn"){
                data = read_from_syn_txt(dataSize, dim, config::dataFolder + "/" + "SYN_100000_6.txt");
            }else if(dataName == "us"){
                data = read_from_US_txt(vector<size_t>{1, 2}, dataSize, config::dataFolder + "/" + "USCensus1990.txt");
            }

            switch(dim){
                case 2:{
                    Entity::DO<2> dataowner(dataName, std::move(data));
                    dataowner.outSource();
                }
                case 3:{
                    Entity::DO<3> dataowner(dataName, std::move(data));
                    dataowner.outSource();
                }
                case 4:{
                    Entity::DO<4> dataowner(dataName, std::move(data));
                    dataowner.outSource();
                }
                case 5:{
                    Entity::DO<5> dataowner(dataName, std::move(data));
                    dataowner.outSource();
                }
                case 6:{
                    Entity::DO<6> dataowner(dataName, std::move(data));
                    dataowner.outSource();
                }
            }
            return;
        }
    }
}

void initDSP(int argc, char* argv[]){
    Log::instance("DSP").setLevel(logging::trivial::severity_level::debug);
    Entity::DSP dsp;
    if(argc > 2){
        if(string(argv[2]) == "query"){
            dataName = argv[3];
            dim = std::stoul(argv[4]);
            dataSize = std::stoul(argv[5]);
            dsp.load(dataName + "-" + std::to_string(dim) + "-" + std::to_string(dataSize));
        }
    }
    dsp.run();
}

void initDAP(int argc, char* argv[]){
    Log::instance("DAP").setLevel(logging::trivial::severity_level::debug);
    Entity::DAP dap;
    dap.run();
}

void initAU(int argc, char* argv[]){
    Log::instance("AU").setLevel(logging::trivial::severity_level::debug);
    Entity::AU au;

    if(argc > 2){
        dataName = argv[2];
    }
    if(argc > 3){
        dim = std::stoul(argv[3]);
    }
    if(argc > 4){
        dataSize = std::stoul(argv[4]);
    }
    if(argc > 5){
        selectivity = argv[5];
    }

    char buf[100];
    string format = "%s_%lu_%s";
    int len = sprintf(buf, format.c_str(), nametrans[dataName].c_str(), dataSize, selectivity.c_str());
    buf[len] = '\0';
    string path(config::queryFolder + "/" + buf);
    if(dim > 2){
        path += "_" + std::to_string(dim);
    }
    path += ".txt";
    vector<vector<uint32_t>> vals = read_vals_from_txt(path);

    dataName = dataName + "-" + std::to_string(dim) + "-" + std::to_string(dataSize);
    //auto data = read_from_roadnet_txt(dataSize, 2, config::dataFolder + "/" + "roadNet-CA-unique.txt");
    for(auto& val : vals){
        QueryRectangle<uint32_t> QR(val);
        au.query(dataName, QR);
        //vector<size_t> res;
        // for(size_t j = 0; j < data.size(); ++j){
        //     if(QR.isFallin(data[j])){
        //         res.push_back(j);
        //     }
        // }
        // std::cout << "real size is " << res.size() << " ";
        // if(res.size() > 0){
        //     std::cout << res.front() << ", " << res.back() << std::endl;
        // }
    }
}

int run(int argc, char* argv[]){
    if(argc < 2) return 0;
    string entityType(argv[1]);
    if(entityType == "CA" || entityType == "ca"){
        initCA(argc, argv);
    }else if(entityType == "DO" || entityType == "do"){
        initDataOwner(argc, argv);
    }else if(entityType == "DSP" || entityType == "dsp"){
        initDSP(argc, argv);
    }else if(entityType == "DAP" || entityType == "dap"){
        initDAP(argc, argv);
    }else if(entityType == "AU" || entityType == "au"){
        initAU(argc, argv);
    }
    return 0;
}

int main(int argc, char* argv[]){
    //test_Integer();
    //test_packLimit();
    //test_IntegerMod();
    run(argc, argv);
    //test_seri();
}