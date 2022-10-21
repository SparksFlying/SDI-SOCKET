#pragma once
#include <vector>
#include <array>
#include <map>
#include <iostream>
#include <memory>
#include <rest_rpc.hpp>
#include <libzinc/zinc.hh>
#include "config.hpp"
#include "segment.hpp"
#include "utility.hpp"
#include "interval-tree/interval_tree.hpp"
#include <thread>
#include "ophelib/paillier_fast.h"
#include "ophelib/vector.h"
#include "ophelib/omp_wrap.h"
#include "ophelib/packing.h"
#include "ophelib/util.h"
#include "ophelib/ml.h"
#include "ophelib/random.h"
#include "utility.hpp"
#include "protocol.hpp"
#include <cmath>

#define ENTITY_NAMESPACE_BEGIN namespace Entity{
#define ENTITY_NAMESPACE_END };


ENTITY_NAMESPACE_BEGIN

#define LEFT 0
#define RIGHT 1

using std::vector;
using std::map;
using std::string;
using std::pair;
using std::array;
using std::shared_ptr;
using namespace lib_interval_tree;
using namespace rest_rpc;
using namespace rpc_service;
using namespace ophelib;


template<uint32_t dim = 2>
class record {
public:
    size_t id;
    std::array<uint32_t, dim> key;
    void* value;
};

template<uint32_t dim = 2>
class encRecord {
public:
    size_t id;
    std::array<Ciphertext, dim> key;
    void* value;
};

template<uint32_t dim = 2>
class DO : public rpc_server{
public:
    vector<record<dim>> data_;
    size_t epsilon_ = 32;
    PaillierFast* crypto;
    std::map<size_t, uint64_t> mortonTable_;
    vector<interval_tree_t<uint64_t> ::node_type*> plainIndex;
    // 对端通信实体
    rpc_client dap_;
    rpc_client dsp_;
    //std::set<std::shared_ptr<AU>> users_;

    DO(const vector<record<dim>>& data, int port = 10001, int thread_num = std::thread::hardware_concurrency()): rpc_server(port, thread_num), data_(data), crypto(nullptr), dsp_(config::DSP_IP, config::DSP_PORT) {
        
        // sort data according its morton code
        for(record<dim> item : data) {
            mortonTable_[item.id] = morton_code<dim, int(64/dim)>::encode(item.key);
        }

        std::sort(data_.begin(), data_.end(), [this](const record<dim>& a, const record<dim>& b){
            return this->mortonTable_[a.id] < this->mortonTable_[b.id];
        });

        // 接收秘钥
        recvKeys();
    }

    void outSource(){
        std::future<vector<string>> index = std::async(&DO::buildIndex, this);
        std::future<vector<pair<size_t, vector<string>>>> encData = std::async(&DO::encryptData, this);
        
        auto res1 = index.get();
        auto res2 = encData.get();

        while(!dsp_.connect());
        // 发送数据和索引到DSP
        dsp_.call("recvIndex", res1);
        dsp_.call("recvData", res2);
    }

    vector<pair<size_t, vector<string>>> encryptData(){
        vector<pair<size_t, vector<string>>> encData(this->data_.size());
        for(size_t i = 0; i < encData.size(); ++i){
            encData[i].first = this->data_[i].id;
            encData[i].second.resize(dim);
            for(size_t j = 0; j < dim; ++j){
                encData[i].second[j] = this->crypto->encrypt(this->data_[i].key[j]).data.get_str();
            }
        }
        return encData;
    }
    vector<string> buildIndex(){
        // 构建树
        lib_interval_tree::interval_tree_t<uint64_t> tree;
        buildTree(tree);

        // 填充
        plainIndex = paddingIntervaltree(tree);

        // 加密
        return encryptIntervaltree(plainIndex);
    }

    void buildTree(interval_tree_t<uint64_t>& tree){
        vector<uint64_t> keys(data_.size());
        vector<size_t> poses(data_.size());

        for(size_t idx = 0; idx < data_.size(); ++idx){
            keys[idx] = mortonTable_[data_[idx].id];
            poses[idx] = data_[idx].id;
        }

        vector<segment<uint64_t, size_t, double>> segs = shringkingCone<uint64_t, size_t>(keys, poses, epsilon_);
        std::vector<pair<interval<uint64_t, closed>, std::shared_ptr<pair<linearModel, linearModel>>>> segNodes = paddingSegments(segs);
        
        for(auto& node : segNodes){
            tree.insert(std::move(node.first), node.second);
        }

    }

    
    // 将树转为满二叉树
    vector<interval_tree_t<uint64_t> ::node_type*> paddingIntervaltree(interval_tree_t<uint64_t>& tree) {
        using node_t=interval_tree_t<uint64_t>::node_type;
        int height = getHeight(tree.root_);
        vector<node_t*> nodevec(std::pow(2, height) - 1, nullptr);
        nodevec[0] = tree.root_;

        for (size_t i = 0; 2 * i + 2 < nodevec.size(); ++i) {
            if (nodevec[i] != nullptr) {
                nodevec[2 * i + 1] = nodevec[i]->left_;
                nodevec[2 * i + 2] = nodevec[i]->right_;
            }
        }

        // 扩充为满二叉树的同时保留结点间关系
        uint64_t minv = 1;
        constexpr uint64_t maxv = (std::numeric_limits<uint64_t>::max)();
        std::default_random_engine e;
        std::uniform_int_distribution<uint64_t> u(minv, maxv);
        std::uniform_real_distribution<double> ud(0,1);
        for (size_t i = 1; i < nodevec.size(); ++i) {
            if (nodevec[i] == nullptr) {
                // 随即生成伪结点
                size_t rand_number = u(e);
                nodevec[i] = new node_t(nodevec[size_t((i - 1) / 2)],node_t::interval_type{rand_number,rand_number - 1});
                nodevec[i]->data_ptr = std::make_shared<pair<linearModel, linearModel>>(linearModel{0, 0}, linearModel{0, 0});
                nodevec[i]->left_=nullptr;
                nodevec[i]->right_=nullptr;
            }
        }
        for (size_t i = 0; 2 * i + 2 < nodevec.size(); ++i) {
            nodevec[i]->left_ = nodevec[2 * i + 1];
            nodevec[i]->right_ = nodevec[2 * i + 2];
        }
        return nodevec;
    }

    vector<string> encryptIntervaltree(const vector<interval_tree_t<uint64_t> ::node_type*>& nodevec){
        vector<string> encNodes(nodevec.size());

        // 加密并序列化节点参数
        for(size_t i = 0; i < encNodes.size(); ++i){
            encNodes[i] = seriEncSegmentNode(encSegmentNode(*(this->crypto), nodevec[i]));
        }
        return encNodes;
    }

    void recvKeys(){
        rpc_client cli(config::CA_IP, config::CA_PORT);
        while(!cli.connect());
        pair<string, string> param = cli.call<pair<string, string>>("getPub");
        array<string, 4> param2 = cli.call<array<string, 4>>("getPriv");
        PublicKey pk(config::KEY_SIZE, Integer(param.first.c_str()), Integer(param.second.c_str()));
        PrivateKey sk(config::KEY_SIZE, std::stoul(param2[0]), Integer(param2[1].c_str()), Integer(param2[2].c_str()), Integer(param2[3].c_str()));

        crypto = new PaillierFast(pk, sk);
    }
};

class DSP : public rpc_server{
public:
    DSP() : rpc_server(config::DSP_PORT, std::thread::hardware_concurrency()), crypto(nullptr), dap_(config::DAP_IP, config::DAP_PORT){
        recvKeys();
        while(!dap_.connect());

        this->register_handler("recvData", &DSP::recvData, this);
        this->register_handler("recvIndex", &DSP::recvIndex, this);
        this->register_handler("rangeQuery", &DSP::rangeQuery, this);
        this->register_handler("rangeQuery", &DSP::rangeQueryOpt, this);
    }

    void recvKeys(){
        rpc_client cli(config::CA_IP, config::CA_PORT);
        while(!cli.connect());
        pair<string, string> param = cli.call<pair<string, string>>("getPub");
        array<string, 4> param2 = cli.call<array<string, 4>>("getPriv");
        PublicKey pk(config::KEY_SIZE, Integer(param.first.c_str()), Integer(param.second.c_str()));
        PrivateKey sk(config::KEY_SIZE, std::stoul(param2[0]), Integer(param2[1].c_str()), Integer(param2[2].c_str()), Integer(param2[3].c_str()));

        crypto = new PaillierFast(pk, sk);
    }

    void recvData(rpc_conn conn, vector<pair<size_t, vector<string>>>&& encData){
        string remoteIP = conn.lock()->remote_address();
        std::cout << "\nfrom: " << remoteIP << std::endl;
        std::cout << "recv data size: " << encData.size() << std::endl;

        this->ip2data[remoteIP].resize(encData.size());
        for(size_t i = 0; i < encData.size(); ++i){
            this->ip2data[remoteIP][i].first = std::move(encData[i].first);
            this->ip2data[remoteIP][i].second.resize(encData[i].second.size());
            for(size_t j = 0; j < encData[i].second.size(); ++j){
                this->ip2data[remoteIP][i].second[j] = Integer(encData[i].second[j].c_str());
            }
        }
    }

    void recvIndex(rpc_conn conn, vector<string>&& index){
        string remoteIP = conn.lock()->remote_address();
        std::cout << "\nfrom: " << remoteIP << std::endl;
        std::cout << "recv index size: " << index.size() << std::endl;

        this->ip2index[remoteIP].resize(index.size());
        for(size_t i = 0; i < index.size(); ++i){
            this->ip2index[remoteIP][i].reset(deSeriEncSegmentNode(index[i]));
        }
    }

    vector<vector<uint32_t>> rangeQuery(rpc_conn conn, const pair<string, string>& range, const vector<string>& limits){
        string remoteIP = conn.lock()->remote_address();
        if(!ip2index.count(remoteIP)){
            return {};
        }
        std::future<Ciphertext> fut1 = std::async(&DSP::CAP, this, ip2index[remoteIP], Ciphertext(Integer(range.first.c_str())));
        std::future<Ciphertext> fut2 = std::async(&DSP::CAP, this, ip2index[remoteIP], Ciphertext(Integer(range.second.c_str())));

        pair<Ciphertext, Ciphertext> posInfo{fut1.get(), fut2.get()}; 
        Integer startPos(dap_.call<string>("DEC", posInfo.first.data.get_str()).c_str());
        Integer endPos(dap_.call<string>("DEC", posInfo.second.data.get_str()).c_str());

        // process postion info
        {
            startPos /= Integer(10).pow(config::FLOAT_EXP);
            startPos -= config::EPSILON;
            startPos = startPos < 0 ? 0 : startPos;
        }
        {
            endPos /= Integer(10).pow(config::FLOAT_EXP);
            endPos += config::EPSILON;
            endPos = endPos < ip2data[remoteIP].size() ? endPos : ip2data[remoteIP].size() - 1;
        }
        
        // search range [startPos, endPos]
        
    }

    void rangeQueryOpt(rpc_conn conn, const pair<string, string>& range, const vector<string>& limits){

    }

    // secure multiply protocol
    // res = E(a*b)
    Ciphertext SM(const Ciphertext& a, const Ciphertext& b){
        Random& r = Random::instance();
        Integer ra = r.rand_int(crypto->get_pub().n);
        Integer rb = r.rand_int(crypto->get_pub().n);
        Ciphertext tmp_a=crypto->encrypt(ra).data * a.data;
        Ciphertext tmp_b=crypto->encrypt(rb).data * b.data;
        Ciphertext h = Integer(dap_.call<string>("SM", tmp_a.data.get_str(), tmp_b.data.get_str()).c_str());

        auto s = h.data * a.data.pow_mod_n(crypto->get_pub().n - rb, *crypto->get_n2());
        auto tmp_s = s * b.data.pow_mod_n(crypto->get_pub().n - ra, *crypto->get_n2());
        Ciphertext res = tmp_s * crypto->encrypt(ra * rb % crypto->get_pub().n).data.pow_mod_n(crypto->get_pub().n - 1, *crypto->get_n2());
        return res;
    }

    // res=E(a-b)
    Ciphertext SMinus(const Ciphertext& a,const Ciphertext& b){
        return a.data * b.data.pow_mod_n(crypto->get_pub().n - 1, *(crypto->get_n2()));
    }

    // E(res)=E(a|b)
    // res=a+b-a*b
    Ciphertext SOR(const Ciphertext& a,const Ciphertext& b){
        return SMinus(a.data * b.data, SM(a, b));
    }

    // res = times*a
    Ciphertext _times(const Ciphertext& a, Integer times){
        return a.data.pow_mod_n(times, *(crypto->get_n2()));
    }

    // Secure Integer Comparison Protocol
    // 当a<=b return 1,否则return 0
    Ciphertext SIC(const Ciphertext& a, const Ciphertext& b){
        Ciphertext X = _times(a, 2);
        Ciphertext Y = _times(b, 2);
        Integer coin = Random::instance().rand_int(crypto->get_pub().n) % Integer(2);
        Ciphertext Z;
        if(coin == 1){
            Z = SMinus(X, Y);
        }else{
            Z = SMinus(Y, X);
        }

        int maxBitLength = (getMaxBitLength(crypto->get_pub().n) / 4 - 2).to_int();
        Integer r = Random::instance().rand_int(crypto->get_pub().n - 1) % (Integer(2).pow(maxBitLength)) + 1;
        Ciphertext c = _times(Z, r);

        Ciphertext ret = Integer(dap_.call<string>("SIC", c.data.get_str()).c_str());
        return ret;
    }

    Ciphertext SO(const Ciphertext& a_low,const Ciphertext& a_high,const Ciphertext& b_low,const Ciphertext& b_high){
        return SM(SIC(b_low, a_high), SIC(a_low, b_high));
    }

    Ciphertext SCO(const shared_ptr<encSegmentNode>& node, const Ciphertext& input){
        return SO(node->low_, node->high_, input, input);
    }

    bool SPFT(const vector<shared_ptr<encSegmentNode>>& index, size_t cur, const Ciphertext& F, const Ciphertext& input){
        if(2 * cur + 1 >= index.size()){
            return RIGHT;
        }
        Ciphertext pf = SIC(input, index[2 * cur + 1]->max_);

        Integer r = Random::instance().rand_int(crypto->get_pub().n) % Integer(2);
        Ciphertext f = SM(SMinus(crypto->encrypt(1), F), pf).data * SM(F, crypto->encrypt(r)).data;

        Integer ret(dap_.call<string>("DEC", f.data.get_str().c_str()) .c_str());
        return ret == 0 ? RIGHT : LEFT;
    }


    Ciphertext CAP(const vector<shared_ptr<encSegmentNode>>& index, const Ciphertext& input){
        Ciphertext P = crypto->encrypt(0);
        Ciphertext F = crypto->encrypt(0);

        size_t cur = 0;
        while(cur < index.size()){
            Ciphertext f = SCO(index[cur], input);
            Ciphertext acc = SM(index[cur]->pred1.first, input).data * index[cur]->pred1.second.data;
            F.data *= f.data; 
            P.data *= SM(F, acc).data;
            if(SPFT(index, cur, F, input) == LEFT){
                cur = 2 * cur + 1;
            }else{
                cur = 2 * cur + 2;
            }
        }
        return P;
    }

public:
    PaillierFast* crypto;
    rpc_client dap_;
    map<string, vector<pair<size_t, vector<Ciphertext>>>> ip2data;
    map<string, vector<shared_ptr<encSegmentNode>>> ip2index;
};

class DAP : public rpc_server{
public:

    DAP() : rpc_server(config::DAP_PORT, std::thread::hardware_concurrency()), dsp_(config::DSP_IP, config::DSP_PORT){
        recvKeys();

        // regist
        this->register_handler("SM", &DAP::SM, this);
        this->register_handler("SIC", &DAP::SIC, this);
        this->register_handler("DEC", &DAP::DEC, this);
    }


    string SM(rpc_conn conn, const string& a, const string& b){
        Integer ha = crypto->decrypt(Integer(a.c_str()) % *crypto->get_n2());
        Integer hb = crypto->decrypt(Integer(b.c_str()) % *crypto->get_n2());
        Integer h = (ha * hb) % crypto->get_pub().n;
        return crypto->encrypt(h).data.get_str();
    }

    string SIC(rpc_conn conn, const string& c){
        Integer m = crypto->decrypt(Ciphertext(Integer(c.c_str()))) % crypto->get_pub().n;
        Integer u = 0;
        if(getMaxBitLength(m >= 0 ? m : -m) > getMaxBitLength(crypto->get_pub().n) / 2){
            u = 1;
        }
        return crypto->encrypt(u).data.get_str();
    }

    string DEC(rpc_conn, const string& c){
        return crypto->decrypt(Integer(c.c_str())).get_str();
    }

private:
    PaillierFast* crypto;
    rpc_client dsp_;
    void recvKeys(){
        rpc_client cli(config::CA_IP, config::CA_PORT);
        while(!cli.connect());
        pair<string, string> param = cli.call<pair<string, string>>("getPub");
        array<string, 4> param2 = cli.call<array<string, 4>>("getPriv");
        PublicKey pk(config::KEY_SIZE, Integer(param.first.c_str()), Integer(param.second.c_str()));
        PrivateKey sk(config::KEY_SIZE, std::stoul(param2[0]), Integer(param2[1].c_str()), Integer(param2[2].c_str()), Integer(param2[3].c_str()));

        // std::cout << param.first << std::endl;
        // std::cout << param.second << std::endl;
        // std::cout << param2.first << std::endl;
        // std::cout << param2.second << std::endl; 
        crypto = new PaillierFast(pk, sk);
    }
    
};

class AU : public rpc_client{
    
};

class CA : public rpc_server{
public:
    CA(const uint32_t key_size = config::KEY_SIZE) : rpc_server(config::CA_PORT, 1), crypto(new PaillierFast(key_size)){
        crypto->generate_keys();
        std::cout << crypto->get_pub().n.get_str() << std::endl;
        std::cout << crypto->get_pub().g.get_str() << std::endl;
        std::cout << crypto->get_priv().key_size_bits << std::endl;
        std::cout << crypto->get_priv().p.get_str() << std::endl;
        std::cout << crypto->get_priv().q.get_str() << std::endl;
        this->register_handler("getPub", &CA::getPub, this);
        this->register_handler("getPriv", &CA::getPriv, this);
    }
    pair<string, string> getPub(rpc_conn conn){
        return {crypto->get_pub().n.get_str(), crypto->get_pub().g.get_str()};
    }
    array<string, 4> getPriv(rpc_conn conn){
        return {std::to_string(crypto->get_priv().a_bits), crypto->get_priv().p.get_str(), crypto->get_priv().q.get_str(), crypto->get_priv().a.get_str()};
    }

private:
    PaillierFast* crypto;
};

ENTITY_NAMESPACE_END