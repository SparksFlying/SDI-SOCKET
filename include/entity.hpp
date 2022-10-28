#pragma once
#include <vector>
#include <array>
#include <map>
#include <iostream>
#include <memory>
#include <list>
#include <rest_rpc.hpp>
#include <libzinc/zinc.hh>
#include <chrono>
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
using std::list;
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
    string dataName;
    vector<record<dim>> data_;
    size_t epsilon_ = 32;
    PaillierFast* crypto;
    std::map<size_t, uint64_t> mortonTable_;
    vector<interval_tree_t<uint64_t> ::node_type*> plainIndex;
    // 对端通信实体
    rpc_client dap_;
    rpc_client dsp_;

    // 日志系统
    Log olog;
    //std::set<std::shared_ptr<AU>> users_;

    DO(const string& dataName, const vector<record<dim>>& data, int port = 10001, int thread_num = std::thread::hardware_concurrency()): rpc_server(port, thread_num), data_(data), crypto(nullptr), dsp_(config::DSP_IP, config::DSP_PORT), dataName(dataName), olog("DO") {
        
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
        // if(config::DEBUG){
        //     auto idx = buildIndex();
        // }

        std::future<vector<string>> index = std::async(&DO::buildIndex, this);
        std::future<vector<pair<size_t, vector<string>>>> encData = std::async(&DO::encryptData, this);
        
        auto res1 = index.get();
        auto res2 = encData.get();

        while(!dsp_.connect());
        // 发送数据和索引到DSP
        dsp_.call("recvIndex", dataName, res1);
        dsp_.call("recvData", dataName, res2);
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
    DSP() : rpc_server(config::DSP_PORT, std::thread::hardware_concurrency()), crypto(nullptr), dap_(config::DAP_IP, config::DAP_PORT), olog("DSP"){
        recvKeys();
        while(!dap_.connect()){
            //std::this_thread::sleep_for(std::chrono::duration<int>(1));
        }

        this->register_handler("recvData", &DSP::recvData, this);
        this->register_handler("recvIndex", &DSP::recvIndex, this);
        this->register_handler("rangeQuery", &DSP::rangeQuery, this);
        this->register_handler("rangeQueryOpt", &DSP::rangeQueryOpt, this);
    }

    void recvKeys(){
        rpc_client cli(config::CA_IP, config::CA_PORT);
        while(!cli.connect()){

        }
        pair<string, string> param = cli.call<pair<string, string>>("getPub");
        array<string, 4> param2 = cli.call<array<string, 4>>("getPriv");
        PublicKey pk(config::KEY_SIZE, Integer(param.first.c_str()), Integer(param.second.c_str()));
        PrivateKey sk(config::KEY_SIZE, std::stoul(param2[0]), Integer(param2[1].c_str()), Integer(param2[2].c_str()), Integer(param2[3].c_str()));

        crypto = new PaillierFast(pk, sk);
        olog.infoLog("recv keys");
    }

    void recvData(rpc_conn conn, const string& dataName, vector<pair<size_t, vector<string>>>&& encData){
        string remoteIP = conn.lock()->remote_address();
        olog.infoLog("from " + remoteIP + " recv data size: " + std::to_string(encData.size()));

        this->name2data[dataName].resize(encData.size());
        auto iter = this->name2data[dataName].begin();
        for(size_t i = 0; i < encData.size(); ++i, ++iter){
            this->id2data[dataName][encData[i].first] = iter;
            iter->resize(encData[i].second.size());
            for(size_t j = 0; j < encData[i].second.size(); ++j){
                (*iter)[j] = Integer(encData[i].second[j].c_str());
            }
        }
    }

    void recvIndex(rpc_conn conn, const string& dataName, vector<string>&& index){
        string remoteIP = conn.lock()->remote_address();
        olog.infoLog("from " + remoteIP + " recv index size: " + std::to_string(index.size()));

        this->name2index[dataName].resize(index.size());
        for(size_t i = 0; i < index.size(); ++i){
            this->name2index[dataName][i].reset(deSeriEncSegmentNode(index[i]));
        }
        if(config::DEBUG){
            for(size_t i = 0; i < index.size(); ++i){
                printEncSegmentNode(crypto, this->name2index[dataName][i]);
            }
        }
    }

    Ciphertext checkData(const string& dataName, const EQueryRectangle& QR, const size_t& id){
        Ciphertext res = crypto->encrypt(0);
        auto iter = this->id2data.find(dataName)->second.find(id)->second;
        for(size_t i = 0; i < (*iter).size(); ++i){
            Integer cmp = SM(SMinus((*iter)[i], QR.get_minvec()[i]), SMinus((*iter)[i], QR.get_maxvec()[i])).data;
            res.data *= SIC(cmp.pow_mod_n(crypto->get_pub().n-1,*(crypto->get_n2())), crypto->encrypt(0)).data;
        }
        return res;
	}

    vector<vector<uint32_t>> rangeQuery(rpc_conn conn, const string& dataName, const pair<string, string>& range, const vector<string>& limits){
        olog.infoLog("recv request from" + conn.lock()->remote_address());
        if(!name2index.count(dataName)){
            return {};
        }
        //std::future<Ciphertext> fut1 = std::async(&DSP::CAP, this, name2index[dataName], Ciphertext(Integer(range.first.c_str())));
        //std::future<Ciphertext> fut2 = std::async(&DSP::CAP, this, name2index[dataName], Ciphertext(Integer(range.second.c_str())));

        //pair<Ciphertext, Ciphertext> posInfo{fut1.get(), fut2.get()}; 
        pair<Ciphertext, Ciphertext> encPosInfo{CAP(name2index[dataName], Ciphertext(Integer(range.first.c_str())), 0), CAP(name2index[dataName], Ciphertext(Integer(range.second.c_str())), 1)};
        Integer startPos(dap_.call<string>("DEC", encPosInfo.first.data.get_str()).c_str());
        Integer endPos(dap_.call<string>("DEC", encPosInfo.second.data.get_str()).c_str());

        char buf[50];
        std::sprintf(buf, "predict posinfo = [%s, %s]", (startPos / Integer(10).pow(config::FLOAT_EXP)).get_str().c_str(), (endPos / Integer(10).pow(config::FLOAT_EXP)).get_str().c_str());
        olog.debugLog(string(buf));

        {
            startPos /= Integer(10).pow(config::FLOAT_EXP);
            startPos -= Integer(config::EPSILON);
            if(startPos < 0) startPos = 0;
        }
        {
            endPos /= Integer(10).pow(config::FLOAT_EXP);
            endPos += Integer(config::EPSILON);
            if(endPos >= Integer(name2data[dataName].size())) endPos = Integer(name2data[dataName].size()) - 1;
        }
        pair<uint32_t, uint32_t> posInfo{startPos.to_uint(), endPos.to_uint()};

        vector<vector<uint32_t>> ret{{posInfo.first, posInfo.second}};
        // search range [startPos, endPos]
        return ret;
    }

    vector<vector<uint32_t>> rangeQueryOpt(rpc_conn conn, const string& dataName, const pair<string, string>& range, const vector<string>& limits){
        return {};
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
    Ciphertext SMinus(const Ciphertext& a,const Ciphertext& b) {
        return a.data * b.data.pow_mod_n(crypto->get_pub().n - 1, *(crypto->get_n2()));
    }

    // E(res)=E(a|b)
    // res=a+b-a*b
    Ciphertext SOR(const Ciphertext& a,const Ciphertext& b) {
        return SMinus(a.data * b.data, SM(a, b));
    }

    // res = times*a
    Ciphertext _times(const Ciphertext& a, Integer times) {
        return a.data.pow_mod_n(times, *(crypto->get_n2()));
    }

    // Secure Integer Comparison Protocol
    // 当a<=b return 1,否则return 0
    Ciphertext SIC(const Ciphertext& a, const Ciphertext& b) {
        Ciphertext X = _times(a, 2);
        Ciphertext Y = _times(b, 2).data * crypto->encrypt(1).data;
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
        if(coin == 0){
            ret = SMinus(crypto->encrypt(1), ret);
        }
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


    Ciphertext CAP(const vector<shared_ptr<encSegmentNode>>& index, const Ciphertext& input, bool whichPred = 0){
        char buf[30];
        std::sprintf(buf, "input = %s", crypto->decrypt(input).get_str().c_str());
        olog.debugLog(string(buf));

        Ciphertext P = crypto->encrypt(0);
        Ciphertext F = crypto->encrypt(0);

        size_t cur = 0;
        while(cur < index.size()){
            Ciphertext f = SCO(index[cur], input);
            Ciphertext acc = whichPred == 0 ? SM(index[cur]->pred1.first, input).data * index[cur]->pred1.second.data : SM(index[cur]->pred2.first, input).data * index[cur]->pred2.second.data;
            F.data *= f.data; 
            P.data *= SM(f, acc).data;

            std::sprintf(buf, "cur = %lu, F = %s, P = %s", cur, crypto->decrypt(f).get_str().c_str(), (crypto->decrypt(P) / Integer(10).pow(config::FLOAT_EXP)).get_str().c_str());
            olog.debugLog(string(buf));

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

    // dataname to data
    map<string, list<vector<Ciphertext>>> name2data;
    // dataname to index
    map<string, vector<shared_ptr<encSegmentNode>>> name2index;
    // data id to iterator of data
    map<string, map<size_t, list<vector<Ciphertext>>::iterator>> id2data;


    // 日志系统
    Log olog;
};

class DAP : public rpc_server{
public:

    DAP() : rpc_server(config::DAP_PORT, std::thread::hardware_concurrency()), dsp_(config::DSP_IP, config::DSP_PORT), olog("DAP"){
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

public:
    PaillierFast* crypto;
    rpc_client dsp_;
    void recvKeys(){
        rpc_client cli(config::CA_IP, config::CA_PORT);
        while(!cli.connect());
        pair<string, string> param = cli.call<pair<string, string>>("getPub");
        array<string, 4> param2 = cli.call<array<string, 4>>("getPriv");
        PublicKey pk(config::KEY_SIZE, Integer(param.first.c_str()), Integer(param.second.c_str()));
        PrivateKey sk(config::KEY_SIZE, std::stoul(param2[0]), Integer(param2[1].c_str()), Integer(param2[2].c_str()), Integer(param2[3].c_str()));

        crypto = new PaillierFast(pk, sk);
        olog.infoLog("recv keys");
    }
    // 日志系统
    Log olog;
};

class AU {
public:
    AU(){
        recvPubKey();
    }

    void recvPubKey(){
        rpc_client ca(config::CA_IP, config::CA_PORT);
        while(!ca.connect());
        
        pair<string, string> param = ca.call<pair<string, string>>("getPub");
        PublicKey pk(config::KEY_SIZE, Integer(param.first.c_str()), Integer(param.second.c_str()));
        crypto = new PaillierFast(pk);
    }
    void query(const QueryRectangle<uint32_t>& QR){
        rpc_client dsp_(config::DSP_IP, config::DSP_PORT);
        while(!dsp_.connect());
        pair<uint64_t, uint64_t> range{};
        vector<string> limits(QR.dim * 2);

        switch(QR.dim){
            case 2:{
                range.first = morton_code<2, 32>::encode(vec2arr<uint32_t, 2>(QR.get_minvec()));
                range.second = morton_code<2, 32>::encode(vec2arr<uint32_t, 2>(QR.get_maxvec()));
                break;
            }
            case 3:{
                range.first = morton_code<3, 21>::encode(vec2arr<uint32_t, 3>(QR.get_minvec()));
                range.second = morton_code<3, 21>::encode(vec2arr<uint32_t, 3>(QR.get_maxvec()));
                break;
            }
            case 4:{
                range.first = morton_code<4, 16>::encode(vec2arr<uint32_t, 4>(QR.get_minvec()));
                range.second = morton_code<4, 16>::encode(vec2arr<uint32_t, 4>(QR.get_maxvec()));
                break;
            }
            case 5:{
                range.first = morton_code<5, 12>::encode(vec2arr<uint32_t, 5>(QR.get_minvec()));
                range.second = morton_code<5, 12>::encode(vec2arr<uint32_t, 5>(QR.get_maxvec()));
                break;
            }
            case 6:{
                range.first = morton_code<6, 10>::encode(vec2arr<uint32_t, 6>(QR.get_minvec()));
                range.second = morton_code<6, 10>::encode(vec2arr<uint32_t, 6>(QR.get_maxvec()));
                break;
            }
        }

        // encrypt

        for(size_t i = 0; i < QR.dim; ++i){
            limits[i] = crypto->encrypt(QR.get_minvec()[i]).data.get_str();
            limits[i + QR.dim] = crypto->encrypt(QR.get_maxvec()[i]).data.get_str();
        }
        pair<string, string> encRange{crypto->encrypt(range.first).data.get_str(), crypto->encrypt(range.second).data.get_str()};
        string dataName = "test";
        //vector<vector<uint32_t>> res = dsp_.call<vector<vector<uint32_t>>>("rangeQuery", dataName, encRange, limits);
        auto fut = dsp_.async_call<FUTURE>("rangeQuery", dataName, encRange, limits);
        vector<vector<uint32_t>> res = fut.get().as<vector<vector<uint32_t>>>();
        if(!res.empty()){
            std::cout << res[0][0] << ", " << res[0][1] << std::endl;
        }else{
            std::cout << "empty" << std::endl;
        }
    }
private:
    PaillierFast* crypto;
};

class CA : public rpc_server{
public:
    CA(const uint32_t key_size = config::KEY_SIZE) : rpc_server(config::CA_PORT, 1), crypto(new PaillierFast(key_size)), olog("CA"){
        crypto->generate_keys();
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
    // 日志系统
    Log olog;
};

ENTITY_NAMESPACE_END