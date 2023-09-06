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
#include "ophelib/packing.h"
#include "ophelib/util.h"
#include "ophelib/ml.h"
#include "ophelib/random.h"
#include "query.hpp"
#include <cmath>
#include <sstream>
#include <mutex>
#include <atomic>
#include "logging.hpp"

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
using std::stringstream;
using std::mutex;
using std::lock_guard;
using namespace lib_interval_tree;
using namespace rest_rpc;
using namespace rpc_service;
using namespace ophelib;

enum STATUS{
    SUCCESS = 200,
    NOSUCHDATA,
    NOTCOMPLATE,
    DUPLICATE
};

// static TimerClock tc;


template<uint32_t dim = 2>
class DO : public rpc_server{
public:
    using node_t = interval_tree_t<uint64_t>::node_type;
    string dataName;
    vector<record<dim>> data_;
    size_t epsilon_ = config::EPSILON; // error-bound
    PaillierFast* crypto;
    std::unordered_map<size_t, uint64_t> mortonTable_; // id -> morton code
    vector<node_t*> plainIndex;

    // place inserted new items
    vector<pair<uint64_t, record<dim>>> buffer; // vector of <morton code, record item>
    // comm clients
    rpc_client dap_;
    rpc_client dsp_;

    DO(const string& dataName, vector<record<dim>>&& data, int port = 10001, int thread_num = config::NUM_THREADS): rpc_server(port, thread_num), data_(data), crypto(nullptr), dsp_(config::DSP_IP, config::DSP_PORT), dataName(dataName){
        
        // calc records' morton code, take 'id' as primary key
        for(record<dim> item : data) {
            mortonTable_[item.id] = morton_code<dim, int(64/dim)>::encode(item.key);
        }
        
        // sort data according its morton code
        std::sort(data_.begin(), data_.end(), [this](const record<dim>& a, const record<dim>& b){
            return this->mortonTable_[a.id] < this->mortonTable_[b.id];
        });

        // recv paillier key
        recvKeys();

        // set props of dsp
        dsp_.set_connect_timeout(1000 * 100);
    }
    DO(const string& dataName, vector<vector<uint32_t>>&& data, int port = 10001, int thread_num = config::NUM_THREADS): DO(dataName, convert<dim>(std::forward<vector<vector<uint32_t>>&&>(data))){
        // TO DO
    }
    
    // 分块序列化加密数据
    vector<pair<pair<size_t, size_t>, string>> seriEncData(vector<pair<size_t, vector<string>>>& encData, const size_t blockSize = MAX_BUF_LEN - 10)
    {
        vector<pair<pair<size_t, size_t>, string>> ret;
        stringstream ss;
        size_t pre = 0;
        for(size_t i = 0; i < encData.size(); ++i)
        {
            size_t len = sizeof(encData[i].first);
            for(auto& str : encData[i].second)
            {
                len += str.size();
            }
            if(ss.str().size() + len >= blockSize - 16)
            {
                ret.push_back(pair<pair<size_t, size_t>, string>{{pre, i}, std::move(ss.str())});
                ss.str("");
                pre = i;
            }
            ss << std::move(encData[i].first) << " ";
            for(auto& str : encData[i].second)
            {
                ss << std::move(str) << " ";
            }
        }
        ret.push_back(pair<pair<size_t, size_t>, string>{{pre, encData.size()}, std::move(ss.str())});
        return ret;
    }

    // 分块序列化索引
    vector<pair<pair<size_t, size_t>, string>> seriIndex(vector<string>& index, const size_t blockSize = MAX_BUF_LEN - 10)
    {
        vector<pair<pair<size_t, size_t>, string>> ret;
        stringstream ss;
        size_t pre = 0;
        for(size_t i = 0; i < index.size(); ++i)
        {
            if(ss.str().size() + index[i].size() >= blockSize - 16)
            {
                ret.push_back(pair<pair<size_t, size_t>, string>{{pre, i}, std::move(ss.str())});
                ss.str("");
                pre = i;
            }
            ss << std::move(index[i]) << " ";
        }
        ret.push_back(pair<pair<size_t, size_t>, string>{{pre, index.size()}, std::move(ss.str())});
        return ret;
    }

    void outSource()
    {
        vector<pair<pair<size_t, size_t>, string>> encDataBlocks;
        vector<pair<pair<size_t, size_t>, string>> indexBlocks;
        {
            
            vector<string> index;
            vector<pair<size_t, vector<string>>> encData;
            if(config::PARAL){
                std::future<vector<string>> res1 = std::async(&DO::buildIndex, this);
                std::future<vector<pair<size_t, vector<string>>>> res2 = std::async(&DO::encryptData, this);
                index = res1.get();
                encData = res2.get();
            }
            else{
                index = buildIndex();
                encData = encryptData();
            }

            encDataBlocks = seriEncData(encData);
            //debug("encData seri %lu blocks", encDataBlocks.size());
            indexBlocks = seriIndex(index);
            //debug("index seri %lu blocks", indexBlocks.size());

            // saveData(config::dataFolder + "/" + dataName + ".dat", encData);
            // saveIndex(config::dataFolder + "/" + dataName + ".idx", index);
        }
        

        while(!dsp_.connect());

        dsp_.call("recvDataInfo", dataName, this->data_.size(), dim);
        dsp_.call("recvIndexInfo", dataName, plainIndex.size());

        vector<rest_rpc::future_result<rest_rpc::req_result>> futs(encDataBlocks.size() + indexBlocks.size());
        // 分批发送数据
        for(size_t i = 0; i < encDataBlocks.size(); ++i)
        {
            futs[i] = dsp_.async_call<FUTURE>("recvData", dataName, encDataBlocks[i]);
        }

        // 分批发送索引
        for(size_t i = 0; i < indexBlocks.size(); ++i)
        {
            futs[i + encDataBlocks.size()] = dsp_.async_call<FUTURE>("recvIndex", dataName, indexBlocks[i]);
        }

        // wait until finish
        for(auto& fut: futs){
            fut.get();
        }
        info("DO outsource end");
    }

    vector<pair<size_t, vector<string>>> encryptData(){
        vector<pair<size_t, vector<string>>> encData(this->data_.size()); // pair<id, vector<string of integer>>
        for(size_t i = 0; i < encData.size(); ++i){
            encData[i].first = this->data_[i].id;
            encData[i].second.resize(dim);
            for(size_t j = 0; j < dim; ++j){
                encData[i].second[j] = this->crypto->encrypt(this->data_[i].key[j]).data.get_str();
            }
        }
        return encData;
    }

    void saveData(const string& filePath, vector<pair<size_t, vector<string>>>& encData){
        std::ofstream out(filePath, std::ios::out | std::ios::binary);
        if(!out.is_open()){
            info("open %s failed!", filePath.c_str());
            return;
        }
        out << encData.size() << ' ' << dim << ' ';


        std::for_each(encData.begin(), encData.end(), [&out, this](const pair<size_t, vector<string>>& item){
            out << item.first << ' ';
            for(size_t i = 0; i < item.second.size(); ++i){
                out << item.second[i] << ' ';
            }
        });
        out.close();
    }

    void saveIndex(const string& filePath, vector<string>& index){
        std::ofstream out(filePath, std::ios::out | std::ios::binary);
        if(!out.is_open()){
            info("open %s failed!", filePath.c_str());
            return;
        }
        out << index.size() << ' ';

        for(size_t i = 0; i < index.size(); ++i){
            out << seriEncSegmentNode(*deSeriEncSegmentNode(index[i])) << ' ';
        }
        
        out.close();
    }

    vector<string> buildIndex(){
        // build segment nodes
        std::vector<pair<interval<uint64_t, closed>, std::shared_ptr<pair<linearModel, linearModel>>>> segNodes = buildTree();

        // padding nodes and make them a binary tree
        plainIndex = paddingIntervaltree(segNodes);

        // encrypt
        return encryptIntervaltree(plainIndex);
    }

    std::vector<pair<interval<uint64_t, closed>, std::shared_ptr<pair<linearModel, linearModel>>>> buildTree(){
        vector<int64_t> keys(data_.size());
        for(size_t idx = 0; idx < data_.size(); ++idx){
            keys[idx] = mortonTable_[data_[idx].id];
        }

        vector<segment<uint64_t, size_t, double>> segs = pgm::transform(keys, pgm::fit<int64_t, size_t>(keys, epsilon_));
        //vector<segment<uint64_t, size_t, double>> segs = shringkingCone<int64_t, size_t>(keys, epsilon_);
        //debug("size of fitting segments is %lu, epsilon is %lu", segs.size(), epsilon_);
        std::vector<pair<interval<uint64_t, closed>, std::shared_ptr<pair<linearModel, linearModel>>>> segNodes = paddingSegments(segs);

        return segNodes;
    }

    void recurFilling(std::vector<pair<interval<uint64_t, closed>, std::shared_ptr<pair<linearModel, linearModel>>>>& segNodes, vector<interval_tree_t<uint64_t>::node_type*>& nodevec, size_t rootIdx, size_t start, size_t end)
    {
        if(start > end) return;
        using node_t = interval_tree_t<uint64_t>::node_type;
        size_t segIdx = (start + end) / 2;
        nodevec[rootIdx] = new interval_tree_t<uint64_t>::node_type(rootIdx == 0 ? nullptr : nodevec[size_t((rootIdx - 1) / 2)], std::move(segNodes[segIdx].first));
        if(rootIdx != 0)
        {
            if(rootIdx == 2 * size_t((rootIdx - 1) / 2) + 1) nodevec[size_t((rootIdx - 1) / 2)]->left_ = nodevec[rootIdx];
            else nodevec[size_t((rootIdx - 1) / 2)]->right_ = nodevec[rootIdx];
        }
        nodevec[rootIdx]->data_ptr = segNodes[segIdx].second;

        if(segIdx > 1) recurFilling(segNodes, nodevec, 2 * rootIdx + 1, start, segIdx - 1);
        recurFilling(segNodes, nodevec, 2 * rootIdx + 2, segIdx + 1, end);
    }

    void postOrder(node_t* root, const std::function<void(node_t*)>& function)
    {
        if(!root) return;
        postOrder(root->left_, function);
        postOrder(root->right_, function);
        if(!root->left_ && !root->right_){
            function(root);
        }
    }

    vector<interval_tree_t<uint64_t> ::node_type*> paddingIntervaltree(std::vector<pair<interval<uint64_t, closed>, std::shared_ptr<pair<linearModel, linearModel>>>>& segNodes)
    {
        int height = std::ceil(std::log2(segNodes.size() + 1));
        vector<node_t*> nodevec(std::pow(2, height) - 1, nullptr);
        recurFilling(segNodes, nodevec, 0, 0, segNodes.size() - 1);
        {
            lib_interval_tree::interval_tree_t<uint64_t> tree;
            postOrder(nodevec[0], [&tree](node_t* root){
                tree.recalculate_max(root);
            });
        }
        
        // 扩充为满二叉树的同时保留结点间关系
        // make these nodes a binary tree
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
                nodevec[i]->max_ = 0;
                nodevec[i]->data_ptr = std::make_shared<pair<linearModel, linearModel>>(linearModel{0, 0}, linearModel{0, 0});
            }
        }
        return nodevec;
    }
    
    // 将树转为满二叉树
    vector<interval_tree_t<uint64_t> ::node_type*> paddingIntervaltree(interval_tree_t<uint64_t>& tree) {
        int height = getHeight(tree.root_);
        //debug("height of interval tree is %d", height);
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
        size_t mem = 0;
        // 加密并序列化节点参数
        for(size_t i = 0; i < encNodes.size(); ++i){
            auto node = encSegmentNode(*(this->crypto), nodevec[i]);
            mem += node.low_.data.size_bits() / 8;
            mem += node.high_.data.size_bits() / 8;
            mem += node.max_.data.size_bits() / 8;
            mem += node.pred1.first.data.size_bits() / 8;
            mem += node.pred1.second.data.size_bits() / 8;
            mem += node.pred2.first.data.size_bits() / 8;
            mem += node.pred2.second.data.size_bits() / 8;
            // printEncSegmentNode(crypto, &node);
            encNodes[i] = seriEncSegmentNode(node);
        }
        // printf("mem=%lu\n", mem);
        return encNodes;
    }

    void recvKeys(){
        rpc_client cli(config::CA_IP, config::CA_PORT);
        while(!cli.connect());
        size_t keySize = cli.call<size_t>("getKeySize");
        pair<string, string> param = cli.call<pair<string, string>>("getPub");
        array<string, 4> param2 = cli.call<array<string, 4>>("getPriv");
        PublicKey pk(keySize, Integer(param.first.c_str()), Integer(param.second.c_str()));
        PrivateKey sk(keySize, std::stoul(param2[0]), Integer(param2[1].c_str()), Integer(param2[2].c_str()), Integer(param2[3].c_str()));

        crypto = new PaillierFast(pk, sk);
        info("recv keys from %s:%d", config::CA_IP.c_str(), config::CA_PORT);
    }

    // accord morton code 'val' calc its position
    size_t caculateNewId(uint64_t val){
        size_t idx = 0;

        // for(size_t i = 0; i < buffer.size(); ++i){
        //     if(buffer[i].first < val) idx++;
        //     else if(buffer[i].first == val){
        //         return -1; // repeat then return
        //     }
        // }

        auto iter = std::lower_bound(data_.begin(), data_.end(), val, [this](const record<dim>& a, uint64_t val){
            return this->mortonTable_[a.id] < val;
        });

        if(iter == data_.end()){
            return data_.size();
        }

        if(this->mortonTable_[iter->id] == val){
            return -1; // repeat
        }

        return iter - data_.begin();
    }

    // find seg node idx which covers key(morton code)
    size_t searchSegNodeIdx(uint64_t val){
        size_t cur = 0;
        while(cur < plainIndex.size()){
            if(val >= plainIndex[cur]->low() && val <= plainIndex[cur]->high()){
                break;
            }

            if(!plainIndex[cur]->left()) return cur;
            else if(val <= plainIndex[cur]->left()->max()){
                cur = 2 * cur + 1;
            }else{
                cur = 2 * cur + 2;
            }
        }
        return cur;
    }

    // insert k-v record
    STATUS insert(const array<uint32_t, dim>& item){
        uint64_t val = morton_code<dim, int(64/dim)>::encode(item);
        // find item's position and check duplicate
        size_t dataIdx = caculateNewId(val);
        if(dataIdx == size_t(-1)){
            return DUPLICATE;
        }

        record<dim> newRecord{data_.size(), item};
        this->mortonTable_[newRecord.id] = val;
        data_.insert(data_.begin() + dataIdx, newRecord);

        // search affected segments
        size_t nodeIdx = searchSegNodeIdx(val);
        // gen a new seg node
        {
            size_t leftDataIdx = dataIdx, rightDataIdx = dataIdx;
            for(; leftDataIdx > 0 && this->mortonTable_[data_[leftDataIdx].id] > plainIndex[nodeIdx]->low(); --leftDataIdx);
            for(; rightDataIdx < data_.size() && this->mortonTable_[data_[rightDataIdx].id] < plainIndex[nodeIdx]->high(); ++rightDataIdx);
            
            vector<int64_t> keys(rightDataIdx - leftDataIdx + 1);
            for(size_t idx = 0; idx < keys.size(); ++idx){
                keys[idx] = mortonTable_[data_[leftDataIdx + idx].id];
            }

            vector<segment<uint64_t, size_t, double>> segs = pgm::transform(keys, pgm::fit<int64_t, size_t>(keys, epsilon_));
            size_t inc = 1;
            while(segs.size() > 2){
                segs = pgm::transform(keys, pgm::fit<int64_t, size_t>(keys, epsilon_ + inc));
                inc *= 2;
            }

            node_t* newNode = new node_t(plainIndex[nodeIdx]->parent_, node_t::interval_type{plainIndex[nodeIdx]->low(), plainIndex[nodeIdx]->high()});
            newNode->data_ptr = std::make_shared<pair<linearModel, linearModel>>(
                    linearModel{segs[0].slope, Integer(segs[0].pos) - Integer(uint64_t(segs[0].slope * double(segs[0].start))) - Integer(leftDataIdx)},
                    linearModel{segs[0].slope, Integer(segs[0].pos) - Integer(uint64_t(segs[0].slope * double(segs[0].start))) - Integer(leftDataIdx)}
            );
            newNode->left_ = plainIndex[nodeIdx]->left_;
            newNode->right_ = plainIndex[nodeIdx]->right_;
            if(plainIndex[nodeIdx]->parent_ && plainIndex[nodeIdx]->parent_->left_ == plainIndex[nodeIdx]){
                plainIndex[nodeIdx]->parent_->left_ = newNode;
            }else if(plainIndex[nodeIdx]->parent_ && plainIndex[nodeIdx]->parent_->right_ == plainIndex[nodeIdx]){
                plainIndex[nodeIdx]->parent_->right_ = newNode;
            }
            auto old = plainIndex[nodeIdx];
            plainIndex[nodeIdx] = newNode;
            delete old;
        }

        if(!dsp_.has_connected()){
            while(!dsp_.connect());
        }

        pair<size_t, vector<string>> encItem(newRecord.id, vector<string>(dim)); // encrypt data item and pack it into a string
        for(size_t i = 0; i < dim; ++i) encItem.second[i] = crypto->encrypt(Integer(newRecord.key[i])).data.get_str();
        string seriEncSegNode = seriEncSegmentNode(encSegmentNode(*(this->crypto), plainIndex[nodeIdx]));
        // info("before dsp insert");
        int status = dsp_.call<int>("insert", dataName, dataIdx, encItem, nodeIdx, seriEncSegNode);
        // info("after dsp insert");
        return STATUS(status);
    }

};

class DSP : public rpc_server{
public:
    DSP() : rpc_server(config::DSP_PORT, config::NUM_THREADS), crypto(nullptr), dap_(config::DAP_IP, config::DAP_PORT){
        recvKeys();
        
        dap_.enable_auto_reconnect(true);
        dap_.enable_auto_heartbeat(true);
        while(!dap_.connect());
        info("connect to dap %s:%lu", config::DAP_IP.c_str(), config::DAP_PORT);

        this->threshold = std::thread::hardware_concurrency() * config::NUM_ONE_BATCH * 10;

        this->register_handler("recvData", &DSP::recvData, this);
        this->register_handler("recvIndex", &DSP::recvIndex, this);
        this->register_handler("insert", &DSP::insert, this);
        this->register_handler("rangeQuery", &DSP::rangeQuery, this);
        this->register_handler("rangeQueryOpt", &DSP::rangeQueryOpt, this);
        this->register_handler("recvDataInfo", &DSP::recvDataInfo, this);
        this->register_handler("recvIndexInfo", &DSP::recvIndexInfo, this);

    }

    void recvKeys(){
        rpc_client cli(config::CA_IP, config::CA_PORT);
        while(!cli.connect());
        size_t keySize = cli.call<size_t>("getKeySize");
        pair<string, string> param = cli.call<pair<string, string>>("getPub");
        array<string, 4> param2 = cli.call<array<string, 4>>("getPriv");
        PublicKey pk(keySize, Integer(param.first.c_str()), Integer(param.second.c_str()));
        PrivateKey sk(keySize, std::stoul(param2[0]), Integer(param2[1].c_str()), Integer(param2[2].c_str()), Integer(param2[3].c_str()));

        crypto = new PaillierFast(pk, sk);
        info("recv keys from %s:%d", config::CA_IP.c_str(), config::CA_PORT);
    }

    void recvDataInfo(rpc_conn conn, const string& dataName, const size_t dataSize, const uint32_t dim)
    {
        lock_guard g(m_n2d);
        this->name2data[dataName].resize(dataSize);
        // init delmap
        this->name2delmap[dataName] = BitMap<uint64_t>(dataSize + dataSize);
        auto iter = this->name2data[dataName].begin();
        for(size_t i = 0; i < dataSize; ++i, ++iter){
            iter->second.resize(dim);
        }
    }

    void recvIndexInfo(rpc_conn conn, const string& dataName, const size_t indexSize)
    {
        lock_guard g(m_n2i);
        this->name2index[dataName].resize(indexSize);
    }

    void recvData(rpc_conn conn, const string& dataName, pair<pair<size_t, size_t>, string>&& encData)
    {
        string remoteIP = conn.lock()->remote_address();
        //debug("from %s recv dataset %s , block from %lu to %lu", remoteIP.c_str(), dataName.c_str(), encData.first.first, encData.first.second);
        stringstream ss(encData.second);
        auto iter = this->name2data[dataName].begin();
        std::advance(iter, encData.first.first);
        for(size_t i = encData.first.first; i < encData.first.second; ++i, ++iter)
        {
            ss >> iter->first;
            // this->id2data[dataName][iter->first] = iter;
            string buf;
            for(size_t j = 0; j < iter->second.size(); ++j){
                ss >> buf;
                iter->second[j] = Integer(buf.c_str());
            }
        }
        this->name2count[dataName] += encData.first.second - encData.first.first;
        if(name2count[dataName] == name2data[dataName].size() + name2index[dataName].size()){
            info("recv dataset %s from %s", dataName.c_str(), remoteIP.c_str());
            // check dat and index file exists
            save(dataName);
        }
    }


    void recvIndex(rpc_conn conn, const string& dataName, pair<pair<size_t, size_t>, string>&& index){
        string remoteIP = conn.lock()->remote_address();
        //debug("from %s recv index for dataset %s , block from %lu to %lu", remoteIP.c_str(), dataName.c_str(), index.first.first, index.first.second);

        stringstream ss(index.second);
        for(size_t i = index.first.first; i < index.first.second; ++i)
        {
            string buf;
            ss >> buf;
            this->name2index[dataName][i].reset(deSeriEncSegmentNode(buf));
        }
        if(0)
        {
            for(size_t i = index.first.first; i < index.first.second; ++i)
            {
                printEncSegmentNode(crypto, this->name2index[dataName][i].get());
            }
        }
        this->name2count[dataName] += index.first.second - index.first.first;

        if(name2count[dataName] == name2data[dataName].size() + name2index[dataName].size()){
            info("recv dataset %s from %s", dataName.c_str(), remoteIP.c_str());
            // check dat and index file exists
            save(dataName);
        }
    }

    void save(const string& dataName){
        //debug("save dataset %s", dataName.c_str());
        if(!checkFileExist(config::dataFolder + "/" + dataName + ".dat")){
            if(config::PARAL){
                std::thread t(&DSP::saveData, this, dataName, config::dataFolder + "/" + dataName + ".dat");
                t.detach();
            }else{
                saveData(dataName, config::dataFolder + "/" + dataName + ".dat");
            }
        }
        if(!checkFileExist(config::dataFolder + "/" + dataName + ".idx")){
            if(config::PARAL){
                std::thread t(&DSP::saveIndex, this, dataName, config::dataFolder + "/" + dataName + ".idx");
                t.detach();
            }else{
                saveIndex(dataName, config::dataFolder + "/" + dataName + ".idx");
            }
        }
    }

    void load(const string& dataName){
        if(checkFileExist(config::dataFolder + "/" + dataName + ".dat")){
            // std::thread t(&DSP::loadData, this, dataName, config::dataFolder + "/" + dataName + ".dat");
            // if(t.joinable()) t.join();
            loadData(dataName, config::dataFolder + "/" + dataName + ".dat");
        }else{
            printf("no such data %s\n", dataName.c_str());
            return;
        }
        if(checkFileExist(config::dataFolder + "/" + dataName + ".idx")){
            // std::thread t(&DSP::loadIndex, this, dataName, config::dataFolder + "/" + dataName + ".idx");
            // if(t.joinable()) t.join();
            loadIndex(dataName, config::dataFolder + "/" + dataName + ".idx");
        }
        printf("load %s done\n", dataName.c_str());
    }

    void saveData(const string& dataName, const string& filePath){
        std::ofstream out(filePath, std::ios::out | std::ios::binary);
        if(!out.is_open()){
            info("open %s failed!", filePath.c_str());
            return;
        }
        std::lock(m_n2d, m_n2dm);
        out << name2data[dataName].size() - name2delmap[dataName].getCount() << ' ' << name2data[dataName].begin()->second.size() << ' ';


        std::for_each(name2data[dataName].begin(), name2data[dataName].end(), [&out, dataName, this](const pair<size_t, vector<Ciphertext>>& item){
            if(this->name2delmap[dataName].test(item.first)) return;
            out << item.first << ' ';
            for(size_t i = 0; i < item.second.size(); ++i){
                out << item.second[i].data.get_str() << ' ';
            }
        });
        out.close();
    }

    void saveIndex(const string& dataName, const string& filePath){
        std::ofstream out(filePath, std::ios::out | std::ios::binary);
        if(!out.is_open()){
            info("open %s failed!", filePath.c_str());
            return;
        }
        lock_guard<mutex> g(m_n2i);
        out << name2index[dataName].size() << ' ';
        std::for_each(name2index[dataName].begin(), name2index[dataName].end(), [&out](const shared_ptr<encSegmentNode>& node){
            out << seriEncSegmentNode(*node) << ' ';
        });
        out.close();
    }

    void loadData(const string& dataName, const string& filePath){
        std::ifstream in(filePath, std::ios::in | std::ios::binary);
        if(!in.is_open()){
            info("open %s failed!", filePath.c_str());
            return;
        }
        std::lock(m_n2d, m_i2d);
        size_t size;
        uint32_t dim;
        in >> size >> dim;
        name2data[dataName].resize(size);
        auto iter = name2data[dataName].begin();
        string str;
        for(size_t i = 0; i < size; ++i, ++iter){
            in >> iter->first;
            // id2data[dataName][iter->first] = iter;
            iter->second.resize(dim);
            for(size_t j = 0; j < dim; ++j){
                in >> str;
                iter->second[j].data = Integer(str.c_str());
            }
        }
        name2count[dataName] += size;
        in.close();
    }

    void loadIndex(const string& dataName, const string& filePath){
        std::ifstream in(filePath, std::ios::in | std::ios::binary);
        if(!in.is_open()){
            info("open %s failed!", filePath.c_str());
            return;
        }
        lock_guard<mutex> g(m_n2i);
        size_t size;
        in >> size;
        name2index[dataName].resize(size);
        std::for_each(name2index[dataName].begin(), name2index[dataName].end(), [&in](shared_ptr<encSegmentNode>& item){
            string str;
            in >> str;
            item.reset(deSeriEncSegmentNode(str));
        });

        name2count[dataName] += size;
        in.close();
    }

    Ciphertext checkData(const string& dataName, const EQueryRectangle& QR, const size_t& posIdx){
        // {
        //     lock_guard g(m_n2dm);
        //     if(name2delmap[dataName].test(id)) return crypto->encrypt(0);
        // }
        Ciphertext res = crypto->encrypt(0);
        auto iter = this->name2data[dataName].begin();
        std::advance(iter, posIdx);
        for(size_t i = 0; i < iter->second.size(); ++i){
            Ciphertext cmp = SM(SMinus(iter->second[i], QR.get_minvec()[i]), SMinus(iter->second[i], QR.get_maxvec()[i]));
            res.data *= SIC(crypto->encrypt(0), cmp).data;
        }
        return res;
	}

    size_t predict(const string& dataName, const Ciphertext& input, bool whichPred){
        Ciphertext encPos = CAP(name2index[dataName], input, whichPred);
        Integer pos(dap_.call<string>("DEC", encPos.data.get_str()).c_str());
        // Integer pos(this->DEC(encPos.data.get_str()).c_str());

        pos /= Integer(10).pow(config::FLOAT_EXP);
        if(whichPred == 0) pos -= Integer(config::EPSILON);
        else pos += Integer(config::EPSILON);
        pos = pos < 0 ? 0 : pos;
        pos = pos < Integer(name2data[dataName].size()) ? pos : Integer(name2data[dataName].size()) - 1;
        return pos.to_ulong();
    }

    pair<size_t, size_t> predictRange(const string& dataName, const pair<Ciphertext, Ciphertext>& range){
        pair<size_t, size_t> posInfo;

        std::future<size_t> fut1 = std::async(&DSP::predict, this, dataName, range.first, 0);
        std::future<size_t> fut2 = std::async(&DSP::predict, this, dataName, range.second, 1);

        posInfo.first = fut1.get();
        posInfo.second = fut2.get();

        return posInfo;
    }

    vector<pair<std::future<Vec<PackedCiphertext>>, pair<size_t, size_t>>> genSearchTasks(const string& dataName, const pair<size_t, size_t>& posInfo, const EQueryRectangle& EQR){
        debug("[%lu, %lu]", posInfo.first, posInfo.second);
        // parallel pack searching
        size_t numTasks = (posInfo.second - posInfo.first + 1 - 1) / config::NUM_ONE_BATCH + 1;
        vector<pair<std::future<Vec<PackedCiphertext>>, pair<size_t, size_t>>> futs(numTasks);
        for(size_t i = 0; i < numTasks; ++i){
            futs[i].second = pair<size_t, size_t>(posInfo.first + i * config::NUM_ONE_BATCH, std::min(posInfo.first + (i + 1) * config::NUM_ONE_BATCH - 1, posInfo.second));
            futs[i].first = std::async(&DSP::packSearching, this, dataName, futs[i].second, EQR);
        }
        return futs;
    }

    void unpack(vector<size_t>& results, pair<std::future<Vec<PackedCiphertext>>, pair<size_t, size_t>>& fut, const string& dataName){
        size_t dim = name2data[dataName].begin()->second.size();
        Vec<PackedCiphertext> packedCmpResult = fut.first.get();
        Vec<Integer> cmpResult = Vector::decrypt_pack(packedCmpResult, *crypto);
        // debug("[%lu, %lu], cmp size = %lld", fut.second.first, fut.second.second, cmpResult.length());
        for(size_t posIdx = fut.second.first; posIdx <= fut.second.second; ++posIdx){
            if(cmpResult[posIdx - fut.second.first] == 2 * dim){
                results.push_back(posIdx);
            }
        }
        // debug("unpack done");
    }

    void split(const string& dataName, vector<pair<size_t, size_t>>& posInfos, const pair<size_t, size_t>& posInfo, const EQueryRectangle& EQR){
        if(posInfo.first > posInfo.second) return;
        if(posInfo.second - posInfo.first + 1 <= threshold){
            debug("[%lu, %lu]", posInfo.first, posInfo.second);
            posInfos.emplace_back(posInfo.first, posInfo.second);
        }else{
            auto splitedEQRs = splitQR(EQR);
            Ciphertext firstEnd = encode(splitedEQRs.first.get_maxvec());
            Ciphertext secondStart = encode(splitedEQRs.second.get_minvec());
            
            std::future<size_t> fut1 = std::async(&DSP::predict, this, dataName, firstEnd, 1);
            std::future<size_t> fut2 = std::async(&DSP::predict, this, dataName, secondStart, 0);

            size_t firstEndPos = fut1.get();
            size_t secondStartPos = fut2.get();

            split(dataName, posInfos, pair<size_t, size_t>{posInfo.first, firstEndPos}, splitedEQRs.first);
            split(dataName, posInfos, pair<size_t, size_t>{secondStartPos, posInfo.second}, splitedEQRs.second);
        }
    }

    pair<EQueryRectangle, EQueryRectangle> splitQR(const EQueryRectangle& EQR){
        size_t dim = EQR.dim;
        Integer r = Random::instance().rand_int(10) % Integer(dim);
        pair<EQueryRectangle, EQueryRectangle> ret{EQueryRectangle(dim), EQueryRectangle(dim)};
        Ciphertext midVal = divide(EQR.get_minvec()[r.to_uint()].data * EQR.get_maxvec()[r.to_uint()].data, 2);
        for(size_t i = 0; i < dim; ++i){
            if(i == r){
                ret.first.set_minvec(i, EQR.get_minvec()[i]);
                ret.first.set_maxvec(i, midVal);
                ret.second.set_minvec(i, midVal);
                ret.second.set_minvec(i, EQR.get_maxvec()[i]);
            }else{
                ret.first.set_minvec(i, EQR.get_minvec()[i]);
                ret.first.set_maxvec(i, EQR.get_maxvec()[i]);
                ret.second.set_minvec(i, EQR.get_minvec()[i]);
                ret.second.set_maxvec(i, EQR.get_maxvec()[i]);
            }
        }
        return ret;
    }

    pair<int, vector<size_t>> rangeQuery(rpc_conn conn, const string& dataName, const pair<string, string>& range, const vector<string>& limits){
        info("recv query request from %s", conn.lock()->remote_address().c_str());
        if(!name2index.count(dataName)){
            return {NOSUCHDATA, {}};
        }
        if(name2count[dataName] != name2data[dataName].size() + name2index[dataName].size()){
            return {NOTCOMPLATE, {}};
        }

        pair<Ciphertext, Ciphertext> encRange{Ciphertext(Integer(range.first.c_str())), Ciphertext(Integer(range.second.c_str()))};
        EQueryRectangle EQR(limits);
        // predict
        // tc.update();
        pair<size_t, size_t> posInfo = predictRange(dataName, encRange);
        // debug("first stage %.3f", tc.getTimerMilliSec());
        // result empty
        if(posInfo.first > posInfo.second)
        {
            return {SUCCESS, {}};
        }

        // debug("search range is [%lu, %lu]", posInfo.first, posInfo.second);

        // if search range over threshold, then split, else pack search
        vector<size_t> results;
        if(posInfo.second - posInfo.first + 1 <= threshold){
            vector<pair<std::future<Vec<PackedCiphertext>>, pair<size_t, size_t>>> futs = genSearchTasks(dataName, posInfo, EQR);          
            // debug("gen done");
            for(size_t i = 0; i < futs.size(); ++i){
                unpack(results, futs[i], dataName);
            }
        }else{
            vector<pair<size_t, size_t>> posInfos;
            // split(dataName, posInfos, posInfo, EQR);
            posInfos.emplace_back(posInfo.first, posInfo.second);
            vector<vector<pair<std::future<Vec<PackedCiphertext>>, pair<size_t, size_t>>>> futs(posInfos.size());
            for(size_t i = 0; i < posInfos.size(); ++i){
                futs[i] = genSearchTasks(dataName, posInfos[i], EQR);
            }
            for(size_t i = 0; i < posInfos.size(); ++i){
                for(size_t j = 0; j < futs[i].size(); ++j){
                    unpack(results, futs[i][j], dataName);
                }
            }
        }
        pair<int, vector<size_t>> ret{SUCCESS, std::move(results)};
        return ret;
    }

    pair<int, vector<size_t>> local_rangeQuery(const string& dataName, const pair<string, string>& range, const vector<string>& limits){
        if(!name2index.count(dataName)){
            return {NOSUCHDATA, {}};
        }
        if(name2count[dataName] != name2data[dataName].size() + name2index[dataName].size()){
            return {NOTCOMPLATE, {}};
        }

        pair<Ciphertext, Ciphertext> encRange{Ciphertext(Integer(range.first.c_str())), Ciphertext(Integer(range.second.c_str()))};
        EQueryRectangle EQR(limits);
        // predict
        // tc.update();
        pair<size_t, size_t> posInfo = predictRange(dataName, encRange);
        // debug("first stage %.3f, [%lu, %lu]", tc.getTimerMilliSec(), posInfo.first, posInfo.second);
        // result empty
        if(posInfo.first > posInfo.second)
        {
            return {SUCCESS, {}};
        }

        // debug("search range is [%lu, %lu]", posInfo.first, posInfo.second);

        // if search range over threshold, then split, else pack search
        vector<size_t> results;
        EQueryRectangle QR(limits);
        size_t dim = limits.size() / 2;
        for(size_t i = posInfo.first; i <= posInfo.second; ++i){
            Ciphertext flag = checkData(dataName, QR, i);
            int mf = Integer(this->DEC(flag.data.get_str()).c_str()).to_int();
            if(mf == dim) results.push_back(i);
        }
        pair<int, vector<size_t>> ret{SUCCESS, std::move(results)};
        return ret;
    }

    pair<int, vector<size_t>> local_rangeQueryOpt(const string& dataName, const pair<string, string>& range, const vector<string>& limits){
        if(!name2index.count(dataName)){
            return {NOSUCHDATA, {}};
        }
        if(name2count[dataName] != name2data[dataName].size() + name2index[dataName].size()){
            return {NOTCOMPLATE, {}};
        }

        pair<Ciphertext, Ciphertext> encRange{Ciphertext(Integer(range.first.c_str())), Ciphertext(Integer(range.second.c_str()))};
        EQueryRectangle EQR(limits);
        // predict
        // tc.update();
        pair<size_t, size_t> posInfo = predictRange(dataName, encRange);
        // debug("first stage %.3f", tc.getTimerMilliSec());
        // result empty
        if(posInfo.first > posInfo.second)
        {
            return {SUCCESS, {}};
        }

        // debug("search range is [%lu, %lu]", posInfo.first, posInfo.second);

        // if search range over threshold, then split, else pack search
        vector<size_t> results;
        if(posInfo.second - posInfo.first + 1 <= threshold){
            vector<pair<std::future<Vec<PackedCiphertext>>, pair<size_t, size_t>>> futs = genSearchTasks(dataName, posInfo, EQR);          
            // debug("gen done");
            for(size_t i = 0; i < futs.size(); ++i){
                unpack(results, futs[i], dataName);
            }
        }else{
            vector<pair<size_t, size_t>> posInfos;
            // split(dataName, posInfos, posInfo, EQR);
            posInfos.emplace_back(posInfo.first, posInfo.second);
            vector<vector<pair<std::future<Vec<PackedCiphertext>>, pair<size_t, size_t>>>> futs(posInfos.size());
            for(size_t i = 0; i < posInfos.size(); ++i){
                futs[i] = genSearchTasks(dataName, posInfos[i], EQR);
            }
            for(size_t i = 0; i < posInfos.size(); ++i){
                for(size_t j = 0; j < futs[i].size(); ++j){
                    unpack(results, futs[i][j], dataName);
                }
            }
        }
        pair<int, vector<size_t>> ret{SUCCESS, std::move(results)};
        return ret;
    }

    Vec<PackedCiphertext> packSearching(const string& dataName, const pair<size_t, size_t>& posInfo, const EQueryRectangle& EQR)
    {
        //debug("packing SVC start");
        size_t dim = name2data[dataName].begin()->second.size();
        size_t n = posInfo.second - posInfo.first + 1;
        // 将数据以列重新排列
        vector<Vec<Ciphertext>> encData(dim, Vec<Ciphertext>(NTL::INIT_SIZE_TYPE{}, n));
        vector<Vec<Ciphertext>> lowLimits(dim);
        vector<Vec<Ciphertext>> highLimits(dim);

        list<pair<size_t, vector<Ciphertext>>>::iterator iter = name2data[dataName].begin();
        std::advance(iter, posInfo.first);
        for(size_t i = posInfo.first; i <= posInfo.second; ++i, ++iter){
            for(size_t j = 0; j < dim; ++j){
                encData[j][i - posInfo.first] = iter->second[j];
            }
        }

        for(size_t i = 0; i < dim; ++i){
            lowLimits[i] = std::move(Vec<Ciphertext>(NTL::INIT_SIZE_TYPE{}, n, EQR.get_minvec()[i]));
            highLimits[i] = std::move(Vec<Ciphertext>(NTL::INIT_SIZE_TYPE{}, n, EQR.get_maxvec()[i]));
        }

        size_t numIntegerPerCiphertext = Vector::pack_count(32, *(this->crypto));
        size_t numPackedCiphertext = size_t(std::ceil(double(n) / double(numIntegerPerCiphertext)));
        Vec<PackedCiphertext> ret(NTL::INIT_SIZE_TYPE{}, numPackedCiphertext, PackedCiphertext(crypto->encrypt(0), numIntegerPerCiphertext, 32));

        // if(!dap_.has_connected()){
        //     //debug("reconnect to dap");
        //     while(!dap_.connect());
        // }
        
        for(size_t i = 0; i < dim; ++i){
            vector<PackedCiphertext> lowCmpRes = SVC(lowLimits[i], encData[i]);
            vector<PackedCiphertext> highCmpRes = SVC(encData[i], highLimits[i]);

            for(size_t j = 0; j < numPackedCiphertext; ++j){
                ret[j].data.data *= lowCmpRes[j].data.data;
                ret[j].data.data *= highCmpRes[j].data.data;
            }
        }

        //debug("packing SVC finish");
        return ret;
    }

    pair<int, vector<size_t>> rangeQueryOpt(rpc_conn conn, const string& dataName, const pair<string, string>& range, const vector<string>& limits){
        info("recv query request from %s", conn.lock()->remote_address().c_str());
        if(!name2index.count(dataName)){
            return {NOSUCHDATA, {}};
        }
        if(name2count[dataName] != name2data[dataName].size() + name2index[dataName].size()){
            return {NOTCOMPLATE, {}};
        }

        EQueryRectangle EQR(limits);
        pair<Ciphertext, Ciphertext> encPosInfo;
        if(config::PARAL){
            std::future<Ciphertext> fut1 = std::async(&DSP::CAP, this, name2index[dataName], Ciphertext(Integer(range.first.c_str())), 0);
            std::future<Ciphertext> fut2 = std::async(&DSP::CAP, this, name2index[dataName], Ciphertext(Integer(range.second.c_str())), 1);
            encPosInfo.first = fut1.get();
            encPosInfo.second = fut2.get();
        }else{
            encPosInfo.first = CAP(name2index[dataName], Ciphertext(Integer(range.first.c_str())), 0);
            encPosInfo.second = CAP(name2index[dataName], Ciphertext(Integer(range.second.c_str())), 1);
        }

        Integer startPos(dap_.call<string>("DEC", encPosInfo.first.data.get_str()).c_str());
        Integer endPos(dap_.call<string>("DEC", encPosInfo.second.data.get_str()).c_str());
        // Integer startPos(this->DEC(encPosInfo.first.data.get_str()).c_str());
        // Integer endPos(this->DEC(encPosInfo.second.data.get_str()).c_str());

        
        //debug("predict posinfo = [%s, %s]", (startPos / Integer(10).pow(config::FLOAT_EXP)).get_str().c_str(), (endPos / Integer(10).pow(config::FLOAT_EXP)).get_str().c_str());

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
        pair<size_t, size_t> posInfo{startPos.to_ulong(), endPos.to_ulong()};

        // debug("search range is [%lu, %lu]", posInfo.first, posInfo.second);
        // result empty
        if(posInfo.first > posInfo.second)
        {
            return {SUCCESS, {}};
        }

        // parallel pack searching
        size_t numTasks = (posInfo.second - posInfo.first + 1 - 1) / config::NUM_ONE_BATCH + 1;
        vector<std::future<Vec<PackedCiphertext>>> futs(numTasks);
        vector<pair<size_t, size_t>> splitedPosInfo(numTasks);
        for(size_t i = 0; i < numTasks; ++i){
            splitedPosInfo[i] = pair<size_t, size_t>(posInfo.first + i * config::NUM_ONE_BATCH, std::min(posInfo.first + (i + 1) * config::NUM_ONE_BATCH - 1, posInfo.second));
            futs[i] = std::async(&DSP::packSearching, this, dataName, splitedPosInfo[i], EQR);
        }

        // vector<size_t> results{};
        // size_t dim = name2data[dataName].begin()->second.size();
        // for(size_t i = 0; i < numTasks; ++i){
        //     Vec<PackedCiphertext> packedCmpResult = futs[i].get();
        //     Vec<Integer> cmpResult = Vector::decrypt_pack(packedCmpResult, *crypto);
        //     for(size_t posIdx = splitedPosInfo[i].first; posIdx <= splitedPosInfo[i].second; ++posIdx){
        //         if(cmpResult[posIdx - splitedPosInfo[i].first] == 2 * dim){
        //             results.push_back(posIdx);
        //         }
        //     }
        // }
        vector<size_t> results{};
        size_t dim = name2data[dataName].begin()->second.size();
        for(size_t i = 0; i < numTasks; ++i){
            Vec<PackedCiphertext> packedCmpResult = futs[i].get();
            Vec<Integer> cmpResult = Vector::decrypt_pack(packedCmpResult, *crypto);
            for(size_t posIdx = splitedPosInfo[i].first; posIdx <= splitedPosInfo[i].second; ++posIdx){
                if(cmpResult[posIdx - splitedPosInfo[i].first] == 2 * dim){
                    results.push_back(posIdx);
                }
            }
        }

        pair<int, vector<size_t>> ret{SUCCESS, std::move(results)};
        return ret;
    }

    // insert an item
    // posIdx: position to insert the item
    // item: <id, vector<Integer>>
    // segIdx: idx of segment node
    // seriEncSegNode: replace the origin 'idx' th segment node
    int insert(rpc_conn conn, const string& dataName, const size_t& posIdx, pair<size_t, vector<string>>&& item, const size_t& segIdx, const string& seriEncSegNode){
        // std::lock(m_n2d, m_i2d);
        pair<size_t, vector<Ciphertext>> tmp{item.first, vector<Ciphertext>(item.second.size())};
        for(size_t i = 0; i < tmp.second.size(); ++i) tmp.second[i] = Integer(item.second[i].c_str());
        auto iter = name2data[dataName].begin();
        std::advance(iter, posIdx);
        auto newIter = name2data[dataName].insert(iter, std::move(tmp));
        // info("insert item success");
        // update index nodes
        name2index[dataName][segIdx].reset(deSeriEncSegmentNode(seriEncSegNode));
        updateNodes(dataName, segIdx);
        // info("update success");
        name2count[dataName]++;
        return SUCCESS;
    }

    void updateNodes(const string& dataName, const size_t& segIdx){
        lock_guard<mutex> g(m_n2i);
        bool update = false;
        recurUpdateNodes(name2index[dataName], 0, segIdx, update);
    }

    void recurUpdateNodes(vector<shared_ptr<encSegmentNode>>& index, size_t curIdx, size_t targetIdx, bool& update){
        if(2 * curIdx + 1 < index.size()) recurUpdateNodes(index, 2 * curIdx + 1, targetIdx, update);
        if(update){
            index[curIdx]->pred1.second.data *= crypto->encrypt(1).data;
            index[curIdx]->pred2.second.data *= crypto->encrypt(1).data;
        }else{
            if(curIdx == targetIdx) update = true;
        }
        if(2 * curIdx + 1 < index.size()) recurUpdateNodes(index, 2 * curIdx + 1, targetIdx, update);
    }

    // delete data item
    int del(rpc_conn conn, const string& dataName, size_t id){
        lock_guard g(m_n2dm);
        name2delmap[dataName].set(id);
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
        // Ciphertext h = Integer(this->SM(tmp_a.data.get_str(), tmp_b.data.get_str()).c_str());

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
        // Ciphertext ret = Integer(this->SIC(c.data.get_str()).c_str());
        if(coin == 0){
            ret = SMinus(crypto->encrypt(1), ret);
        }
        return ret;
    }

    // packed SIC
    vector<PackedCiphertext> SVC(const Vec<Ciphertext>& a, const Vec<Ciphertext>& b){
        //debug("dsp SVC start");
        assert(a.length() == b.length());
        size_t n = a.length();
        size_t numIntegerPerCiphertext = Vector::pack_count(32, *(this->crypto));
        size_t numPackedCiphertext = size_t(std::ceil(double(n) / double(numIntegerPerCiphertext)));
        
        Integer coin = Random::instance().rand_int(crypto->get_pub().n) % Integer(2);
        Vec<Ciphertext> Z(NTL::INIT_SIZE_TYPE{}, n);

        if(coin == 1){
            for(size_t i = 0; i < n; ++i){
                Z[i] = SMinus(_times(a[i], 2), _times(b[i], 2).data * crypto->encrypt(1).data).data * crypto->encrypt(Integer(2).pow(32 - 1)).data;
            }
        }else{
            for(size_t i = 0; i < n; ++i){
                Z[i] = SMinus(_times(b[i], 2).data * crypto->encrypt(1).data, _times(a[i], 2)).data * crypto->encrypt(Integer(2).pow(32 - 1)).data;
            }
        }

        Integer r = Random::instance().rand_int_bits((getMaxBitLength(crypto->get_pub().n) / 4).to_ulong() - 2);

        r = 1;
        for(auto& z : Z){
            z = _times(z, r);
        }

        vector<PackedCiphertext> C(numPackedCiphertext, PackedCiphertext(crypto->encrypt(0), numIntegerPerCiphertext, 32));

        size_t idx = 0;
        for(size_t i = 0; i < numPackedCiphertext; ++i){
            size_t s = idx;
            size_t e = std::min(idx + numIntegerPerCiphertext, n);
            Vec<Ciphertext> packedC(NTL::INIT_SIZE_TYPE{}, numIntegerPerCiphertext, crypto->encrypt(0));
            for(size_t j = s; j < e; ++j, ++idx){
                packedC[j - s].data = Z[idx].data;
            }
            C[i] = Vector::pack_ciphertexts(packedC, 32, *crypto);
        }

        vector<string> seriC(C.size());
        for(size_t i = 0; i < C.size(); ++i){
            seriC[i] = C[i].data.data.get_str();
        }
        vector<string> packedCmpResult = dap_.call<vector<string>>("SVC", seriC);
        // vector<string> packedCmpResult = this->SVC(seriC);

        //debug("call dap SVC finish");

        vector<PackedCiphertext> ret(numPackedCiphertext);
        for(size_t i = 0; i < numPackedCiphertext; ++i){
            if(coin == 1){
                ret[i] = PackedCiphertext(crypto->encrypt(0), numIntegerPerCiphertext, 32);
                ret[i].data.data = Integer(packedCmpResult[i].c_str());
            }else{
                ret[i] = Vector::encrypt_pack(Vec<Integer>(NTL::INIT_SIZE_TYPE{}, numIntegerPerCiphertext, 1), 32, *crypto);
                ret[i].data.data = SMinus(ret[i].data.data, Integer(packedCmpResult[i].c_str())).data;
            }
        }
        //debug("dsp SVC finish");
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
        // Integer ret(this->DEC(f.data.get_str().c_str()).c_str());
        return ret == 0 ? RIGHT : LEFT;
    }

    
    Ciphertext CAP(const vector<shared_ptr<encSegmentNode>>& index, const Ciphertext& input, bool whichPred = 0){
        Ciphertext P = crypto->encrypt(0);
        Ciphertext F = crypto->encrypt(0);

        size_t cur = 0;
        while(cur < index.size()){
            Ciphertext f = SCO(index[cur], input);
            Ciphertext acc = whichPred == 0 ? SM(index[cur]->pred1.first, input).data * index[cur]->pred1.second.data : SM(index[cur]->pred2.first, input).data * index[cur]->pred2.second.data;
            F.data *= f.data; 
            P.data *= SM(f, acc).data;
            //debug("cur = %lu [%s, %s], F = %s, P = %s", cur, crypto->decrypt(index[cur]->low_).get_str().c_str(), crypto->decrypt(index[cur]->high_).get_str().c_str(), crypto->decrypt(f).get_str().c_str(), (crypto->decrypt(P) / Integer(10).pow(config::FLOAT_EXP)).get_str().c_str());
    
            if(SPFT(index, cur, F, input) == LEFT){
                cur = 2 * cur + 1;
            }else{
                cur = 2 * cur + 2;
            }
        }
        //debug("dsp CAP end");
        return P;
    }

    Ciphertext SVM(const Vec<Ciphertext>& a, const Vec<Ciphertext>& b, size_t plaintext_bits = 100){
        assert(a.length() == b.length());

        Vec<PackedCiphertext> packedA = Vector::pack_ciphertexts_vec(a, plaintext_bits, *crypto);
        Vec<PackedCiphertext> packedB = Vector::pack_ciphertexts_vec(b, plaintext_bits, *crypto);

        vector<string> seriPackedA(packedA.length());
        vector<string> seriPackedB(packedB.length());

        for(size_t i = 0; i < seriPackedA.size(); ++i){
            seriPackedA[i] = packedA[i].data.data.get_str();
            seriPackedB[i] = packedB[i].data.data.get_str();
        }

        string tmp = dap_.call<string>("SVM", seriPackedA, seriPackedB, plaintext_bits);
        // string tmp = this->SVM(seriPackedA, seriPackedB, plaintext_bits);
        return Integer(tmp.c_str());
    }

    Ciphertext divide(const Ciphertext& a, int b){
        string tmp = dap_.call<string>("DEC", a.data.get_str());
        // string tmp = this->DEC(a.data.get_str());
        return crypto->encrypt(Integer(tmp.c_str()) / b);
    }
    Ciphertext encode(const vector<Ciphertext>& key){
        vector<string> keyStr(key.size());
        for(size_t i = 0; i < key.size(); ++i){
            keyStr[i] = key[i].data.get_str();
        }
        string tmp = dap_.call<string>("SE", keyStr);
        // string tmp = this->SE(std::move(keyStr));
        return Ciphertext(Integer(tmp.c_str()));
    }

    /*****************************************************************************************************/
    // to exactly calc time of computing (without communication time and cost), move DAP's functions to DSP
    /*****************************************************************************************************/
    string SM(const string& a, const string& b){
        communicationCost += Integer(a.c_str()).size_bits();
        communicationCost += Integer(b.c_str()).size_bits();

        Integer ha = crypto->decrypt(Integer(a.c_str()) % *crypto->get_n2());
        Integer hb = crypto->decrypt(Integer(b.c_str()) % *crypto->get_n2());
        Integer h = (ha * hb) % crypto->get_pub().n;

        Ciphertext ret = crypto->encrypt(h);
        communicationCost += ret.data.size_bits();
        return ret.data.get_str();
    }

    string SIC(const string& c){
        communicationCost += Integer(c.c_str()).size_bits();
        Integer m = crypto->decrypt(Ciphertext(Integer(c.c_str()))) % crypto->get_pub().n;
        Integer u = 0;
        if(getMaxBitLength(m >= 0 ? m : -m) > getMaxBitLength(crypto->get_pub().n) / 2){
            u = 1;
        }
        Ciphertext ret = crypto->encrypt(u);
        communicationCost += ret.data.size_bits();
        return ret.data.get_str();
    }

    string DEC(const string& c){
        communicationCost += Integer(c.c_str()).size_bits();
        return crypto->decrypt(Integer(c.c_str())).get_str();
    }

    vector<string> SVC(const vector<string>& C){
        //debug("call SVC from %s", conn.lock()->remote_address().c_str());
        //printf("call SVC from %s", conn.lock()->remote_address().c_str());
        vector<Vec<Integer>> M(C.size());
        vector<string> ret(M.size());
        size_t numIntegerPerCiphertext = Vector::pack_count(32, *(this->crypto));
        for(size_t i = 0; i < C.size(); ++i){
            communicationCost += Integer(C[i].c_str()).size_bits();
            PackedCiphertext packed(Integer(C[i].c_str()), numIntegerPerCiphertext, 32);
            M[i] = Vector::decrypt_pack(packed, *(this->crypto));

            // if(config::LOG){
            //     string content;
            //     for(size_t j = 0; j < M[i].length(); ++j){
            //         content += (M[i][j] - Integer(2).pow(32 - 1)).get_str() + " ";
            //     }
            //     //debug(content.c_str());
            // }

            Vec<Integer> packedCmpResult(NTL::INIT_SIZE_TYPE{}, M[i].length(), 0);
            for(size_t j = 0; j < numIntegerPerCiphertext; ++j){
                M[i][j] -= Integer(2).pow(32 - 1);
                if(getMaxBitLength(M[i][j] % crypto->get_pub().n) > getMaxBitLength(crypto->get_pub().n) / 2){
                    packedCmpResult[j] = 1;
                }
            }
            

            if(config::LOG){
                string content;
                for(size_t j = 0; j < packedCmpResult.length(); ++j){
                    content += packedCmpResult[j].get_str() + " ";
                }
                //debug(content.c_str());
            }
            ret[i] = PackedCiphertext(Vector::encrypt_pack(packedCmpResult, 32, *crypto)).data.data.get_str();
            communicationCost += Integer(ret[i].c_str()).size_bits();
        }
        return ret;
    }

    string SVM(const vector<string>& a, const vector<string>& b, size_t plaintext_bits = 100){
        vector<Vec<Integer>> ma(a.size());
        vector<Vec<Integer>> mb(a.size());
        size_t numIntegerPerCiphertext = Vector::pack_count(plaintext_bits, *(this->crypto));
        Integer ret = 0;
        for(size_t i = 0; i < a.size(); ++i){
            PackedCiphertext pa(Integer(a[i].c_str()), numIntegerPerCiphertext, plaintext_bits);
            PackedCiphertext pb(Integer(b[i].c_str()), numIntegerPerCiphertext, plaintext_bits);
            ma[i] = Vector::decrypt_pack(pa, *(this->crypto));
            mb[i] = Vector::decrypt_pack(pb, *(this->crypto));
            for(size_t j = 0; j < ma[i].length(); ++j){
                ret += (ma[i][j] - Integer(2).pow(plaintext_bits - 1)) * (mb[i][j] - Integer(2).pow(plaintext_bits - 1));
            }
        }
        return crypto->encrypt(ret).data.get_str();
    }

    // encoding enc key into morton code
    string SE(vector<string>&& keyStr){
        size_t dim = keyStr.size();
        vector<uint32_t> key(dim);
        for(size_t i = 0; i < dim; ++i){
            key[i] = crypto->decrypt(Ciphertext(Integer(keyStr[i].c_str()))).to_uint();
        }
        uint64_t val;
        switch(dim){
            case 2:{
                val = morton_code<2, int(64 / 2)>::encode(std::array<uint32_t, 2>{key[0], key[1]});
                break;
            }
            case 3:{
                val = morton_code<3, int(64 / 3)>::encode(std::array<uint32_t, 3>{key[0], key[1], key[2]});
                break;
            }
            case 4:{
                val = morton_code<4, int(64 / 4)>::encode(std::array<uint32_t, 4>{key[0], key[1], key[2], key[3]});
                break;
            }
            case 5:{
                val = morton_code<5, int(64 / 5)>::encode(std::array<uint32_t, 5>{key[0], key[1], key[2], key[3], key[4]});
                break;
            }
            case 6:{
                val = morton_code<6, int(64 / 6)>::encode(std::array<uint32_t, 6>{key[0], key[1], key[2], key[3], key[4], key[5]});
                break;
            }
        }
        return crypto->encrypt(val).data.get_str();
    }
    /*****************************************************************************************************/
    /**********************************************end****************************************************/
    /*****************************************************************************************************/


    void query(const string& dataName, const QueryRectangle<uint32_t>& QR){
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
        
        static TimerClock tc;
        tc.update();

        communicationCost = 0;
        pair<int, vector<size_t>> res = this->local_rangeQueryOpt(dataName, encRange, limits);
        debug("comm cost:%lu\n", communicationCost);
        switch(res.first){
            case SUCCESS:{
                // std::cout << "query result size is " << res.second.size() << std::endl;
                info("result size is %lu, time cost is %.3f ms", res.second.size(), tc.getTimerMilliSec());
                break;
            }
            case NOSUCHDATA:{
                std::cout << "no such dataset " << std::endl;
                break;
            }
            case NOTCOMPLATE:{
                std::cout << "not complate " << std::endl;
                break;
            }
        }
    }

public:
    PaillierFast* crypto;
    rpc_client dap_;

    // dataname to data: each item represents a pair of data id and its enc key
    map<string, list<pair<size_t, vector<Ciphertext>>>> name2data;
    // dataname to index
    map<string, vector<shared_ptr<encSegmentNode>>> name2index;
    // data id to iterator of data: id -> iterator of data item
    // map<string, map<size_t, list<pair<size_t, vector<Ciphertext>>>::iterator>> id2data;
    // bitmap for deleted flags: id -> {0, 1} (1 represents deleted)
    map<string, BitMap<uint64_t>> name2delmap;

    // lock for above data structs
    mutex m_n2d;
    mutex m_n2i;
    mutex m_i2d;
    mutex m_n2dm;

    // actually recv how many data items and index nodes
    map<string, std::atomic<size_t>> name2count;

    // threshold of search range
    size_t threshold;

    // communication cost
    size_t communicationCost = 0;
};

class DAP : public rpc_server{
public:

    DAP() : rpc_server(config::DAP_PORT, config::NUM_THREADS), dsp_(config::DSP_IP, config::DSP_PORT){
        recvKeys();

        // regist
        this->register_handler("SM", &DAP::SM, this);
        this->register_handler("SIC", &DAP::SIC, this);
        this->register_handler("DEC", &DAP::DEC, this);
        this->register_handler("SVC", &DAP::SVC, this);
        this->register_handler("SVM", &DAP::SVM, this);
        this->register_handler("SE", &DAP::SE, this);
    }


    string SM(rpc_conn conn, const string& a, const string& b){
        info("call SM from %s", conn.lock()->remote_address().c_str());
        //printf("call SM from %s", conn.lock()->remote_address().c_str());
        Integer ha = crypto->decrypt(Integer(a.c_str()) % *crypto->get_n2());
        Integer hb = crypto->decrypt(Integer(b.c_str()) % *crypto->get_n2());
        Integer h = (ha * hb) % crypto->get_pub().n;
        return crypto->encrypt(h).data.get_str();
    }

    string SIC(rpc_conn conn, const string& c){
        info("call SIC from %s", conn.lock()->remote_address().c_str());
        //printf("call SIC from %s", conn.lock()->remote_address().c_str());
        Integer m = crypto->decrypt(Ciphertext(Integer(c.c_str()))) % crypto->get_pub().n;
        Integer u = 0;
        if(getMaxBitLength(m >= 0 ? m : -m) > getMaxBitLength(crypto->get_pub().n) / 2){
            u = 1;
        }
        return crypto->encrypt(u).data.get_str();
    }

    string DEC(rpc_conn conn, const string& c){
        //debug("call DEC from %s", conn.lock()->remote_address().c_str());
        //printf("call DEC from %s", conn.lock()->remote_address().c_str());
        return crypto->decrypt(Integer(c.c_str())).get_str();
    }

    vector<string> SVC(rpc_conn conn, const vector<string>& C){
        info("call SVC from %s", conn.lock()->remote_address().c_str());
        //printf("call SVC from %s", conn.lock()->remote_address().c_str());
        vector<Vec<Integer>> M(C.size());
        vector<string> ret(M.size());
        size_t numIntegerPerCiphertext = Vector::pack_count(32, *(this->crypto));
        for(size_t i = 0; i < C.size(); ++i){
            PackedCiphertext packed(Integer(C[i].c_str()), numIntegerPerCiphertext, 32);
            M[i] = Vector::decrypt_pack(packed, *(this->crypto));

            if(config::LOG){
                string content;
                for(size_t j = 0; j < M[i].length(); ++j){
                    content += (M[i][j] - Integer(2).pow(32 - 1)).get_str() + " ";
                }
                //debug(content.c_str());
            }

            Vec<Integer> packedCmpResult(NTL::INIT_SIZE_TYPE{}, M[i].length(), 0);
            for(size_t j = 0; j < numIntegerPerCiphertext; ++j){
                M[i][j] -= Integer(2).pow(32 - 1);
                if(getMaxBitLength(M[i][j] % crypto->get_pub().n) > getMaxBitLength(crypto->get_pub().n) / 2){
                    packedCmpResult[j] = 1;
                }
            }
            

            if(config::LOG){
                string content;
                for(size_t j = 0; j < packedCmpResult.length(); ++j){
                    content += packedCmpResult[j].get_str() + " ";
                }
                //debug(content.c_str());
            }
            ret[i] = PackedCiphertext(Vector::encrypt_pack(packedCmpResult, 32, *crypto)).data.data.get_str();
        }
        return ret;
    }

    string SVM(rpc_conn conn, const vector<string>& a, const vector<string>& b, size_t plaintext_bits = 100){
        vector<Vec<Integer>> ma(a.size());
        vector<Vec<Integer>> mb(a.size());
        size_t numIntegerPerCiphertext = Vector::pack_count(plaintext_bits, *(this->crypto));
        Integer ret = 0;
        for(size_t i = 0; i < a.size(); ++i){
            PackedCiphertext pa(Integer(a[i].c_str()), numIntegerPerCiphertext, plaintext_bits);
            PackedCiphertext pb(Integer(b[i].c_str()), numIntegerPerCiphertext, plaintext_bits);
            ma[i] = Vector::decrypt_pack(pa, *(this->crypto));
            mb[i] = Vector::decrypt_pack(pb, *(this->crypto));
            for(size_t j = 0; j < ma[i].length(); ++j){
                ret += (ma[i][j] - Integer(2).pow(plaintext_bits - 1)) * (mb[i][j] - Integer(2).pow(plaintext_bits - 1));
            }
        }
        return crypto->encrypt(ret).data.get_str();
    }

    // encoding enc key into morton code
    string SE(rpc_conn conn, vector<string>&& keyStr){
        size_t dim = keyStr.size();
        vector<uint32_t> key(dim);
        for(size_t i = 0; i < dim; ++i){
            key[i] = crypto->decrypt(Ciphertext(Integer(keyStr[i].c_str()))).to_uint();
        }
        uint64_t val;
        switch(dim){
            case 2:{
                val = morton_code<2, int(64 / 2)>::encode(std::array<uint32_t, 2>{key[0], key[1]});
                break;
            }
            case 3:{
                val = morton_code<3, int(64 / 3)>::encode(std::array<uint32_t, 3>{key[0], key[1], key[2]});
                break;
            }
            case 4:{
                val = morton_code<4, int(64 / 4)>::encode(std::array<uint32_t, 4>{key[0], key[1], key[2], key[3]});
                break;
            }
            case 5:{
                val = morton_code<5, int(64 / 5)>::encode(std::array<uint32_t, 5>{key[0], key[1], key[2], key[3], key[4]});
                break;
            }
            case 6:{
                val = morton_code<6, int(64 / 6)>::encode(std::array<uint32_t, 6>{key[0], key[1], key[2], key[3], key[4], key[5]});
                break;
            }
        }
        return crypto->encrypt(val).data.get_str();
    }

public:
    PaillierFast* crypto;
    rpc_client dsp_;
    void recvKeys(){
        rpc_client cli(config::CA_IP, config::CA_PORT);
        while(!cli.connect());
        size_t keySize = cli.call<size_t>("getKeySize");
        pair<string, string> param = cli.call<pair<string, string>>("getPub");
        array<string, 4> param2 = cli.call<array<string, 4>>("getPriv");
        PublicKey pk(keySize, Integer(param.first.c_str()), Integer(param.second.c_str()));
        PrivateKey sk(keySize, std::stoul(param2[0]), Integer(param2[1].c_str()), Integer(param2[2].c_str()), Integer(param2[3].c_str()));

        crypto = new PaillierFast(pk, sk);
        info("recv keys from %s:%d", config::CA_IP.c_str(), config::CA_PORT);
    }
};

class AU {
public:
    AU(){
        recvPubKey();
    }

    void recvPubKey(){
        rpc_client ca(config::CA_IP, config::CA_PORT);
        while(!ca.connect());
        size_t keySize = ca.call<size_t>("getKeySize");
        pair<string, string> param = ca.call<pair<string, string>>("getPub");
        PublicKey pk(keySize, Integer(param.first.c_str()), Integer(param.second.c_str()));
        crypto = new PaillierFast(pk);
    }
    void query(const string& dataName, const QueryRectangle<uint32_t>& QR){
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
        //auto fut = dsp_.async_call<FUTURE>("rangeQuery", dataName, encRange, limits);
        // vector<size_t> res = fut.get().as<vector<size_t>>();

        dsp_.set_connect_timeout(1000 * 1000);
        static TimerClock tc;
        tc.update();
        pair<int, vector<size_t>> res = dsp_.call<pair<int, vector<size_t>>>("rangeQuery", dataName, encRange, limits);
        switch(res.first){
            case SUCCESS:{
                // std::cout << "query result size is " << res.second.size() << std::endl;
                info("result size is %lu, time cost is %.3f", res.second.size(), tc.getTimerMilliSec());
                break;
            }
            case NOSUCHDATA:{
                std::cout << "no such dataset " << std::endl;
                break;
            }
            case NOTCOMPLATE:{
                std::cout << "not complate " << std::endl;
                break;
            }
        }
    }
private:
    PaillierFast* crypto;
};

class CA : public rpc_server{
public:
    CA(const uint32_t key_size = config::KEY_SIZE) : rpc_server(config::CA_PORT, 1), crypto(new PaillierFast(key_size)){
        crypto->generate_keys();
        this->register_handler("getKeySize", &CA::getKeySize, this);
        this->register_handler("getPub", &CA::getPub, this);
        this->register_handler("getPriv", &CA::getPriv, this);
    }
    void saveKeys(const string& filePath){
        std::ofstream out(filePath, std::ios::out | std::ios::binary);
        if(!out.is_open()){
            info("open %s failed!", filePath.c_str());
            return;
        }
        // save key size
        out << crypto->get_pub().key_size_bits << ' ';
        // save pub key
        out << crypto->get_pub().n.get_str() << ' ' << crypto->get_pub().g.get_str() << ' ';
        // save priv key
        out << std::to_string(crypto->get_priv().a_bits) << ' ';
        out << crypto->get_priv().p.get_str() << ' ';
        out << crypto->get_priv().q.get_str() << ' ';
        out << crypto->get_priv().a.get_str();
        out.close();
        info("save done\n");
    }
    CA(const string& filePath) : rpc_server(config::CA_PORT, 1), crypto(nullptr){
        readKeys(filePath);
        this->register_handler("getKeySize", &CA::getKeySize, this);
        this->register_handler("getPub", &CA::getPub, this);
        this->register_handler("getPriv", &CA::getPriv, this);
    }
    void readKeys(const string& filePath){
        std::ifstream in(filePath, std::ios::in | std::ios::binary);
        if(!in.is_open()){
            info("open %s failed!", filePath.c_str());
            return;
        }
        size_t keySize;
        pair<string, string> param;
        array<string, 4> param2;

        in >> keySize;
        in >> param.first >> param.second;
        for(size_t i = 0; i < 4; ++i){
            in >> param2[i];
        }
        PublicKey pk(keySize, Integer(param.first.c_str()), Integer(param.second.c_str()));
        PrivateKey sk(keySize, std::stoul(param2[0]), Integer(param2[1].c_str()), Integer(param2[2].c_str()), Integer(param2[3].c_str()));

        if(crypto) delete crypto;
        crypto = new PaillierFast(pk, sk);
        info("read done\n");
    }
    size_t getKeySize(rpc_conn conn){
        return crypto->get_pub().key_size_bits;
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
