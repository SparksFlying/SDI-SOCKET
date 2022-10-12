#pragma once
#include<vector>
#include<map>
#include<iostream>
#include<memory>
#include<rest_rpc.hpp>
#include<libzinc/zinc.hh>
#include"config.hpp"
#include"segment.hpp"
#include"interval-tree/interval_tree.hpp"
using std::vector;
using std::map;

#define ENTITY_NAMESPACE_BEGIN namespace Entity{
#define ENTITY_NAMESPACE_END };


ENTITY_NAMESPACE_BEGIN

class DAP;
class DSP;
class AU;

template<uint32_t dim>
class record {
public:
    size_t id;
    std::array<uint32_t, dim> key;
    void* value;
};

template<uint32_t dim>
class DO {
public:
    vector<record<dim>> data_;
    size_t epsilon_ = 32;
    vector<segment<uint64_t, size_t, double>> segs;
    size_t dim_ = dim;
    
    std::map<size_t, uint64_t> mortonTable;
    // 对端通信实体
    std::shared_ptr<DAP> dap_;
    std::shared_ptr<DSP> dsp_;
    std::set<std::shared_ptr<AU>> users_;

    DO(vector<record<dim>>& data):data_(data) {
        // sort data according its morton code
        for(record<dim>& item : data) {
            mortonTable[item.id] = morton_code<dim, int(64/dim)>::encode(item.key);
        }

        std::sort(data_.begin(), data_.end(), [this](const record<dim>& a, const record<dim>& b){
            return this->mortonTable[a.id] < this->mortonTable[b.id];
        });
    }

    void buildIndex(){
        vector<uint64_t> keys(data_.size());
        vector<size_t> poses(data_.size());

        for(size_t idx = 0; idx < data_.size(); ++idx){
            keys[idx] = mortonTable[data_[idx].id];
            poses[idx] = data_[idx].id;
        }

        segs = shringkingCone<uint64_t, size_t>(keys, poses, epsilon_);
        
        lib_interval_tree::interval_tree_t<uint64_t> tree;
        tree.insert()
    }



};

ENTITY_NAMESPACE_END