#pragma once
#include <stdint.h>
#include <vector>
#include <numeric>
#include <limits>
#include <ophelib/paillier_fast.h>
#include <ophelib/vector.h>
#include "interval-tree/interval_tree.hpp"
#include <functional>
#include "config.hpp"
#include "utility.hpp"

using std::pair;
using std::string;
using std::vector;
using namespace ophelib;
using namespace lib_interval_tree;


template<class FT = double, size_t factor = 30>
std::string float2string(FT number){
    char buf[factor + 1];
    sprintf(buf, string("%." + std::to_string(factor) + "f").c_str(), number);
    std::string s(buf);
    size_t dotpos = s.find('.');
    if(dotpos != std::string::npos){
        s.erase(dotpos, 1);
    }
    return s;
}

struct linearModel{
    double sl;
    Integer b;
};

template<class KT=uint64_t,class PT=size_t,class FT=double>
struct segment{
public:
    KT start;
    KT end;
    FT slope;
    PT pos;
    PT len;
    segment(){}
    segment(KT p1,KT p2,FT p3,FT p4,PT p5):start(p1),end(p2),slope(p3),pos(p4),len(p5){}
};

template<class KT=uint64_t,class PT=size_t>
std::vector<segment<KT,PT>> shringkingCone(std::vector<KT>& keys,std::vector<PT>& pos,PT epsilon=32){
    double high_slope=std::numeric_limits<double>::max();
    double low_slope=std::numeric_limits<double>::min();
    
    std::vector<segment<KT,PT>> all_segs{{keys[0],keys[0],0,pos[0],1}};
    
    for(auto i=1;i<keys.size();++i){
        double new_high_slope=static_cast<double>(int64_t(pos[i])+int64_t(epsilon)-int64_t(all_segs.back().pos))/\
                              static_cast<double>(int64_t(keys[i])-int64_t(all_segs.back().start));
        double new_low_slope=static_cast<double>(int64_t(pos[i])-int64_t(epsilon)-int64_t(all_segs.back().pos))/\
                             static_cast<double>(int64_t(keys[i])-int64_t(all_segs.back().start));
        double tmp_slope=static_cast<double>(int64_t(pos[i])-int64_t(all_segs.back().pos))/\
                         static_cast<double>(int64_t(keys[i])-int64_t(all_segs.back().start));
        if(low_slope<=tmp_slope && high_slope>=tmp_slope){
            high_slope=std::min(new_high_slope,high_slope);
            low_slope=std::max(new_low_slope,low_slope);
            all_segs.back().len+=1;
            all_segs.back().end=keys[i];
        }else{
            all_segs.back().slope = low_slope + (high_slope-low_slope) / 2;

            //开始一个新的段
            all_segs.emplace_back(keys[i],keys[i],0,pos[i],1);
            high_slope=std::numeric_limits<double>::max();
            low_slope=std::numeric_limits<double>::min();
        }
    }
    all_segs.back().slope = low_slope + (high_slope-low_slope) / 2;
    return all_segs;
}


// 将K个段填充为包含整个区间[minv,maxv]的intervals
// 根据flag为前向还是后向，在interval中保存右/左手边最靠近的段指针
std::vector<pair<interval<uint64_t, closed>, std::shared_ptr<pair<linearModel, linearModel>>>> paddingSegments(std::vector<segment<uint64_t, size_t>>& segs) {
		std::vector<pair<interval<uint64_t, closed>, std::shared_ptr<pair<linearModel, linearModel>>>> segNodes;
        using elemType = pair<interval<uint64_t, closed>, std::shared_ptr<pair<linearModel, linearModel>>>;
		// 补充0~segs[0].start-1
        if(segs[0].start != 0){
            segNodes.emplace_back(
                interval<uint64_t, closed>{
                    0, 
                    segs[0].start - 1
                }, 
                std::make_shared<pair<linearModel, linearModel>>(
                    linearModel{segs[0].slope, segs[0].pos},
                    linearModel{0, 0}
                )
            );
        }
		for (size_t i = 0; i < segs.size(); ++i) {
            segNodes.emplace_back(
                interval<uint64_t, closed>{
                    segs[i].start, 
                    segs[i].end
                }, 
                std::make_shared<pair<linearModel, linearModel>>(
                    linearModel{segs[i].slope, Integer(segs[i].pos) - Integer(int64_t(segs[i].slope * double(segs[i].start)))},
                    linearModel{segs[i].slope, Integer(segs[i].pos) - Integer(int64_t(segs[i].slope * double(segs[i].start)))}
                )
            );
			
			if (i < segs.size() - 1) {
				if (segs[i].end + 1 <= segs[i + 1].start - 1) {
                    segNodes.emplace_back(
                        interval<uint64_t, closed>{
                            segs[i].end + 1, 
                            segs[i + 1].start - 1
                        }, 
                        std::make_shared<pair<linearModel, linearModel>>(
                            linearModel{segs[i + 1].slope, Integer(segs[i + 1].pos) - Integer(int64_t(segs[i + 1].slope * double(segs[i + 1].start))) + Integer(int64_t(segs[i + 1].slope * double((segs[i + 1].start - (segs[i].end + 1)))))},
                            linearModel{segs[i].slope, Integer(segs[i].pos) - Integer(int64_t(segs[i].slope * double(segs[i].start))) + Integer(int64_t(segs[i].slope * double(int64_t(segs[i].end) - int64_t(segs[i + 1].start - 1))))}
                        )
                    );
				}
			}
		}
		// 补充segs.back().end+1~maxv
        if( segs.back().end + 1 < std::numeric_limits<uint64_t>::max()){
            segNodes.emplace_back(
                interval<uint64_t, closed>{
                    segs.back().end + 1, 
                    std::numeric_limits<uint64_t>::max()
                }, 
                std::make_shared<pair<linearModel, linearModel>>(
                    linearModel{0, std::numeric_limits<size_t>::max()},
                    linearModel{0, segs.back().pos + segs.back().len}
                )
            );
        }
		return segNodes;
}


// 密文结点
struct encSegmentNode{
    Ciphertext low_;
    Ciphertext high_;
    Ciphertext max_;
    // encSegmentNode* left_ = nullptr;
    // encSegmentNode* right_ = nullptr;
    std::pair<Ciphertext, Ciphertext> pred1;
    std::pair<Ciphertext, Ciphertext> pred2;

    //MSGPACK_DEFINE(low_.data.get_str(), high_.data.get_str(), max_.data.get_str(), pair<string, string>{pred1.first.data.get_str(), pred1.second.data.get_str()}, pair<string, string>{pred2.first.data.get_str(), pred2.second.data.get_str()});
    // std::pair<std::function<Ciphertext(const PaillierFast& crypto,const Ciphertext& input)>,
    //           std::function<Ciphertext(const PaillierFast& crypto,const Ciphertext& input)>> preds;
    encSegmentNode(){

    }
    encSegmentNode(const PaillierFast& crypto, interval_tree_t<uint64_t>::node_type* node){
        std::pair<linearModel, linearModel>* expr = static_cast<std::pair<linearModel, linearModel>*>(node->data_ptr.get());
        
        low_ = crypto.encrypt(node->low());
        high_ = crypto.encrypt(node->high());
        max_ = crypto.encrypt(node->max());

        initPredictor(crypto, pred1, expr->first);
        initPredictor(crypto, pred2, expr->second);
    }

    void initPredictor(const PaillierFast& crypto, std::pair<Ciphertext, Ciphertext>& pred, linearModel& m){
        string base = "1" + string(config::FLOAT_EXP, '0');
        Integer sl(float2string<double, config::FLOAT_EXP>(m.sl).c_str());
        Integer b(base.c_str());
        b *= m.b;
        
        pred.first = std::move(crypto.encrypt(sl));
        pred.second = std::move(crypto.encrypt(b));

        // pred = [](const ophelib::PaillierFast& crypto,const Ciphertext& input)->Ciphertext{
        //     // Ciphertext output=SM(crypto,e_sl,input).data*e_b.data;
        //     // if(config::TRACK_CALC) printf("\n%s * %s + %s = %s\n",crypto.decrypt(e_sl).to_string(false).c_str(),
        //     // crypto.decrypt(input).to_string(false).c_str(),
        //     // crypto.decrypt(e_b).to_string(false).c_str(),
        //     // crypto.decrypt(output).to_string(false).c_str());
        //     // return output;
        // };
        if(config::DEBUG){
            printf("\n[ori sl = %.30f,b = %lu]", m.sl, m.b);
            printf("\n[exp sl = %s,b = %s]", sl.get_str().c_str(), b.get_str().c_str());
        }
    }
};

string seriEncSegmentNode(const encSegmentNode& node){
    string str = node.low_.data.get_str() + " " + 
                 node.high_.data.get_str() + " " + 
                 node.max_.data.get_str() + " " + 
                 node.pred1.first.data.get_str() + " " + 
                 node.pred1.second.data.get_str() + " " + 
                 node.pred2.first.data.get_str() + " " + 
                 node.pred2.second.data.get_str();
    return str;
}

encSegmentNode* deSeriEncSegmentNode(const string& str){
    vector<string> params;
    SplitString(str, params, " ");
    assert(params.size() == 7);
    encSegmentNode* node = new encSegmentNode;
    node->low_.data = Integer(params[0].c_str());
    node->high_.data = Integer(params[1].c_str());
    node->max_.data = Integer(params[2].c_str());
    node->pred1.first.data = Integer(params[3].c_str());
    node->pred1.second.data = Integer(params[4].c_str());
    node->pred2.first.data = Integer(params[5].c_str());
    node->pred2.second.data = Integer(params[6].c_str());
    return node;
}

void printEncSegmentNode(PaillierFast* const crypto, const std::shared_ptr<encSegmentNode>& node){
    printf("\n[low, high] = [%s, %s]\n", crypto->decrypt(node->low_).get_str().c_str(), crypto->decrypt(node->high_).get_str().c_str());
    printf("[max] = [%s]\n", crypto->decrypt(node->max_).get_str().c_str());
    pair<uint64_t, uint64_t> pred1{(crypto->decrypt(node->pred1.first) / Integer(10).pow(config::FLOAT_EXP)).to_ulong(), (crypto->decrypt(node->pred1.second) / Integer(10).pow(config::FLOAT_EXP)).to_ulong()};
    pair<uint64_t, uint64_t> pred2{(crypto->decrypt(node->pred2.first) / Integer(10).pow(config::FLOAT_EXP)).to_ulong(), (crypto->decrypt(node->pred2.second) / Integer(10).pow(config::FLOAT_EXP)).to_ulong()};
    printf("pred1 = [%lu, %lu]\n", pred1.first, pred1.second);
    printf("pred2 = [%lu, %lu]\n", pred2.first, pred2.second);
}