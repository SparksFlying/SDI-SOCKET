#pragma once
#include<stdint.h>
#include<vector>
#include<numeric>
#include<limits>
#include<ophelib/paillier_fast.h>
#include<ophelib/vector.h>
#include<sstream>
using std::stringstream;
//浮点数扩大到整数，如123.123456->123123456,放大倍数为1e6
template<class FT=double>
std::pair<std::string,int> float2string(FT number){
    // 固定移动小数点6位
	std::string s = std::to_string(number * 1e6);
	size_t dotpos = s.find('.');
	return std::pair<std::string, int>(std::to_string(number * 1e6).substr(0,dotpos), 6);
    //
    // char buffer[50];
	// snprintf(buffer, sizeof(buffer),"%.20f",number);
	// std::string s(buffer);
    // //std::string s = std::to_string(number);
	// size_t dotpos = s.find('.');
	// if (dotpos < s.size()) {
	// 	s.erase(s.begin() + dotpos);
	// 	return std::make_pair<std::string, int>(std::move(s), s.size() - dotpos);
	// }
	// return std::make_pair<std::string, int>(std::move(s), 0);
}

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
// template<class KT=uint64_t,class PT=size_t,class FT=double>
// struct ssegment{
//     ssegment(const ophelib::PaillierFast& crypto,const segment<KT,PT,FT>& seg){
//         ophelib::Integer N=crypto.get_pub().n;
//         params.SetLength(6);
//         // 真结点
//         if(seg.start<=seg.end){
//             params[0]=crypto.encrypt(ophelib::Integer(seg.start)%N);
//             params[1]=crypto.encrypt(ophelib::Integer(seg.end)%N);

//             //
//             auto p1=float2string((seg.high_slope+seg.low_slope)/2);

//             params[2]=crypto.encrypt(ophelib::Integer(p1.first.c_str())%N);
//             params[3]=crypto.encrypt(ophelib::Integer(10).pow_mod_n(p1.second,N));

//             // epos=pos*10^sp
//             params[4]=crypto.encrypt(((ophelib::Integer(seg.pos)%N)*(ophelib::Integer(10).pow_mod_n(p1.second,N)))%N);
//             params[5]=crypto.encrypt(ophelib::Integer(seg.len)%N);
//         }else{
//             params[0]=crypto.encrypt(ophelib::Integer(seg.start)%N);
//             params[1]=crypto.encrypt(ophelib::Integer(seg.end)%N);

//             params[2]=crypto.encrypt(ophelib::Integer(0));
//             params[3]=crypto.encrypt(ophelib::Integer(0));
//             params[4]=crypto.encrypt((ophelib::Integer(0)));
//             params[5]=crypto.encrypt(ophelib::Integer(0));
//         }
        
//     }
//     NTL::Vec<ophelib::Ciphertext> params;
//     NTL::Vec<ophelib::Ciphertext> ekeys;
// };

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

