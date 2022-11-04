#pragma once
#include "ophelib/paillier_fast.h"
#include "ophelib/vector.h"
#include "ophelib/packing.h"
#include "ophelib/util.h"
#include "ophelib/ml.h"
#include "ophelib/random.h"
#include <cmath>
#include <pybind11/embed.h>
#include <pybind11/eval.h>
#include <pybind11/pybind11.h>
#include "config.hpp"
#include "utility.hpp"
#include "logging.hpp"
using namespace ophelib;
using std::vector;

inline Integer getMaxBitLength(const Integer& a){
	return a.size_bits();
}

// 查询矩形,矩形格式为[xmin,ymin,zmin,xmax,ymax,zmax]
template<class KT=uint32_t>
class QueryRectangle{
public:
	QueryRectangle(const vector<KT>& vals):dim(vals.size()/2),minvec(vector<KT>(dim)),maxvec(vector<KT>(dim)){
		for(auto i=0;i<dim;++i){
			minvec[i]=vals[i];
			maxvec[i]=vals[i+dim];
		}
	}
	const vector<KT>& get_minvec()const{
		return minvec;
	}
	const vector<KT>& get_maxvec()const{
		return maxvec;
	}
	bool isFallin(const vector<KT>& p)const{
		for(auto i=0;i<dim;++i){
			if(p[i]<minvec[i]||p[i]>maxvec[i]){
				return false;
			}
		}
		return true;
	}
	template<uint32_t dim>
	bool isFallin(const std::array<KT, dim>& p)const{
		for(auto i=0;i<dim;++i){
			if(p[i]<minvec[i]||p[i]>maxvec[i]){
				return false;
			}
		}
		return true;
	}
public:
	int dim;
private:
	vector<KT> minvec;
	vector<KT> maxvec;
};


// 查询矩形,矩形格式为[xmin,ymin,zmin,xmax,ymax,zmax]
class EQueryRectangle{
public:
	EQueryRectangle(const vector<Ciphertext>& vals) : dim(vals.size() / 2), minvec(vector<Ciphertext>(dim)), maxvec(vector<Ciphertext>(dim)){
		for(size_t i = 0; i < dim; ++i){
			minvec[i] = vals[i];
			maxvec[i] = vals[i + dim];
		}
	}

	EQueryRectangle(const vector<string>& vals) : dim(vals.size() / 2), minvec(vector<Ciphertext>(dim)), maxvec(vector<Ciphertext>(dim)){
		for(size_t i = 0; i < dim; ++i){
			minvec[i].data = Integer(vals[i].c_str());
			maxvec[i].data = Integer(vals[i + dim].c_str());
		}
	}
    
    template<class plaintext_type>
    EQueryRectangle(const PaillierFast& crypto,const QueryRectangle<plaintext_type>& QR):dim(QR.dim),minvec(vector<Ciphertext>(dim)),maxvec(vector<Ciphertext>(dim)){
        for(auto i=0;i<QR.dim;++i){
            minvec[i]=crypto.encrypt(QR.get_minvec()[i]);
            maxvec[i]=crypto.encrypt(QR.get_maxvec()[i]);
        }
    }
	const vector<Ciphertext>& get_minvec()const{
		return minvec;
	}
	const vector<Ciphertext>& get_maxvec()const{
		return maxvec;
	}
private:
	int dim;
	vector<Ciphertext> minvec;
	vector<Ciphertext> maxvec;
};
