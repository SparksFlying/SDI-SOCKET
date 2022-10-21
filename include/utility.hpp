#pragma once
#include<iostream>
#include<chrono>
#include<vector>
#include<string>
#include<fstream>
#include<sstream>
#include<set>
#include<random>
#include<array>
#include<thread>
#include<functional>
#include<bitset>
using namespace std::chrono;
using std::vector;
using std::string;
using std::set;
using std::thread;
using std::default_random_engine;
using std::uniform_int_distribution;
using std::bitset;


// string.split()
// param s:目标字符串
// param v:储存分割结果
// param c:分隔符
void SplitString(const std::string& s, std::vector<std::string>& v, const std::string& c)
{
  std::string::size_type pos1, pos2;
  pos2 = s.find(c);
  pos1 = 0;
  while(std::string::npos != pos2)
  {
    v.push_back(s.substr(pos1, pos2-pos1));
 
    pos1 = pos2 + c.size();
    pos2 = s.find(c, pos1);
  }
  if(pos1 != s.length())
    v.push_back(s.substr(pos1));
}

/*===================================================================================================*/
/*===============================================计时器===============================================*/
/*===================================================================================================*/
// 提供s、ms以及us级的计时功能
class TimerClock
{
public:
    TimerClock()
    {
        update();
    }
    
    ~TimerClock()
    {
    }
    
    void update()
    {
    _start = high_resolution_clock::now();
    }
    //获取秒
    double getTimerSecond()
    {
    return getTimerMicroSec() * 0.000001;
    }
    //获取毫秒
    double getTimerMilliSec()
    {
    return getTimerMicroSec()*0.001;
    }
    //获取微妙
    long long getTimerMicroSec()
    {
    //当前时钟减去开始时钟的count
    return duration_cast<microseconds>(high_resolution_clock::now() - _start).count();
    }
private:
    time_point<high_resolution_clock>_start;
};

/*===================================================================================================*/
/*===============================================随机数生成器==========================================*/
/*===================================================================================================*/
// 随机数服从[vmin,vmax]之间的均匀分布
template<class KT=uint32_t>
class generator {
	using ll=size_t;
private:
	default_random_engine e;//随机数引擎
	KT vmin, vmax;//随机数生成范围
	ll grid;//最小单位
public:
	generator(KT vmin, KT vmax):vmin(vmin),vmax(vmax),grid(100000),e(time(0)){}
	void generate_random_vec(vector<KT>& num_vec,size_t start,size_t end) {
		uniform_int_distribution<KT> u(vmin, vmax);
		for (auto i = start; i != end; ++i) num_vec[i] = u(e);
	}
	vector<KT> generate_with_parallel(ll len) {
		ll tnum = 2;//len / grid;//启用线程数
		tnum = tnum == 0 ? 1 : tnum;
		if (tnum > 10) {
			tnum = 10;
			grid = len / tnum;
		}
		grid = len / tnum;
		vector<KT> num_vec(len,0);
		vector<thread> t_vec;
		size_t start=0,end;
		for (auto i = 0; i < tnum;++i) {
			end = start+(len / grid >= 1 ? grid : len)-1;
			//填充
			t_vec.push_back(
				thread(&generator::generate_random_vec,
						this,
						std::ref(num_vec),
						start,
						end)
			);
			len -= grid;
			start=end+1;
		}
		//等待所有线程执行完毕
		std::for_each(t_vec.begin(), t_vec.end(), std::mem_fn(&std::thread::join));
		return num_vec;
	}
	vector<KT> generate_serial(ll len) {
		uniform_int_distribution<KT> u(vmin, vmax);
		vector<KT> num_vec(len, 0);
		for (auto& num : num_vec) num = u(e);
		return num_vec;
	}
};

/*===================================================================================================*/
/*===============================================数据集读取与结果写入==========================================*/
/*===================================================================================================*/
void unique(vector<vector<uint32_t>>&);

//从NYC数据集中抽取PULocationID以及DOLocationID作为二维数据集返回
vector<vector<uint32_t>> read_from_NYC_csv(const string& path,uint64_t size=10) {
	std::ifstream inFile(path, std::ios::in);
	string lineStr;
	vector<vector<uint32_t>> data(size,vector<uint32_t>(2));
	uint64_t i = 0;
	size_t PULocationID_pos = 7, DOLocationID_pos = 8;
	
	//空过行首
	getline(inFile, lineStr);
	while (i<size&&getline(inFile, lineStr))
	{
		// 打印整行字符串
		//cout << lineStr << endl;
		// 存成二维表结构
		//cout<<inFile.is_open()<<"\n";
		std::stringstream ss(lineStr);
		string str;
		//空过前7列
		for (size_t j = 0; j < PULocationID_pos; ++j) getline(ss, str, ',');
		getline(ss, str, ',');
		data[i][0] = std::stoul(str);
		getline(ss, str, ',');
		data[i][1] = std::stoul(str);
		++i;
	}
	inFile.close();
	return data;
}

//从Imis数据集中抽取longitude以及latitude作为二维数据集返回
vector<vector<uint32_t>> read_from_IMIS_txt(const string& path="/mnt/hgfs/share/Imis-3months.txt",uint64_t size=10) {
    std::ifstream inFile(path, std::ios::in);
	string lineStr;
	vector<vector<uint32_t>> data(size,vector<uint32_t>(2));
	uint64_t i = 0;
	size_t longitude_pos= 2;
	
	while (i<size&&getline(inFile, lineStr))
	{
		// 打印整行字符串
		//cout << lineStr << endl;
		// 存成二维表结构
		//cout<<inFile.is_open()<<"\n";
		std::stringstream ss(lineStr);
		string str;
		//空过前7列
		for (auto j = 0; j < longitude_pos; ++j) getline(ss, str, ' ');
		getline(ss, str, ' ');
		data[i][0] = static_cast<uint32_t>(std::stod(str)*1e6);
		getline(ss, str, ' ');
		data[i][1] = static_cast<uint32_t>(std::stod(str)*1e6);
		++i;
	}
	inFile.close();
    unique(data);
	return data;
}

vector<vector<uint32_t>> read_from_roadnet_txt(uint64_t size=2000,int dim=2,const string& path="data/roadNet-CA.txt") {
    std::ifstream inFile(path, std::ios::in);
	string lineStr;
	vector<vector<uint32_t>> data(size,vector<uint32_t>(dim));
	uint64_t i = 0;

	while (i<size&&getline(inFile, lineStr))
	{
		// 打印整行字符串
		//cout << lineStr << endl;
		// 存成二维表结构
		//cout<<inFile.is_open()<<"\n";
		vector<string> strs;
		SplitString(lineStr,strs,"\t");
		bool flag=false;
		for(size_t j=0;j<dim;++j){
			data[i][j] = static_cast<uint32_t>(std::stoul(strs[j]));
			if(data[i][j]>1000){
				flag=true;break;
			}
		}
		if(flag) continue;
		++i;
	}
	inFile.close();
    //unique(data);
	return data;
}

vector<vector<uint32_t>> read_from_foursquare_txt(uint64_t size,const vector<size_t>& chose_which_dims,const string& path="data/dataset_TSMC2014_NYC.txt") {
    std::ifstream inFile(path, std::ios::in);
	string lineStr;
	vector<vector<uint32_t>> data(size,vector<uint32_t>(chose_which_dims.size()));
	uint64_t i = 0;
	
	while (i<size&&getline(inFile, lineStr))
	{
		// 打印整行字符串
		//cout << lineStr << endl;
		// 存成二维表结构
		//cout<<inFile.is_open()<<"\n";
		vector<string> strs;
		SplitString(lineStr,strs,",");
		for(size_t j=0;j<chose_which_dims.size();++j){
			if(chose_which_dims[j]==4||chose_which_dims[j]==5){
				// 将经纬度范围从[-180,180]调整至[0,360]
				float tmp=std::strtof(strs[chose_which_dims[j]].c_str(),nullptr)+180.0;
				data[i][j]=static_cast<uint32_t>(tmp*1e6);
				continue;
			}
			data[i][j]=static_cast<uint32_t>(std::stoul(strs[chose_which_dims[j]]));
		}
		++i;
	}
	inFile.close();
    unique(data);
	return data;
}

vector<vector<uint32_t>> read_from_US_txt(const vector<size_t>& chose_which_dims,uint64_t size=2000,const string& path="data/USCensus1990.txt") {
    std::ifstream inFile(path, std::ios::in);
	string lineStr;
	vector<vector<uint32_t>> data(size,vector<uint32_t>(chose_which_dims.size()));
	uint64_t i = 0;
	getline(inFile,lineStr);
	while (i<size&&getline(inFile, lineStr))
	{
		// 打印整行字符串
		//cout << lineStr << endl;
		// 存成二维表结构
		//cout<<inFile.is_open()<<"\n";
		vector<string> strs;
		SplitString(lineStr,strs,",");
		for(size_t j=0;j<chose_which_dims.size();++j){
			//data[i][j]=static_cast<uint32_t>(std::stoul(strs[chose_which_dims[j]].substr(4))); // for hotel_8.txt
			data[i][j]=static_cast<uint32_t>(std::stoul(strs[chose_which_dims[j]]))*100+rand()%10;
		}
		++i;
	}
	inFile.close();
    unique(data);
	return data;
}

vector<vector<uint32_t>> read_vals_from_txt(const string& path,const char& delimeter=',') {
	std::ifstream inFile(path);
	string line;
	vector<vector<uint32_t>> res;
	while (getline(inFile, line)) {
		vector<uint32_t> tmp;
		std::stringstream ss(line);
		string str;
		while (getline(ss, str, delimeter)) tmp.push_back(std::stod(str));
		res.push_back(tmp);
	}
	return res;
}

vector<vector<uint32_t>> read_from_syn_txt(uint64_t size=2000,int dim=2,const string& path="data/SYN_100000_6.txt") {
    std::ifstream inFile(path, std::ios::in);
	string lineStr;
	vector<vector<uint32_t>> data(size,vector<uint32_t>(dim));
	uint64_t i = 0;
	
	while (i<size&&getline(inFile, lineStr))
	{
		// 打印整行字符串
		//cout << lineStr << endl;
		// 存成二维表结构
		//cout<<inFile.is_open()<<"\n";
		std::stringstream ss(lineStr);
		string str;
		for(size_t j=0;j<dim;++j){
			getline(ss, str, ',');
			data[i][j] = static_cast<uint32_t>(std::stoul(str));
		}
		++i;
	}
	inFile.close();
    unique(data);
	return data;
}
// template<class T1=double,class T2,class T3>
// void write_csv(vector<vector<T1>> arr,const vector<T2>& columns,const vector<T3>& index,const string& path){
//     //py::scoped_interpreter python;
//     py::module_ pd = py::module::import("pandas");
//     py::array_t<T1> nparr=py::cast(arr);//.cast<py::array_t<double>>();
//     py::object dataframe=pd.attr("DataFrame")(nparr,"columns"_a=columns,"index"_a=index);
//     dataframe.attr("to_csv")(path);
// }


//去重
void unique(vector<vector<uint32_t>>& data){
    set<vector<uint32_t>> s(data.begin(),data.end());
    data.assign(s.begin(),s.end());
}

void write_times_to_file(const string& path,const vector<vector<long long>>& times,decltype(std::ios::app) flag=std::ios::out){
	std::ofstream outFile(path,flag);
	for(auto v1:times){
		for(auto v2:v1){
			outFile<<v2<<"\t";
		}
		outFile<<"\n";
	}
	outFile.close();
}

template<class Type,int N>
std::array<Type,N> vec2arr(const std::vector<Type>& vec){
	std::array<Type,N> arr;
	std::copy(vec.begin(),vec.end(),arr.begin());
	return arr;
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
public:
	int dim;
private:
	vector<KT> minvec;
	vector<KT> maxvec;
};




template<class KT=uint32_t,int dim=2,int bitperdim=32>
vector<vector<KT>> gen_dataset(const size_t size=10000) {
	KT minv = (std::numeric_limits<KT>::min)();
	KT maxv = (std::numeric_limits<KT>::max)();
	
	bitset<sizeof(KT)*8> minvbits(minv);
	bitset<sizeof(KT)*8> maxvbits(maxv);
	for (size_t i = sizeof(KT) * 8 - 2; i >= bitperdim; --i) {
		minvbits[i] = 0; maxvbits[i] = 0;
	}
	minv = static_cast<KT>(minvbits.to_ullong());
	maxv = static_cast<KT>(maxvbits.to_ullong());

	generator<KT> g(minv,maxv);
	vector<vector<KT>> res(size, vector<KT>(dim));
	for (size_t i = 0; i < size; ++i) g.generate_random_vec(res[i], 0, dim);
	return res;
}

template<class PT>
PT get_opt_epsilon(PT size){
	PT tmp=static_cast<PT>(ceil(sqrt(size)));
	PT r=static_cast<PT>(ceil(log2(tmp)));
	return pow(2,r);
}

template<class KT,int dim,int bitperdim>
KT calc_number_after_shift(KT r,int dim_seq){
	if(dim==2){
		return pow(2,dim_seq)*(r&1 + 4*r>>1);
	}else if(dim==3){
		// error
		return 0;
	}
}

