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
#include "/usr/lib/gcc/x86_64-linux-gnu/12/include/omp.h"

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
std::vector<segment<uint64_t, PT, double>> shringkingCone(std::vector<KT>& keys, PT epsilon=32){
    double high_slope=std::numeric_limits<double>::max();
    double low_slope=std::numeric_limits<double>::min();
    
    std::vector<segment<uint64_t, PT, double>> all_segs{{keys[0],keys[0],0,0,1}};
    
    for(auto i=1;i<keys.size();++i){
        double new_high_slope=static_cast<double>(int64_t(i)+int64_t(epsilon)-int64_t(all_segs.back().pos))/\
                              static_cast<double>(int64_t(keys[i])-int64_t(all_segs.back().start));
        double new_low_slope=static_cast<double>(int64_t(i)-int64_t(epsilon)-int64_t(all_segs.back().pos))/\
                             static_cast<double>(int64_t(keys[i])-int64_t(all_segs.back().start));
        double tmp_slope=static_cast<double>(int64_t(i)-int64_t(all_segs.back().pos))/\
                         static_cast<double>(int64_t(keys[i])-int64_t(all_segs.back().start));
        if(low_slope<=tmp_slope && high_slope>=tmp_slope){
            high_slope=std::min(new_high_slope,high_slope);
            low_slope=std::max(new_low_slope,low_slope);
            all_segs.back().len+=1;
            all_segs.back().end=keys[i];
        }else{
            all_segs.back().slope = low_slope + (high_slope-low_slope) / 2;

            //开始一个新的段
            all_segs.emplace_back(keys[i],keys[i],0,i,1);
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
        if( segs.back().end < std::numeric_limits<uint64_t>::max() && segs.back().end + 1 < std::numeric_limits<uint64_t>::max()){
            segNodes.emplace_back(
                interval<uint64_t, closed>{
                    segs.back().end + 1, 
                    std::numeric_limits<uint64_t>::max()
                }, 
                std::make_shared<pair<linearModel, linearModel>>(
                    linearModel{0, std::numeric_limits<size_t>::max()},
                    linearModel{0, size_t(segs.back().slope * double(segs.back().end - segs.back().start)) + segs.back().pos + 1}
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
        if(0){
            printf("\n[ori sl = %.30f,b = %lu]", m.sl, m.b.get_str().c_str());
            printf("\n[exp sl = %s,b = %s]", sl.get_str().c_str(), b.get_str().c_str());
        }
    }
};

string seriEncSegmentNode(const encSegmentNode& node, const string split = ","){
    string str = node.low_.data.get_str() + split + 
                 node.high_.data.get_str() + split + 
                 node.max_.data.get_str() + split + 
                 node.pred1.first.data.get_str() + split + 
                 node.pred1.second.data.get_str() + split + 
                 node.pred2.first.data.get_str() + split + 
                 node.pred2.second.data.get_str();
    return str;
}

encSegmentNode* deSeriEncSegmentNode(const string& str, const string& split = ","){
    vector<string> params;
    SplitString(str, params, split);
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


// pgm segment algorithm
#define PGM_NAMESPACE_BEGIN namespace pgm{
#define PGM_NAMESPACE_END };

PGM_NAMESPACE_BEGIN

#define PGM_SUB_EPS(x, epsilon) ((x) <= (epsilon) ? 0 : ((x) - (epsilon)))
#define PGM_ADD_EPS(x, epsilon, size) ((x) + (epsilon) + 2 >= (size) ? (size) : (x) + (epsilon) + 2)


template<typename T>
using LargeSigned = typename std::conditional_t<std::is_floating_point_v<T>,
                                                long double,
                                                std::conditional_t<(sizeof(T) < 8), int64_t, __int128>>;

template<typename X, typename Y>
class OptimalPiecewiseLinearModel {
private:
    using SX = LargeSigned<X>;
    using SY = LargeSigned<Y>;

    struct Slope {
        SX dx{};
        SY dy{};

        bool operator<(const Slope &p) const { return dy * p.dx < dx * p.dy; }
        bool operator>(const Slope &p) const { return dy * p.dx > dx * p.dy; }
        bool operator==(const Slope &p) const { return dy * p.dx == dx * p.dy; }
        bool operator!=(const Slope &p) const { return dy * p.dx != dx * p.dy; }
        explicit operator long double() const { return dy / (long double) dx; }
    };

    struct Point {
        X x{};
        Y y{};

        Slope operator-(const Point &p) const { return {SX(x) - p.x, SY(y) - p.y}; }
    };

    const Y epsilon;
    std::vector<Point> lower;
    std::vector<Point> upper;
    X first_x = 0;
    X last_x = 0;
    size_t lower_start = 0;
    size_t upper_start = 0;
    size_t points_in_hull = 0;
    Point rectangle[4];

    auto cross(const Point &O, const Point &A, const Point &B) const {
        auto OA = A - O;
        auto OB = B - O;
        return OA.dx * OB.dy - OA.dy * OB.dx;
    }

public:

    class CanonicalSegment;

    explicit OptimalPiecewiseLinearModel(Y epsilon) : epsilon(epsilon), lower(), upper() {
        if (epsilon < 0)
            throw std::invalid_argument("epsilon cannot be negative");

        upper.reserve(1u << 16);
        lower.reserve(1u << 16);
    }

    bool add_point(const X &x, const Y &y) {
        if (points_in_hull > 0 && x <= last_x)
            throw std::logic_error("Points must be increasing by x.");

        last_x = x;
        auto max_y = std::numeric_limits<Y>::max();
        auto min_y = std::numeric_limits<Y>::lowest();
        Point p1{x, y >= max_y - epsilon ? max_y : y + epsilon};
        Point p2{x, y <= min_y + epsilon ? min_y : y - epsilon};

        if (points_in_hull == 0) {
            first_x = x;
            rectangle[0] = p1;
            rectangle[1] = p2;
            upper.clear();
            lower.clear();
            upper.push_back(p1);
            lower.push_back(p2);
            upper_start = lower_start = 0;
            ++points_in_hull;
            return true;
        }

        if (points_in_hull == 1) {
            rectangle[2] = p2;
            rectangle[3] = p1;
            upper.push_back(p1);
            lower.push_back(p2);
            ++points_in_hull;
            return true;
        }

        auto slope1 = rectangle[2] - rectangle[0];
        auto slope2 = rectangle[3] - rectangle[1];
        bool outside_line1 = p1 - rectangle[2] < slope1;
        bool outside_line2 = p2 - rectangle[3] > slope2;

        if (outside_line1 || outside_line2) {
            points_in_hull = 0;
            return false;
        }

        if (p1 - rectangle[1] < slope2) {
            // Find extreme slope
            auto min = lower[lower_start] - p1;
            auto min_i = lower_start;
            for (auto i = lower_start + 1; i < lower.size(); i++) {
                auto val = lower[i] - p1;
                if (val > min)
                    break;
                min = val;
                min_i = i;
            }

            rectangle[1] = lower[min_i];
            rectangle[3] = p1;
            lower_start = min_i;

            // Hull update
            auto end = upper.size();
            for (; end >= upper_start + 2 && cross(upper[end - 2], upper[end - 1], p1) <= 0; --end)
                continue;
            upper.resize(end);
            upper.push_back(p1);
        }

        if (p2 - rectangle[0] > slope1) {
            // Find extreme slope
            auto max = upper[upper_start] - p2;
            auto max_i = upper_start;
            for (auto i = upper_start + 1; i < upper.size(); i++) {
                auto val = upper[i] - p2;
                if (val < max)
                    break;
                max = val;
                max_i = i;
            }

            rectangle[0] = upper[max_i];
            rectangle[2] = p2;
            upper_start = max_i;

            // Hull update
            auto end = lower.size();
            for (; end >= lower_start + 2 && cross(lower[end - 2], lower[end - 1], p2) >= 0; --end)
                continue;
            lower.resize(end);
            lower.push_back(p2);
        }

        ++points_in_hull;
        return true;
    }

    CanonicalSegment get_segment() {
        if (points_in_hull == 1)
            return CanonicalSegment(rectangle[0], rectangle[1], first_x);
        return CanonicalSegment(rectangle, first_x);
    }

    void reset() {
        points_in_hull = 0;
        lower.clear();
        upper.clear();
    }
};

template<typename X, typename Y>
class OptimalPiecewiseLinearModel<X, Y>::CanonicalSegment {
    friend class OptimalPiecewiseLinearModel;

    Point rectangle[4];
    X first;

    CanonicalSegment(const Point &p0, const Point &p1, X first) : rectangle{p0, p1, p0, p1}, first(first) {};

    CanonicalSegment(const Point (&rectangle)[4], X first)
        : rectangle{rectangle[0], rectangle[1], rectangle[2], rectangle[3]}, first(first) {};

    bool one_point() const {
        return rectangle[0].x == rectangle[2].x && rectangle[0].y == rectangle[2].y
            && rectangle[1].x == rectangle[3].x && rectangle[1].y == rectangle[3].y;
    }

public:

    CanonicalSegment() = default;

    X get_first_x() const { return first; }

    std::pair<long double, long double> get_intersection() const {
        auto &p0 = rectangle[0];
        auto &p1 = rectangle[1];
        auto &p2 = rectangle[2];
        auto &p3 = rectangle[3];
        auto slope1 = p2 - p0;
        auto slope2 = p3 - p1;

        if (one_point() || slope1 == slope2)
            return {p0.x, p0.y};

        auto p0p1 = p1 - p0;
        auto a = slope1.dx * slope2.dy - slope1.dy * slope2.dx;
        auto b = (p0p1.dx * slope2.dy - p0p1.dy * slope2.dx) / static_cast<long double>(a);
        auto i_x = p0.x + b * slope1.dx;
        auto i_y = p0.y + b * slope1.dy;
        return {i_x, i_y};
    }

    std::pair<long double, SY> get_floating_point_segment(const X &origin) const {
        if (one_point())
            return {0, (rectangle[0].y + rectangle[1].y) / 2};

        if constexpr (std::is_integral_v<X> && std::is_integral_v<Y>) {
            auto slope = rectangle[3] - rectangle[1];
            auto intercept_n = slope.dy * (SX(origin) - rectangle[1].x);
            auto intercept_d = slope.dx;
            auto rounding_term = ((intercept_n < 0) ^ (intercept_d < 0) ? -1 : +1) * intercept_d / 2;
            auto intercept = (intercept_n + rounding_term) / intercept_d + rectangle[1].y;
            return {static_cast<long double>(slope), intercept};
        }

        auto[i_x, i_y] = get_intersection();
        auto[min_slope, max_slope] = get_slope_range();
        auto slope = (min_slope + max_slope) / 2.;
        auto intercept = i_y - (i_x - origin) * slope;
        return {slope, intercept};
    }

    std::pair<long double, long double> get_slope_range() const {
        if (one_point())
            return {0, 1};

        auto min_slope = static_cast<long double>(rectangle[2] - rectangle[0]);
        auto max_slope = static_cast<long double>(rectangle[3] - rectangle[1]);
        return {min_slope, max_slope};
    }
};

template<class KT = int64_t, class Floating = double>
struct Segment {
    int64_t key;
    double slope;
    int32_t intercept;

    Segment() = default;

    Segment(KT key, double slope, size_t intercept) : key(key), slope(slope), intercept(intercept){}

    explicit Segment(size_t n) : key(std::numeric_limits<KT>::max()), slope(), intercept(n) {};

    explicit Segment(const typename OptimalPiecewiseLinearModel<KT, size_t>::CanonicalSegment &cs)
        : key(cs.get_first_x()) {
        auto[cs_slope, cs_intercept] = cs.get_floating_point_segment(key);
        if (cs_intercept > std::numeric_limits<decltype(intercept)>::max())
            throw std::overflow_error("Change the type of Segment::intercept to int64");
        slope = cs_slope;
        intercept = cs_intercept;
    }

    friend inline bool operator<(const Segment &s, const KT &k) { return s.key < k; }
    friend inline bool operator<(const KT &k, const Segment &s) { return k < s.key; }
    friend inline bool operator<(const Segment &s, const Segment &t) { return s.key < t.key; }

    operator KT() { return key; };

    /**
     * Returns the approximate position of the specified key.
     * @param k the key whose position must be approximated
     * @return the approximate position of the specified key
     */
    inline size_t operator()(const KT &k) const {
        auto pos = int64_t(slope * (k - key)) + intercept;
        return pos > 0 ? size_t(pos) : 0ull;
    }
};

template<typename Fin, typename Fout>
size_t make_segmentation(size_t n, size_t epsilon, Fin in, Fout out) {
    if (n == 0)
        return 0;

    using X = typename std::invoke_result_t<Fin, size_t>::first_type;
    using Y = typename std::invoke_result_t<Fin, size_t>::second_type;
    size_t c = 0;
    auto p = in(0);

    OptimalPiecewiseLinearModel<X, Y> opt(epsilon);
    opt.add_point(p.first, p.second);

    for (size_t i = 1; i < n; ++i) {
        auto next_p = in(i);
        if (next_p.first == p.first)
            continue;
        p = next_p;
        if (!opt.add_point(p.first, p.second)) {
            out(opt.get_segment());
            opt.add_point(p.first, p.second);
            ++c;
        }
    }

    out(opt.get_segment());
    return ++c;
}

template<typename RandomIt>
auto make_segmentation(RandomIt first, RandomIt last, size_t epsilon) {
    using key_type = typename RandomIt::value_type;
    using canonical_segment = typename OptimalPiecewiseLinearModel<key_type, size_t>::CanonicalSegment;
    using pair_type = typename std::pair<key_type, size_t>;

    size_t n = std::distance(first, last);
    std::vector<canonical_segment> out;
    out.reserve(epsilon > 0 ? n / (epsilon * epsilon) : n / 16);

    auto in_fun = [first](auto i) { return pair_type(first[i], i); };
    auto out_fun = [&out](const auto &cs) { out.push_back(cs); };
    make_segmentation(n, epsilon, in_fun, out_fun);

    return out;
}


template<typename Fin, typename Fout>
size_t make_segmentation_par(size_t n, size_t epsilon, Fin in, Fout out) {
    auto parallelism = std::min(std::min(omp_get_num_procs(), omp_get_max_threads()), 20);
    auto chunk_size = n / parallelism;
    auto c = 0ull;

    if (parallelism == 1 || n < 1ull << 15)
        return make_segmentation(n, epsilon, in, out);

    using X = typename std::invoke_result_t<Fin, size_t>::first_type;
    using Y = typename std::invoke_result_t<Fin, size_t>::second_type;
    using canonical_segment = typename OptimalPiecewiseLinearModel<X, Y>::CanonicalSegment;
    std::vector<std::vector<canonical_segment>> results(parallelism);

    #pragma omp parallel for reduction(+:c) num_threads(parallelism)
    for (auto i = 0; i < parallelism; ++i) {
        auto first = i * chunk_size;
        auto last = i == parallelism - 1 ? n : first + chunk_size;
        if (first > 0) {
            for (; first < last; ++first)
                if (in(first).first != in(first - 1).first)
                    break;
            if (first == last)
                continue;
        }

        auto in_fun = [in, first](auto j) { return in(first + j); };
        auto out_fun = [&results, i](const auto &cs) { results[i].emplace_back(cs); };
        results[i].reserve(chunk_size / (epsilon > 0 ? epsilon * epsilon : 16));
        c += make_segmentation(last - first, epsilon, in_fun, out_fun);
    }

    for (auto &v : results)
        for (auto &cs : v)
            out(cs);

    return c;
}

template<class KT = int64_t, class PT = size_t>
vector<Segment<int64_t, double>> fit(vector<KT>& keys, PT epsilon = 32)
{
    auto first = keys.begin();
    auto last = keys.end();
    size_t n = std::distance(first, last);
    auto ignore_last = *std::prev(last) == std::numeric_limits<KT>::max(); // max() is the sentinel value
    auto last_n = n - ignore_last;
    last -= ignore_last;
    vector<Segment<int64_t, double>> segments;

    auto in_fun = [&](auto i) {
        auto x = first[i];
        // Here there is an adjustment for inputs with duplicate keys: at the end of a run of duplicate keys equal
        // to x=first[i] such that x+1!=first[i+1], we map the values x+1,...,first[i+1]-1 to their correct rank i
        auto flag = i > 0 && i + 1u < n && x == first[i - 1] && x != first[i + 1] && x + 1 != first[i + 1];
        return std::pair<KT, PT>(x + flag, i);
    };
    auto out_fun = [&](auto cs) { segments.emplace_back(cs); };

    auto build_level = [&](auto epsilon, auto in_fun, auto out_fun) {
        auto n_segments = make_segmentation_par(last_n, epsilon, in_fun, out_fun);
        if (last_n > 1 && segments.back().slope == 0) {
            // Here we need to ensure that keys > *(last-1) are approximated to a position == prev_level_size
            segments.emplace_back(*std::prev(last) + 1, 0, last_n);
            ++n_segments;
        }
        segments.emplace_back(last_n); // Add the sentinel segment
        return n_segments;
    };
    build_level(epsilon, in_fun, out_fun);
    return segments;
}

vector<segment<uint64_t, size_t, double>> transform(const vector<int64_t>& keys, vector<Segment<int64_t, double>>&& pgmSegments)
{
    // {
    //     vector<pair<size_t, size_t>> preds(keys.size());
    //     for(size_t i = 0; i < preds.size(); ++i){
    //         auto it = std::prev(std::upper_bound(pgmSegments.begin(), pgmSegments.end(), keys[i]));
    //         auto pos = std::min<size_t>((*it)(keys[i]), std::next(it)->intercept);
    //         preds[i].first = PGM_SUB_EPS(pos, 32);
    //         preds[i].second = PGM_ADD_EPS(pos, 32, keys.size());
    //     }
    //     std::cout << "end";
    // }
    vector<segment<uint64_t, size_t, double>> segments(pgmSegments.size());
    for(size_t i = 0; i < segments.size(); ++i)
    {
        segments[i].start = pgmSegments[i].key;
        segments[i].end = (i != segments.size() - 1) ? pgmSegments[i + 1].key - 1 : std::numeric_limits<uint64_t>::max();
        segments[i].slope = pgmSegments[i].slope;
        segments[i].pos = pgmSegments[i].intercept;
    }
    return segments;
}

PGM_NAMESPACE_END
