#pragma once
#include "ophelib/paillier_fast.h"
#include "ophelib/vector.h"
#include "ophelib/omp_wrap.h"
#include "ophelib/packing.h"
#include "ophelib/util.h"
#include "ophelib/ml.h"
#include "ophelib/random.h"
#include <cmath>
using namespace ophelib;

double log2fbi(mpq_t m_op);

Integer getMaxBitLength(const Integer& a){
    mpq_t m_op;
    mpq_set_str(m_op, a.get_str().c_str(), 10);
    return uint64_t(std::floor(log2fbi(m_op)));
}

// log2 for big integer
double log2fbi(mpq_t m_op)
{
    // log(a/b) = log(a) - log(b)
    // And if a is represented in base B as:
    // a = a_N B^N + a_{N-1} B^{N-1} + ... + a_0
    // => log(a) \approx log(a_N B^N)
    // = log(a_N) + N log(B)
    // where B is the base; ie: ULONG_MAX

    static double logB = log(ULONG_MAX);

    // Undefined logs (should probably return NAN in second case?)
    if (mpz_get_ui(mpq_numref(m_op)) == 0 || mpz_sgn(mpq_numref(m_op)) < 0)
        return -INFINITY;               

    // Log of numerator
    double lognum = log(mpq_numref(m_op)->_mp_d[abs(mpq_numref(m_op)->_mp_size) - 1]);
    lognum += (abs(mpq_numref(m_op)->_mp_size)-1) * logB;

    // Subtract log of denominator, if it exists
    if (abs(mpq_denref(m_op)->_mp_size) > 0)
    {
        lognum -= log(mpq_denref(m_op)->_mp_d[abs(mpq_denref(m_op)->_mp_size)-1]);
        lognum -= (abs(mpq_denref(m_op)->_mp_size)-1) * logB;
    }
    return lognum;
}
