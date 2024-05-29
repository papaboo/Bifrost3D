// Bifrost statistics on a number of samples.
// ------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _BIFROST_MATH_STATISTICS_H_
#define _BIFROST_MATH_STATISTICS_H_

#include <Bifrost/Core/Defines.h>

namespace Bifrost {
namespace Math {

//-------------------------------------------------------------------------------------------------
// Statistics of a list of values.
// Contains the minimum, maximum, mean and the variance.
// See https://en.wikipedia.org/wiki/Moment_(mathematics) for a list of other interesting values.
//-------------------------------------------------------------------------------------------------
template <typename T>
struct Statistics final {
    T minimum;
    T maximum;
    T sum;
    T sum_squared;
    size_t sample_count;

    //---------------------------------------------------------------------------------------------
    // Constructors
    //---------------------------------------------------------------------------------------------
    Statistics()
        : minimum(T(-1e30)), maximum(T(1e30)), sum(0), sum_squared(0), sample_count(0) {}

    template <typename RandomAccessItr, class UnaryPredicate>
    Statistics(RandomAccessItr first, RandomAccessItr last, UnaryPredicate predicate) {
        sample_count = last - first;

        minimum = 1e30f;
        maximum = -1e30f;

        double* means = new double[sample_count];
        double* means_sqrd = new double[sample_count];
        while (first != last) {
            T val = T(predicate(first));
            minimum = std::min(minimum, val);
            maximum = std::max(maximum, val);
            *means = val;
            *means_sqrd = val* val;
            ++first; ++means; ++means_sqrd;
        }
        means -= sample_count; means_sqrd -= sample_count;

        sum = T(sort_and_pairwise_summation(means, means + sample_count));
        sum_squared = T(sort_and_pairwise_summation(means_sqrd, means_sqrd + sample_count));

        delete[] means;
        delete[] means_sqrd;
    }

    template <typename RandomAccessItr>
    Statistics(RandomAccessItr first, RandomAccessItr last)
        : Statistics(first, last, [](RandomAccessItr v) -> T { return T(*v); }) { }

    //---------------------------------------------------------------------------------------------
    // Getters
    //---------------------------------------------------------------------------------------------
    __always_inline__ T mean() const { return sum / sample_count; }
    __always_inline__ T rms() const { return (T)sqrt(sum_squared / sample_count); }
    __always_inline__ T variance() const {
        T m2 = sum_squared / sample_count;
        T m1 = mean();
        return m2 - m1 * m1;
    }
    __always_inline__ T standard_deviation() const { return (T)sqrt(variance()); }

    //---------------------------------------------------------------------------------------------
    // Operations.
    //---------------------------------------------------------------------------------------------
    inline void add(T v) {
        minimum = std::min(minimum, v);
        maximum = std::max(maximum, v);

        sum += v;
        sum_squared += v * v;
        sample_count += 1;
    }

    inline void merge_with(Statistics other) {
        minimum = std::min(minimum, other.minimum);
        maximum = std::max(maximum, other.maximum);
        
        sum += other.sum;
        sum_squared += other.sum_squared;
        sample_count += other.sample_count;
    }

    template <typename RandomAccessItr>
    static inline Statistics merge(RandomAccessItr first, RandomAccessItr last) {
        Statistics res = *first;
        while (++first < last)
            res.merge_with(*first);
        return res;
    }
};

} // NS Math
} // NS Bifrost

#endif // _BIFROST_MATH_STATISTICS_H_
