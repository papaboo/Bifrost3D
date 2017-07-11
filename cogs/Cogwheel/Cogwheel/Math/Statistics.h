// Cogwheel statistics on a number of samples.
// ------------------------------------------------------------------------------------------------
// Copyright (C) 2015, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _COGWHEEL_MATH_STATISTICS_H_
#define _COGWHEEL_MATH_STATISTICS_H_

#include <Cogwheel/Math/Utils.h>

namespace Cogwheel {
namespace Math {

//-------------------------------------------------------------------------------------------------
// Statistics of a list of values.
// Contains the minimum, maximum, mean and the variance.
// See https://en.wikipedia.org/wiki/Moment_(mathematics) for a list of other interesting values.
//-------------------------------------------------------------------------------------------------
struct Statistics final {
    float minimum;
    float maximum;
    float mean;
    float m2;
    size_t sample_count;

    //---------------------------------------------------------------------------------------------
    // Constructors
    //---------------------------------------------------------------------------------------------
    template <typename RandomAccessItr, class UnaryPredicate>
    Statistics(RandomAccessItr first, RandomAccessItr last, UnaryPredicate predicate) {
        sample_count = last - first;

        minimum = 1e30f;
        maximum = -1e30f;

        double* means = new double[sample_count];
        double* means_sqrd = new double[sample_count];
        while (first != last) {
            float val = float(predicate(first));
            minimum = std::min(minimum, val);
            maximum = std::max(maximum, val);
            *means = val;
            *means_sqrd = val* val;
            ++first; ++means; ++means_sqrd;
        }
        means -= sample_count; means_sqrd -= sample_count;

        mean = float(sort_and_pairwise_summation(means, means + sample_count) / sample_count);
        m2 = float(sort_and_pairwise_summation(means_sqrd, means_sqrd + sample_count) / sample_count);

        delete[] means;
        delete[] means_sqrd;
    }

    template <typename RandomAccessItr>
    Statistics(RandomAccessItr first, RandomAccessItr last)
        : Statistics(first, last, [](RandomAccessItr v) -> float { return float(*v); }) { }

    //---------------------------------------------------------------------------------------------
    // Getters
    //---------------------------------------------------------------------------------------------
    inline float variance() const { return m2 - mean * mean; }
    inline float standard_deviation() const { return sqrt(variance()); }

    //---------------------------------------------------------------------------------------------
    // Operations.
    //---------------------------------------------------------------------------------------------
    void merge_with(Statistics other) {
        minimum = std::min(minimum, other.minimum);
        maximum = std::max(maximum, other.maximum);
        
        size_t total_sample_count = sample_count + other.sample_count;
        mean = (mean * sample_count + other.mean * other.sample_count) / total_sample_count;
        m2 = (m2 * sample_count + other.m2 * other.sample_count) / total_sample_count;
        sample_count = total_sample_count;
    }
};

} // NS Math
} // NS Cogwheel

#endif // _COGWHEEL_MATH_STATISTICS_H_