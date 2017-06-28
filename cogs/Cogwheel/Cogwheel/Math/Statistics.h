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
    float variance;

    template <typename RandomAccessItr, class UnaryPredicate>
    Statistics(RandomAccessItr first, RandomAccessItr last, UnaryPredicate predicate) {
        size_t count = last - first;

        minimum = 1e30f;
        maximum = -1e30f;

        double* means = new double[count];
        double* means_sqrd = new double[count];
        while (first != last) {
            float val = float(predicate(first));
            minimum = std::min(minimum, val);
            maximum = std::max(maximum, val);
            *means = val;
            *means_sqrd = val* val;
            ++first; ++means; ++means_sqrd;
        }
        means -= count; means_sqrd -= count;

        mean = float(sort_and_pairwise_summation(means, means + count) / count);
        variance = float(sort_and_pairwise_summation(means_sqrd, means_sqrd + count) / count - mean * mean);

        delete[] means;
        delete[] means_sqrd;
    }

    template <typename RandomAccessItr>
    Statistics(RandomAccessItr first, RandomAccessItr last)
        : Statistics(first, last, [](RandomAccessItr v) -> float { return float(*v); }) { }

};

} // NS Math
} // NS Cogwheel

#endif // _COGWHEEL_MATH_STATISTICS_H_