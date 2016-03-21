// OptiXRenderer testing utils.
// ---------------------------------------------------------------------------
// Copyright (C) 2015-2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _OPTIXRENDERERTEST_UTILS_H_
#define _OPTIXRENDERERTEST_UTILS_H_

#include <algorithm>

namespace OptiXRenderer {

// Inplace iterative pairwise summation.
// Uses the input iterators to store the temporaries.
// http://en.wikipedia.org/wiki/Pairwise_summation
template <typename InputIterator>
typename std::iterator_traits<InputIterator>::value_type pairwise_summation(InputIterator begin, InputIterator end) {
    size_t elementCount = end - begin;

    while (elementCount > 1) {
        size_t summations = elementCount / 2;
        for (size_t s = 0; s < summations; ++s)
            begin[s] = begin[2 * s] + begin[2 * s + 1];

        // Copy last element if element count is odd.
        if ((elementCount % 2) == 1)
            begin[summations] = begin[elementCount - 1];

        elementCount = summations + (elementCount & 1);
    }

    return *begin;
}

template <typename InputIterator>
typename std::iterator_traits<InputIterator>::value_type sort_and_pairwise_summation(InputIterator begin, InputIterator end) {
    std::sort(begin, end);
    return pairwise_summation(begin, end);
}

inline bool almost_equal_eps(float lhs, float rhs, float eps) {
    return lhs < rhs + eps && lhs + eps > rhs;
}

} // NS OptiXRenderer

#endif // _OPTIXRENDERERTEST_UTILS_H_