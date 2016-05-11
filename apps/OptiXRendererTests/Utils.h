// OptiXRenderer testing utils.
// ---------------------------------------------------------------------------
// Copyright (C) 2015-2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _OPTIXRENDERERTEST_UTILS_H_
#define _OPTIXRENDERERTEST_UTILS_H_

#include <optixu/optixu_math_namespace.h>

#include <algorithm>
#include <string>

//-----------------------------------------------------------------------------
// Comparison helpers.
//-----------------------------------------------------------------------------

inline bool almost_equal_eps(float lhs, float rhs, float eps) {
    return lhs < rhs + eps && lhs + eps > rhs;
}

static bool equal_float3(optix::float3 lhs, optix::float3 rhs) {
    return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z;
}

#define EXPECT_FLOAT3_EQ(expected, actual) EXPECT_PRED2(equal_float3, expected, actual)

static bool equal_normal(optix::float3 lhs, optix::float3 rhs, float epsilon) {
    return abs(lhs.x - rhs.x) < epsilon && abs(lhs.y - rhs.y) < epsilon && abs(lhs.z - rhs.z) < epsilon;
}

#define EXPECT_NORMAL_EQ(expected, actual, epsilon) EXPECT_PRED3(equal_normal, expected, actual, epsilon)

//-----------------------------------------------------------------------------
// To string functions.
//-----------------------------------------------------------------------------

inline std::ostream& operator<<(std::ostream& s, const optix::float3 v) {
    return s << "[x: " << v.x << ", y: " << v.y << ", z: " << v.z << "]";
}

//-----------------------------------------------------------------------------
// Stable pairwise summation.
//-----------------------------------------------------------------------------

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

#endif // _OPTIXRENDERERTEST_UTILS_H_