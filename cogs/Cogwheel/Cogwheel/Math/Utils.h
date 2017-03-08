// Cogwheel mathematical utilities.
// ----------------------------------------------------------------------------
// Copyright (C) 2015, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ----------------------------------------------------------------------------

#ifndef _COGWHEEL_MATH_UTILS_H_
#define _COGWHEEL_MATH_UTILS_H_

#include <Cogwheel/Math/Constants.h>

#include <algorithm>
#include <cmath>
#include <string>

namespace Cogwheel {
namespace Math {

//*****************************************************************************
// Floating point precision helpers.
//*****************************************************************************

inline int compute_ulps(float a, float b) {
    static_assert(sizeof(float) == sizeof(int), "Implementation needed for when float and int have different sizes.");

    int a_as_int;
    memcpy(&a_as_int, &a, sizeof(a));
    // Make a_as_int lexicographically ordered as a twos-complement int
    if (a_as_int < 0)
        a_as_int = int(0x80000000) - a_as_int;

    int b_as_int;
    memcpy(&b_as_int, &b, sizeof(a));
    // Make b_as_int lexicographically ordered as a twos-complement int
    if (b_as_int < 0)
        b_as_int = int(0x80000000) - b_as_int;

    return abs(a_as_int - b_as_int);
}

// Floating point almost_equal function.
// http://www.cygnus-software.com/papers/comparingfloats/comparingfloats.htm
inline bool almost_equal(float a, float b, unsigned short max_ulps = 4) {
    return compute_ulps(a, b) <= max_ulps;
}

// Returns the previous floating point number.
inline float previous_float(float v) {
    int vi;
    memcpy(&vi, &v, sizeof(int));
    --vi;
    memcpy(&v, &vi, sizeof(int));
    return v;
}

// Returns the next floating point number.
inline float next_float(float v) {
    int vi;
    memcpy(&vi, &v, sizeof(int));
    ++vi;
    memcpy(&v, &vi, sizeof(int));
    return v;
}

//*****************************************************************************
// General helper methods.
//*****************************************************************************

inline unsigned int ceil_divide(unsigned int a, unsigned int b) {
    return (a / b) + ((a % b) > 0);
}

// Linear interpolation of arbitrary types that implement addition, subtraction and multiplication.
template <typename T>
inline T lerp(const T a, const T b, const float t) {
    return a + (b - a) * t;
}

template <typename T>
inline T min(const T a, const T b) {
    return a < b ? a : b;
}

template <typename T>
inline T max(const T a, const T b) {
    return a > b ? a : b;
}
template <typename T>
inline T clamp(const T value, const T lower_bound, const T upper_bound) {
    return min(max(value, lower_bound), upper_bound);
}

// Finds the smallest power of 2 greater or equal to x.
inline unsigned int next_power_of_two(unsigned int x) {
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return x + 1;
}

inline bool is_power_of_two(unsigned int v) {
    return v && !(v & (v - 1));;
}

inline float degrees_to_radians(float degress) {
    return degress * (PI<float>() / 180.0f);
}

inline float radians_to_degress(float radians) {
    return radians * (180.0f / PI<float>());
}

//-----------------------------------------------------------------------------
// Stable pairwise summation.
//-----------------------------------------------------------------------------

// Inplace iterative pairwise summation.
// Uses the input iterators to store the temporaries.
// http://en.wikipedia.org/wiki/Pairwise_summation
template <typename InputIterator>
inline typename std::iterator_traits<InputIterator>::value_type pairwise_summation(InputIterator begin, InputIterator end) {
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
inline typename std::iterator_traits<InputIterator>::value_type sort_and_pairwise_summation(InputIterator begin, InputIterator end) {
    std::sort(begin, end);
    return pairwise_summation(begin, end);
}

} // NS Math
} // NS Cogwheel

#endif // _COGWHEEL_MATH_UTILS_H_