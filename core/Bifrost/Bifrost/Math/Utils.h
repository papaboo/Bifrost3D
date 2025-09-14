// Bifrost mathematical utilities.
// ------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _BIFROST_MATH_UTILS_H_
#define _BIFROST_MATH_UTILS_H_

#include <Bifrost/Core/Defines.h>
#include <Bifrost/Math/Constants.h>
#include <Bifrost/Math/Vector.h>

#include <algorithm>
#include <cmath>

namespace Bifrost {
namespace Math {

// ------------------------------------------------------------------------------------------------
// Floating point precision helpers.
// ------------------------------------------------------------------------------------------------

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
__always_inline__ float previous_float(float v) {
    int vi;
    memcpy(&vi, &v, sizeof(int));
    if (vi < 0)
        vi = int(0x80000000) - vi;
    --vi;
    if (vi < 0)
        vi = int(0x80000000) - vi;
    memcpy(&v, &vi, sizeof(int));
    return v;
}

// Returns the next floating point number.
__always_inline__ float next_float(float v) {
    int vi;
    memcpy(&vi, &v, sizeof(int));
    if (vi < 0)
        vi = int(0x80000000) - vi;
    ++vi;
    if (vi < 0)
        vi = int(0x80000000) - vi;
    memcpy(&v, &vi, sizeof(int));
    return v;
}

// ------------------------------------------------------------------------------------------------
// Trigonometry.
// ------------------------------------------------------------------------------------------------

__always_inline__ Vector2f direction_to_latlong_texcoord(Vector3f direction) {
    float u = (atan2f(direction.z, direction.x) + PI<float>()) * 0.5f / PI<float>();
    float v = (asinf(direction.y) + PI<float>() * 0.5f) / PI<float>();
    return Vector2f(u, v);
}

__always_inline__ Vector3f latlong_texcoord_to_direction(Vector2f uv) {
    float phi = uv.x * 2.0f * PI<float>();
    float theta = uv.y * PI<float>();
    float sin_theta = sinf(theta);
    return -Vector3f(sin_theta * cosf(phi), cosf(theta), sin_theta * sinf(phi));
}

// ------------------------------------------------------------------------------------------------
// General helper methods.
// ------------------------------------------------------------------------------------------------

__always_inline__ unsigned int ceil_divide(unsigned int a, unsigned int b) {
    return (a / b) + ((a % b) > 0);
}

__always_inline__ float non_zero_sign(float v) {
    return signbit(v) ? -1.0f : 1.0f;
}

template <typename T>
__always_inline__ T min(const T a, const T b) {
    return a < b ? a : b;
}

template <typename T>
__always_inline__ T max(const T a, const T b) {
    return a > b ? a : b;
}

template <typename T>
__always_inline__ T clamp(const T value, const T lower_bound, const T upper_bound) {
    return min(max(value, lower_bound), upper_bound);
}

template <typename T>
__always_inline__ T clamp01(const T value) {
    return min(max(value, T(0)), T(1));
}

// Finds the smallest power of 2 greater or equal to x.
__always_inline__ unsigned int next_power_of_two(unsigned int x) {
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return x + 1;
}

__always_inline__ bool is_power_of_two(unsigned int v) {
    return v && !(v & (v - 1));
}

__always_inline__ float pow2(float x) { return x * x; }

// See answer from johnwbyrd on https://stackoverflow.com/questions/2589096/find-most-significant-bit-left-most-that-is-set-in-a-bit-array
__always_inline__ unsigned int most_significant_bit(unsigned int v) {
    static const int MultiplyDeBruijnBitPosition[32] = { 0, 9, 1, 10, 13, 21, 2, 29, 11, 14, 16, 18, 22, 25, 3, 30,
                                                         8, 12, 20, 28, 15, 17, 24, 7, 19, 27, 23, 6, 26, 5, 4, 31};

    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;

    return MultiplyDeBruijnBitPosition[(v * 0x07C4ACDDU) >> 27];
}

__always_inline__ float degrees_to_radians(float degress) {
    return degress * (PI<float>() / 180.0f);
}

__always_inline__ float radians_to_degress(float radians) {
    return radians * (180.0f / PI<float>());
}

template <typename T>
__always_inline__ Vector3<T> reflect(Vector3<T> incident, Vector3<T> normal) {
    return incident - normal * dot(normal, incident) * T(2);
}

__always_inline__ unsigned int reverse_bits(unsigned int n) {
    // Reverse bits of n.
    n = (n << 16) | (n >> 16);
    n = ((n & 0x00ff00ff) << 8) | ((n & 0xff00ff00) >> 8);
    n = ((n & 0x0f0f0f0f) << 4) | ((n & 0xf0f0f0f0) >> 4);
    n = ((n & 0x33333333) << 2) | ((n & 0xcccccccc) >> 2);
    n = ((n & 0x55555555) << 1) | ((n & 0xaaaaaaaa) >> 1);
    return n;
}

// Specularity of dielectrics at normal incidence, where the ray is leaving a medium with index of refraction ior_o
// and entering a medium with index of refraction, ior_i.
// Ray Tracing Gems 2, Chapter 9, The Schlick Fresnel Approximation, page 110 footnote.
__always_inline__ float dielectric_specularity(float ior_o, float ior_i) {
    return pow2((ior_o - ior_i) / (ior_o + ior_i));
}

// ------------------------------------------------------------------------------------------------
// Interpolation
// ------------------------------------------------------------------------------------------------

// Linear interpolation of arbitrary types that implement addition, subtraction and multiplication.
template <typename T, typename U>
__always_inline__ T lerp(T a, T b, U t) {
    return a + (b - a) * t;
}

template <typename T, typename U>
__always_inline__ T inverse_lerp(T a, T b, U v) {
    return (v - a) / (b - a);
}

// Smooth Hermite interpolation of arbitrary types that implement addition, subtraction and multiplication.
// Mirrors the implementation of the smoothstep functionality found in most graphics API's.
// https://en.wikipedia.org/wiki/Smoothstep
__always_inline__ float smoothstep(float a, float b, float t) {
    // Scale, and clamp x to 0..1 range
    float x = clamp01((t - a) / (b - a));
    return x * x * (3.0f - 2.0f * x);
}

// ------------------------------------------------------------------------------------------------
// Stable pairwise summation.
// ------------------------------------------------------------------------------------------------

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

// ------------------------------------------------------------------------------------------------
// Tap for a bilinearly sampled gaussian filter.
// ------------------------------------------------------------------------------------------------

struct Tap {
    float offset; // Unnormalized offset. Normalize by size to get texture coordinates.
    float weight;

    float normalized_offset(int width) { return offset / width; }
};

// Fills the list of Taps with bilinearly sampled weighted taps corresponding to a gaussian filter.
// See http://rastergrid.com/blog/2010/09/efficient-gaussian-blur-with-linear-sampling/
// A good number of samples is between 2 * std_dev and 3 * std_dev.
inline void fill_bilinear_gaussian_samples(float std_dev, Tap* samples_begin, Tap* samples_end) {
    int sample_count = int(samples_end - samples_begin);
    float double_variance = 2.0f * std_dev * std_dev;

    float total_weight = 0.0f;
    for (int s = 0; s < sample_count; ++s) {
        int t1 = s * 2;
        float w1 = exp(-(t1 * t1) / double_variance);
        if (s == 0) w1 *= 0.5f;
        int t2 = t1 + 1;
        float w2 = exp(-(t2 * t2) / double_variance);

        float weight = w1 + w2;
        float offset = (t1 * w1 + t2 * w2) / weight;
        if (isnan(offset)) offset = float(t1);

        samples_begin[s] = { offset, weight };

        total_weight += weight;
    }
    total_weight *= 2; // Double the total weight as it's only summed for the one half of the symetric bell curve.

    for (int s = 0; s < sample_count; ++s)
        samples_begin[s].weight /= total_weight;
}

} // NS Math
} // NS Bifrost

#endif // _BIFROST_MATH_UTILS_H_
