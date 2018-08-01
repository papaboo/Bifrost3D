// Cogwheel random number generators.
// ----------------------------------------------------------------------------
// Copyright (C) 2015, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ----------------------------------------------------------------------------

#ifndef _COGWHEEL_MATH_RNG_H_
#define _COGWHEEL_MATH_RNG_H_

#include <Cogwheel/Math/MortonEncode.h>
#include <Cogwheel/Math/Vector.h>

namespace Cogwheel {
namespace Math {
namespace RNG {

// Reverse bits of n.
inline unsigned int reverse_bits(unsigned int n) {
    n = (n << 16) | (n >> 16);
    n = ((n & 0x00ff00ff) << 8) | ((n & 0xff00ff00) >> 8);
    n = ((n & 0x0f0f0f0f) << 4) | ((n & 0xf0f0f0f0) >> 4);
    n = ((n & 0x33333333) << 2) | ((n & 0xcccccccc) >> 2);
    n = ((n & 0x55555555) << 1) | ((n & 0xaaaaaaaa) >> 1);
    return n;
}

inline float van_der_corput(unsigned int n, unsigned int scramble) {
    n = reverse_bits(n ^ scramble);
    return float((n >> 8) & 0xffffff) / float(1 << 24);
}

inline float sobol2(unsigned int n, unsigned int scramble) {

    for (unsigned int v = 1u << 31u; n != 0; n >>= 1u, v ^= v >> 1u)
        if (n & 0x1) scramble ^= v;

    return float((scramble >> 8) & 0xffffff) / float(1 << 24);
}

inline Vector2f sample02(unsigned int n, Vector2ui scramble = Vector2ui(5569, 95597)) {
    return Vector2f(van_der_corput(n, scramble.x), sobol2(n, scramble.y));
}

// Optimized Spatial Hashing for Collision Detection of Deformable Objects.
// Teschner et al, 2013
inline unsigned int teschner_hash(unsigned int x, unsigned int y) {
    return (x * 73856093) ^ (y * 19349669);
}
inline unsigned int teschner_hash(unsigned int x, unsigned int y, unsigned int z) {
    return (x * 73856093) ^ (y * 19349669) ^ (z * 83492791);
}

// Robert Jenkins hash function.
// https://gist.github.com/badboy/6267743
inline unsigned int jenkins_hash(unsigned int a) {
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) ^ (a >> 19);
    a = (a + 0x165667b1) + (a << 5);
    a = (a + 0xd3a2646c) ^ (a << 9);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) ^ (a >> 16);
    return a;
}

// Hashes x and y ensuring maximal distance between consecutive xs and ys.
// NOTE: Unless filtered afterwards it visually displays a ton of correlation.
inline unsigned int even_distribution_2D(unsigned int x, unsigned int y) { return reverse_bits(morton_encode(x, y)); }

// Computes the power heuristic of pdf1 and pdf2.
// It is assumed that pdf1 is always valid, i.e. not NaN.
// pdf2 is allowed to be NaN, but generally try to avoid it. :)
inline float power_heuristic(float pdf1, float pdf2) {
    pdf1 *= pdf1;
    pdf2 *= pdf2;
    float result = pdf1 / (pdf1 + pdf2);
    // This is where floating point math gets tricky!
    // If the mis weight is NaN then it can be caused by three things.
    // 1. pdf1 is so insanely high that pdf1 * pdf1 = infinity. In that case we end up with inf / (inf + pdf2^2) and return 1, unless pdf2 was larger than pdf1, i.e. 'more infinite :p', then we return 0.
    // 2. Conversely pdf2 can also be so insanely high that pdf2 * pdf2 = infinity. This is handled analogously to above.
    // 3. pdf2 can also be NaN. In this case the power heuristic is ill-defined and we return 0.
    return !isnan(result) ? result : (pdf1 > pdf2 ? 1.0f : 0.0f);
}

} // NS RNG
} // NS Math
} // NS Cogwheel

#endif // _COGWHEEL_MATH_RNG_H_