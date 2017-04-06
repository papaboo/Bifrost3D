// Cogwheel random number generators.
// ----------------------------------------------------------------------------
// Copyright (C) 2015, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ----------------------------------------------------------------------------

#ifndef _COGWHEEL_MATH_RNG_H_
#define _COGWHEEL_MATH_RNG_H_

#include <Cogwheel/Math/Vector.h>

namespace Cogwheel {
namespace Math {
namespace RNG {

inline float van_der_corput(unsigned int n, unsigned int scramble) {

    // Reverse bits of n.
    n = (n << 16) | (n >> 16);
    n = ((n & 0x00ff00ff) << 8) | ((n & 0xff00ff00) >> 8);
    n = ((n & 0x0f0f0f0f) << 4) | ((n & 0xf0f0f0f0) >> 4);
    n = ((n & 0x33333333) << 2) | ((n & 0xcccccccc) >> 2);
    n = ((n & 0x55555555) << 1) | ((n & 0xaaaaaaaa) >> 1);
    n ^= scramble;

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

// Robert Jenkins hash function.
// https://gist.github.com/badboy/6267743
inline unsigned int hash(unsigned int a) {
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) ^ (a >> 19);
    a = (a + 0x165667b1) + (a << 5);
    a = (a + 0xd3a2646c) ^ (a << 9);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) ^ (a >> 16);
    return a;
}

} // NS RNG
} // NS Math
} // NS Cogwheel

#endif // _COGWHEEL_MATH_RNG_H_