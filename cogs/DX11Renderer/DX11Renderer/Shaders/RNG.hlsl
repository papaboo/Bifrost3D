// Random number generation.
// ------------------------------------------------------------------------------------------------
// Copyright (C) 2018, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _DX11_RENDERER_SHADERS_RNG_H_
#define _DX11_RENDERER_SHADERS_RNG_H_

namespace RNG {

float van_der_corput(uint n, uint scramble) {
    n = reversebits(n) ^ scramble;
    return float((n >> 8) & 0xffffff) / float(1 << 24);
}

float sobol2(uint n, uint scramble) {
    for (uint v = 1u << 31u; n != 0; n >>= 1u, v ^= v >> 1u)
        if (n & 0x1)
            scramble ^= v;

    return float((scramble >> 8) & 0xffffff) / float(1 << 24);
}

float2 sample02(uint n, uint2 scramble = uint2(5569, 95597)) {
    return float2(van_der_corput(n, scramble.x), sobol2(n, scramble.y));
}

float lcg_sample(inout uint state) {
    const uint multiplier = 1664525u;
    const uint increment = 1013904223u;
    state = multiplier * state + increment;
    return float((state >> 8) & 0xffffff) / float(1 << 24);
}

// Optimized Spatial Hashing for Collision Detection of Deformable Objects.
// Teschner et al, 2013
uint teschner_hash(uint x, uint y) {
    return (x * 73856093) ^ (y * 19349669);
}

// Computes the power heuristic of pdf1 and pdf2.
// It is assumed that pdf1 is always valid, i.e. not NaN.
// pdf2 is allowed to be NaN, but generally try to avoid it. :)
float power_heuristic(float pdf1, float pdf2) {
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

#endif // _DX11_RENDERER_SHADERS_RNG_H_