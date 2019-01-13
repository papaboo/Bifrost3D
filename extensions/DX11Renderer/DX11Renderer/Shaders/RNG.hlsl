// Random number generation.
// ------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _DX11_RENDERER_SHADERS_RNG_H_
#define _DX11_RENDERER_SHADERS_RNG_H_

// Insert a 0 bit in between each of the 16 low bits of v.
uint part_by_1(uint v) {
    v &= 0x0000ffff;                 // v = ---- ---- ---- ---- fedc ba98 7654 3210
    v = (v ^ (v << 8)) & 0x00ff00ff; // v = ---- ---- fedc ba98 ---- ---- 7654 3210
    v = (v ^ (v << 4)) & 0x0f0f0f0f; // v = ---- fedc ---- ba98 ---- 7654 ---- 3210
    v = (v ^ (v << 2)) & 0x33333333; // v = --fe --dc --ba --98 --76 --54 --32 --10
    v = (v ^ (v << 1)) & 0x55555555; // v = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
    return v;
}

uint morton_encode(uint x, uint y) {
    return part_by_1(y) | (part_by_1(x) << 1);
}

namespace RNG {

float van_der_corput(uint n, uint scramble) {
    n = reversebits(n) ^ scramble;
    return n / 4294967296.0f;
}

float sobol2(uint n, uint scramble) {
#pragma warning (disable: 3557) // False positive warning that loop in sobol2 doesn't seem to do anything.

    for (uint v = 1u << 31u; n != 0; n >>= 1u, v ^= v >> 1u)
        if (n & 0x1)
            scramble ^= v;

    return scramble / 4294967296.0f;
}

float2 sample02(uint n, uint2 scramble = uint2(0, 0)) {
    return float2(van_der_corput(n, scramble.x), sobol2(n, scramble.y));
}

float lcg_sample(inout uint state) {
    const uint multiplier = 1664525u;
    const uint increment = 1013904223u;
    state = multiplier * state + increment;
    return state / 4294967296.0f;
}

// Optimized Spatial Hashing for Collision Detection of Deformable Objects.
// Teschner et al, 2013
uint teschner_hash(uint x, uint y) {
    return (x * 73856093) ^ (y * 19349669);
}

// Hashes x and y ensuring maximal distance between consecutive xs and ys.
// NOTE: Unless filtered afterwards it visually displays a ton of correlation.
uint evenly_distributed_2D_seed(uint x, uint y) { return reversebits(morton_encode(x, y)); }
uint evenly_distributed_2D_seed(uint2 i) { return evenly_distributed_2D_seed(i.x, i.y); }

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