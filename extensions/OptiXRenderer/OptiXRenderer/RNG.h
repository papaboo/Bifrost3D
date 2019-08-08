// OptiX random number generators.
// ------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _OPTIXRENDERER_RNG_H_
#define _OPTIXRENDERER_RNG_H_

#include <OptiXRenderer/Defines.h>

#include <optixu/optixu_math_namespace.h>

namespace OptiXRenderer {
namespace RNG {

// ------------------------------------------------------------------------------------------------
// Primes.
// See https://primes.utm.edu/lists/small/1000.txt for more.
// ------------------------------------------------------------------------------------------------
#if GPU_DEVICE
__constant__ unsigned short primes[128] =
#else
static const unsigned short primes[128] = 
#endif
    { 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 
    31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 
    73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 
    127, 131, 137, 139, 149, 151, 157, 163, 167, 173,
    179, 181, 191, 193, 197, 199, 211, 223, 227, 229,
    233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 
    283, 293, 307, 311, 313, 317, 331, 337, 347, 349,
    353, 359, 367, 373, 379, 383, 389, 397, 401, 409,
    419, 421, 431, 433, 439, 443, 449, 457, 461, 463,
    467, 479, 487, 491, 499, 503, 509, 521, 523, 541,
    547, 557, 563, 569, 571, 577, 587, 593, 599, 601,
    607, 613, 617, 619, 631, 641, 643, 647, 653, 659,
    661, 673, 677, 683, 691, 701, 709, 719,
};

#if GPU_DEVICE
__constant__ float uint_normalizer = 1.0f / 4294967296.0f;
#else
static const float uint_normalizer = 1.0f / 4294967296.0f;
#endif

// ------------------------------------------------------------------------------------------------
// RNG sampling utils.
// ------------------------------------------------------------------------------------------------
__inline_all__ float van_der_corput(unsigned int n, unsigned int scramble) {

    // Reverse bits of n.
#if GPU_DEVICE
    n = __brev(n);
#else
    n = (n << 16) | (n >> 16);
    n = ((n & 0x00ff00ff) << 8) | ((n & 0xff00ff00) >> 8);
    n = ((n & 0x0f0f0f0f) << 4) | ((n & 0xf0f0f0f0) >> 4);
    n = ((n & 0x33333333) << 2) | ((n & 0xcccccccc) >> 2);
    n = ((n & 0x55555555) << 1) | ((n & 0xaaaaaaaa) >> 1);
#endif
    n ^= scramble;

    return n * uint_normalizer;
}

__inline_all__ float sobol2(unsigned int n, unsigned int scramble) {

    for (unsigned int v = 1u << 31u; n != 0; n >>= 1u, v ^= v >> 1u)
        if (n & 0x1) scramble ^= v;

    return scramble * uint_normalizer;
}

__inline_all__ optix::float2 sample02(unsigned int n, optix::uint2 scramble = optix::make_uint2(5569, 95597)) {
    return optix::make_float2(van_der_corput(n, scramble.x), sobol2(n, scramble.y));
}

// Optimized Spatial Hashing for Collision Detection of Deformable Objects.
// Teschner et al, 2013
__inline_all__ unsigned int teschner_hash(unsigned int x, unsigned int y) {
    return (x * 73856093) ^ (y * 19349669);
}
__inline_all__ unsigned int teschner_hash(unsigned int x, unsigned int y, unsigned int z) {
    return (x * 73856093) ^ (y * 19349669) ^ (z * 83492791);
}

// Computes the power heuristic of pdf1 and pdf2.
// It is assumed that pdf1 is always valid, i.e. not NaN.
// pdf2 is allowed to be NaN, but generally try to avoid it. :)
__inline_all__ float power_heuristic(float pdf1, float pdf2) {
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

__inline_all__ float balance_heuristic(float pdf1, float pdf2) {
    return pdf1 / (pdf1 + pdf2);
}

// ------------------------------------------------------------------------------------------------
// Linear congruential random number generator.
// ------------------------------------------------------------------------------------------------
struct LinearCongruential {
private:
    static const unsigned int multiplier = 1664525u;
    static const unsigned int increment = 1013904223u;

    unsigned int m_state;

public:
    __inline_all__ void seed(unsigned int seed) { m_state = seed; }
    __inline_all__ unsigned int get_seed() const { return m_state; }

    __inline_all__ unsigned int sample1ui() {
        m_state = multiplier * m_state + increment;
        return m_state;
    }

    __inline_all__ float sample1f() { return float(sample1ui()) * uint_normalizer; }
    __inline_all__ optix::float2 sample2f() { return optix::make_float2(sample1f(), sample1f()); }
    __inline_all__ optix::float3 sample3f() { return optix::make_float3(sample2f(), sample1f()); }
    __inline_all__ optix::float4 sample4f() { return optix::make_float4(sample2f(), sample2f()); }
};

// ------------------------------------------------------------------------------------------------
// Tiny Van Der Corput RNG wrapper. Will wrap around pretty quickly.
// ------------------------------------------------------------------------------------------------
struct VanDerCorput {
    unsigned int m_state;

    __inline_all__ void initialize(unsigned int scramble) { m_state = scramble << 8; }

    __inline_all__ float sample1f() {
        float res = van_der_corput(m_state & 0xFF, m_state);
        ++m_state;
        return res;
    }

    __inline_all__ optix::float2 sample2f() { return optix::make_float2(sample1f(), sample1f()); }
    __inline_all__ optix::float3 sample3f() { return optix::make_float3(sample2f(), sample1f()); }
    __inline_all__ optix::float4 sample4f() { return optix::make_float4(sample2f(), sample2f()); }
};

// ------------------------------------------------------------------------------------------------
// Reverse halton random number generator.
// See Toshiya's smallppm for the reference implementation.
// ------------------------------------------------------------------------------------------------
struct ReverseHalton {

    __inline_all__ void initialize(unsigned int index, int prime_index = 0) {
        m_index = index;
        m_prime_index = prime_index;
    }

    __inline_all__ float sample1f() {
        // Reset m_prime_index if it points outside the primes array.
        m_prime_index &= 0x7F;
        return reverse_halton(m_prime_index++, m_index);
    }

    __inline_all__ optix::float2 sample2f() { return optix::make_float2(sample1f(), sample1f()); }
    __inline_all__ optix::float3 sample3f() { return optix::make_float3(sample2f(), sample1f()); }
    __inline_all__ optix::float4 sample4f() { return optix::make_float4(sample2f(), sample2f()); }

private:
    __inline_all__ int reverse(const int i, const int p) const {
        return i == 0 ? i : (p - i);
    }

    __inline_all__ float reverse_halton(const int p, int i) const {
        const unsigned short prime = primes[p];
        double h = 0.0, f = 1.0 / (double)prime, fct = f;
        while (i > 0) {
            h += reverse(i % prime, prime) * fct;
            i /= prime;
            fct *= f;
        }
        return (float)h;
    }

    int m_index;
    int m_prime_index;
};

} // NS RNG
} // NS OptiXRenderer

#endif // _OPTIXRENDERER_RNG_H_