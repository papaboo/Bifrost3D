// OptiX random number generators.
// ------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _OPTIXRENDERER_RNG_H_
#define _OPTIXRENDERER_RNG_H_

#include <OptiXRenderer/Utils.h>

#include <optixu/optixu_math_namespace.h>

namespace OptiXRenderer {
namespace RNG {

// ------------------------------------------------------------------------------------------------
// Primes.
// See https://primes.utm.edu/lists/small/1000.txt for more.
// ------------------------------------------------------------------------------------------------
__constant_all__ unsigned short primes[128] =
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

__constant_all__ unsigned int sobol_direction_numbers[4][32] = {
    0x80000000, 0x40000000, 0x20000000, 0x10000000,
    0x08000000, 0x04000000, 0x02000000, 0x01000000,
    0x00800000, 0x00400000, 0x00200000, 0x00100000,
    0x00080000, 0x00040000, 0x00020000, 0x00010000,
    0x00008000, 0x00004000, 0x00002000, 0x00001000,
    0x00000800, 0x00000400, 0x00000200, 0x00000100,
    0x00000080, 0x00000040, 0x00000020, 0x00000010,
    0x00000008, 0x00000004, 0x00000002, 0x00000001,

    0x80000000, 0xc0000000, 0xa0000000, 0xf0000000,
    0x88000000, 0xcc000000, 0xaa000000, 0xff000000,
    0x80800000, 0xc0c00000, 0xa0a00000, 0xf0f00000,
    0x88880000, 0xcccc0000, 0xaaaa0000, 0xffff0000,
    0x80008000, 0xc000c000, 0xa000a000, 0xf000f000,
    0x88008800, 0xcc00cc00, 0xaa00aa00, 0xff00ff00,
    0x80808080, 0xc0c0c0c0, 0xa0a0a0a0, 0xf0f0f0f0,
    0x88888888, 0xcccccccc, 0xaaaaaaaa, 0xffffffff,

    0x80000000, 0xc0000000, 0x60000000, 0x90000000,
    0xe8000000, 0x5c000000, 0x8e000000, 0xc5000000,
    0x68800000, 0x9cc00000, 0xee600000, 0x55900000,
    0x80680000, 0xc09c0000, 0x60ee0000, 0x90550000,
    0xe8808000, 0x5cc0c000, 0x8e606000, 0xc5909000,
    0x6868e800, 0x9c9c5c00, 0xeeee8e00, 0x5555c500,
    0x8000e880, 0xc0005cc0, 0x60008e60, 0x9000c590,
    0xe8006868, 0x5c009c9c, 0x8e00eeee, 0xc5005555,

    0x80000000, 0xc0000000, 0x20000000, 0x50000000,
    0xf8000000, 0x74000000, 0xa2000000, 0x93000000,
    0xd8800000, 0x25400000, 0x59e00000, 0xe6d00000,
    0x78080000, 0xb40c0000, 0x82020000, 0xc3050000,
    0x208f8000, 0x51474000, 0xfbea2000, 0x75d93000,
    0xa0858800, 0x914e5400, 0xdbe79e00, 0x25db6d00,
    0x58800080, 0xe54000c0, 0x79e00020, 0xb6d00050,
    0x800800f8, 0xc00c0074, 0x200200a2, 0x50050093,
};

__constant_all__ float uint_normalizer = 1.0f / 4294967296.0f;

// ------------------------------------------------------------------------------------------------
// RNG sampling utils.
// ------------------------------------------------------------------------------------------------

__inline_all__ float van_der_corput(unsigned int n, unsigned int scramble) {
    n = reverse_bits(n);
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

// Stratified Sampling for Stochastic Transparency, Laine and Karras, 2011
// The hashing extracted from Figure 4 in the original paper and as used in
// Practical Hash-based Owen Scrambling, Brent Burley, 2020
// Can be used as a fast replacement for Owen-scrambling.
__inline_all__ unsigned int laine_karras_hash(unsigned int x, unsigned int seed) {
    x += seed;
    x ^= x * 0x6c50b47cu;
    x ^= x * 0xb82f1e52u;
    x ^= x * 0xc7afe638u;
    x ^= x * 0x8d22f6e6u;
    return x;
}

// Hash developed by cessen and used in pbrt4.
// Can be used as a fast replacement for Owen-scrambling.
// https://psychopath.io/post/2021_01_30_building_a_better_lk_hash
// https://pbr-book.org/4ed/Sampling_and_Reconstruction/Sobol_Samplers
__inline_all__ unsigned int cessen_owen_hash(unsigned int x, unsigned int seed) {
    x ^= x * 0x3d20adea;
    x += seed;
    x *= (seed >> 16) | 1;
    x ^= x * 0x05526c56;
    x ^= x * 0x53a22864;
    return x;
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
    __inline_all__ void set_state(unsigned int seed) { m_state = seed; }
    __inline_all__ unsigned int get_state() const { return m_state; }

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
// Reverse halton random number generator.
// See Toshiya's smallppm for the reference implementation.
// ------------------------------------------------------------------------------------------------
struct ReverseHalton {

    __inline_all__ void initialize(unsigned int index, int prime_index = 0) {
        m_index = index;
        m_prime_index = prime_index;
    }

    __inline_all__ float sample1f() {
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
        const unsigned short prime = primes[p % 128];
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

// ------------------------------------------------------------------------------------------------
// Practical Hash-based Owen Scrambling, Brent Burley, 2020
// We use primes as our per bounce seed to decorrelate the samples.
// ------------------------------------------------------------------------------------------------
struct __align__(4) PracticalScrambledSobol {
private:
    __inline_all__ static unsigned int hash_combine(unsigned int seed, unsigned int v) {
        return seed ^ (v + (seed << 6) + (seed >> 2));
    }

    __inline_all__ static unsigned int nested_uniform_scramble_base2(unsigned int x, unsigned int seed) {
        x = reverse_bits(x);
        x = cessen_owen_hash(x, seed);
        x = reverse_bits(x);
        return x;
    }

    __inline_all__ static optix::uint4 sobol_sample4ui(unsigned int index) {
        unsigned int res[4];
        for (int dim = 0; dim < 4; dim++) {
            res[dim] = 0;
            for (int bit = 0; bit < 32; bit++) {
                int mask = (index >> bit) & 1;
                res[dim] ^= mask * sobol_direction_numbers[dim][bit];
            }
        }

        return { res[0], res[1], res[2], res[3] };
    }

    __inline_all__ static optix::uint4 shuffled_scrambled_sobol4d(unsigned int index, unsigned int seed) {
        index = nested_uniform_scramble_base2(index, seed);
        optix::uint4 xs = sobol_sample4ui(index);
        xs.x = nested_uniform_scramble_base2(xs.x, hash_combine(seed, 0));
        xs.y = nested_uniform_scramble_base2(xs.y, hash_combine(seed, 1));
        xs.z = nested_uniform_scramble_base2(xs.z, hash_combine(seed, 2));
        xs.w = nested_uniform_scramble_base2(xs.w, hash_combine(seed, 3));
        return xs;
    }

    unsigned int m_index : 24;
    unsigned int m_dimension : 8;

public:
    PracticalScrambledSobol() = default;

    __inline_all__ PracticalScrambledSobol(unsigned int x, unsigned int y, unsigned int accumulation_count) {
        // Poor mans attempt at simple blue noise.
        // morton_encode(x, y) ^ accumulation_count gives perfect blue noise in the first frame,
        // but starts to share random number chains between pixels from the second accumulation.
        // That means information loss for denoisers and RESTIR.
        // To avoid that we space out the index repetition by offseting the index by morton index << 8,
        // this offsets the indices by enough that shared information is far enough apart that RESTIR shouldn't care
        // and it seems fair to assume that after 256 samples per pixel we'd have enough information for a denoiser.
        // If more than 256 samples per pixel are needed then the user is assummed to want a ground truth reference without denoising,
        // at which point the sample correlation isn't an issue for the individual pixel.
        int morton_index = morton_encode(x, y);
        m_index = ((morton_index << 8) + morton_index) ^ accumulation_count;
        m_dimension = 0;
    }

    __inline_all__ unsigned int get_state() const { return m_dimension; }
    __inline_all__ void set_state(unsigned int state) { m_dimension = state; }

    __inline_all__ optix::uint4 sample4ui() {
        return shuffled_scrambled_sobol4d(m_index, primes[m_dimension++ % 128]);
    }

    __inline_all__ optix::float4 sample4f() { return optix::make_float4(sample4ui()) * uint_normalizer; }
};

} // NS RNG
} // NS OptiXRenderer

#endif // _OPTIXRENDERER_RNG_H_