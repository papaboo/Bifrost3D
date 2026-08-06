// Bifrost random number generators and utilities.
// ----------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ----------------------------------------------------------------------------

#ifndef _BIFROST_MATH_RNG_H_
#define _BIFROST_MATH_RNG_H_

#include <Bifrost/Core/Defines.h>
#include <Bifrost/Math/MortonEncode.h>
#include <Bifrost/Math/Vector.h>

namespace Bifrost::Math::RNG {

// ------------------------------------------------------------------------------------------------
// Primes.
// See https://primes.utm.edu/lists/small/1000.txt for more.
// ------------------------------------------------------------------------------------------------
_constant_all_archs_ unsigned short primes[128] =
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

_constant_all_archs_ unsigned int sobol_direction_numbers[4][32] = {
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

constexpr float uint_normalizer = 1.0f / 4294967296.0f;

// Reverse bits of n.
_inline_all_archs_ unsigned int reverse_bits(unsigned int n) {
    n = (n << 16) | (n >> 16);
    n = ((n & 0x00ff00ff) << 8) | ((n & 0xff00ff00) >> 8);
    n = ((n & 0x0f0f0f0f) << 4) | ((n & 0xf0f0f0f0) >> 4);
    n = ((n & 0x33333333) << 2) | ((n & 0xcccccccc) >> 2);
    n = ((n & 0x55555555) << 1) | ((n & 0xaaaaaaaa) >> 1);
    return n;
}

_inline_all_archs_ float van_der_corput(unsigned int n, unsigned int scramble) {
    n = reverse_bits(n) ^ scramble;
    return n * uint_normalizer;
}

_inline_all_archs_ float sobol2(unsigned int n, unsigned int scramble) {

    for (unsigned int v = 1u << 31u; n != 0; n >>= 1u, v ^= v >> 1u)
        if (n & 0x1) scramble ^= v;

    return scramble * uint_normalizer;
}

_inline_all_archs_ Vector2f sample02(unsigned int n, Vector2ui scramble = Vector2ui(5569, 95597)) {
    return Vector2f(van_der_corput(n, scramble.x), sobol2(n, scramble.y));
}

// Optimized Spatial Hashing for Collision Detection of Deformable Objects.
// Teschner et al, 2013
_inline_all_archs_ unsigned int teschner_hash(unsigned int x, unsigned int y) {
    return (x * 73856093) ^ (y * 19349669);
}
_inline_all_archs_ unsigned int teschner_hash(unsigned int x, unsigned int y, unsigned int z) {
    return (x * 73856093) ^ (y * 19349669) ^ (z * 83492791);
}

// Robert Jenkins hash function.
// https://gist.github.com/badboy/6267743
_inline_all_archs_ unsigned int jenkins_hash(unsigned int a) {
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) ^ (a >> 19);
    a = (a + 0x165667b1) + (a << 5);
    a = (a + 0xd3a2646c) ^ (a << 9);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) ^ (a >> 16);
    return a;
}

// Hash Functions for GPU Rendering, Jarzynski et al., 2020, https://jcgt.org/published/0009/03/02/
// Source of all hash functions https://www.shadertoy.com/view/XlGcRh
// Fast hash with good distribution.
_inline_all_archs_ Vector2ui pcg2d(unsigned int x, unsigned int y) {
    x = x * 1664525u + 1013904223u;
    y = y * 1664525u + 1013904223u;

    x += y * 1664525u;
    y += x * 1664525u;

    x = x ^ (x >> 16u);
    y = y ^ (y >> 16u);

    x += y * 1664525u;
    y += x * 1664525u;

    x = x ^ (x >> 16u);
    y = y ^ (y >> 16u);

    return { x, y };
}

// Hash developed by cessen and used in pbrt4.
// Can be used as a fast replacement for Owen-scrambling.
// https://psychopath.io/post/2021_01_30_building_a_better_lk_hash
// https://pbr-book.org/4ed/Sampling_and_Reconstruction/Sobol_Samplers
_inline_all_archs_ unsigned int cessen_owen_hash(unsigned int x, unsigned int seed) {
    x ^= x * 0x3d20adea;
    x += seed;
    x *= (seed >> 16) | 1;
    x ^= x * 0x05526c56;
    x ^= x * 0x53a22864;
    return x;
}

// Hashes x and y ensuring maximal distance between consecutive xs and ys.
// NOTE: Unless filtered afterwards it visually displays a ton of correlation.
_inline_all_archs_ unsigned int even_distribution_2D(unsigned int x, unsigned int y) { return reverse_bits(morton_encode(x, y)); }

// Computes the power heuristic of pdf1 and pdf2.
// It is assumed that pdf1 is always valid, i.e. not NaN.
// pdf2 is allowed to be NaN, but generally try to avoid it. :)
_inline_all_archs_ float power_heuristic(float pdf1, float pdf2) {
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

#ifndef GPU_COMPILATION
// ------------------------------------------------------------------------------------------------
// Generate progressive multi-jittered samples with a blue noise approximation.
// Progressive Multi-Jittered Sample Sequences - Supplemental materials, Christensen et al., 2018
// http://graphics.pixar.com/library/ProgressiveMultiJitteredSampling/pmj_suppl.pdf.
// The nearest neighbour search is implemented by searcing nearby strata for their random samples.
// ------------------------------------------------------------------------------------------------
void fill_progressive_multijittered_bluenoise_samples(Vector2f* samples_begin, Vector2f* samples_end, unsigned int blue_noise_samples = 8);

// ------------------------------------------------------------------------------------------------
// Up to three dimensional blue noise distribution.
// Useful for integrating over BRDFs.
// ------------------------------------------------------------------------------------------------
struct PmjbRNG {
    unsigned int m_max_sample_capacity;
    Bifrost::Math::Vector2f* m_samples;

    PmjbRNG(unsigned int max_sample_capacity) {
        m_max_sample_capacity = max_sample_capacity;
        m_samples = new Bifrost::Math::Vector2f[max_sample_capacity];
        Bifrost::Math::RNG::fill_progressive_multijittered_bluenoise_samples(m_samples, m_samples + max_sample_capacity);
    }
    ~PmjbRNG() {
        delete[] m_samples;
        m_samples = nullptr;
        m_max_sample_capacity = 0;
    }

    Vector2f sample2f(unsigned int i) const { return m_samples[i]; }
    Vector3f sample3f(unsigned int i, unsigned int max_sample_count) const { return Vector3f(m_samples[i], (i + 0.5f) / max_sample_count); }

private:
    // Avoid handling memory moves and copies by deleting all operations.
    PmjbRNG(PmjbRNG& other) = delete;
    PmjbRNG(PmjbRNG&& other) = delete;

    PmjbRNG& operator=(PmjbRNG& rhs) = delete;
    PmjbRNG& operator=(PmjbRNG&& rhs) = delete;
};
#endif

// ------------------------------------------------------------------------------------------------
// Linear congruential random number generator
// ------------------------------------------------------------------------------------------------
struct LinearCongruential final {
    static const unsigned int multiplier = 1664525u;
    static const unsigned int increment = 1013904223u;

    unsigned int m_state;

    LinearCongruential() = default;
    explicit _all_archs_ LinearCongruential(unsigned int seed) : m_state(seed) { }

    _inline_all_archs_ unsigned int sample1ui() {
        m_state = multiplier * m_state + increment;
        return m_state;
    }

    _inline_all_archs_ float sample1f() { return float(sample1ui()) * uint_normalizer; }
    _inline_all_archs_ Vector2f sample2f() { return Vector2f(sample1f(), sample1f()); }
    _inline_all_archs_ Vector3f sample3f() { return Vector3f(sample1f(), sample1f(), sample1f()); }
    _inline_all_archs_ Vector4f sample4f() { return Vector4f(sample1f(), sample1f(), sample1f(), sample1f()); }
};

// ------------------------------------------------------------------------------------------------
// Xor shift random number generator
// https://en.wikipedia.org/wiki/Xorshift
// ------------------------------------------------------------------------------------------------
struct XorShift32 final {
    unsigned int m_state;

    XorShift32() = default;
    explicit _all_archs_ XorShift32(unsigned int seed) : m_state(seed) { }

    _inline_all_archs_ unsigned int sample1ui() {
        m_state ^= m_state << 13;
        m_state ^= m_state >> 17;
        m_state ^= m_state << 5;
        return m_state;
    }

    _inline_all_archs_ float sample1f() { return float(sample1ui()) * uint_normalizer; }
    _inline_all_archs_ Vector2f sample2f() { return Vector2f(sample1f(), sample1f()); }
    _inline_all_archs_ Vector3f sample3f() { return Vector3f(sample1f(), sample1f(), sample1f()); }
    _inline_all_archs_ Vector4f sample4f() { return Vector4f(sample1f(), sample1f(), sample1f(), sample1f()); }
};

// ------------------------------------------------------------------------------------------------
// Practical Hash-based Owen Scrambling, Brent Burley, 2020
// We use primes as our per bounce seed to decorrelate the samples.
// For a practical implementation in a path tracer see Blender's Cycles: sobol_burley.h and path_rng_4D in pattern.h
// ------------------------------------------------------------------------------------------------
struct PracticalScrambledSobol {
private:
    // The original source code from Burley uses a function called hash_combine to mix the seed and dimensions.
    // But no implementation is given in the source and there is no standard c++ hash_combine function.
    // This implementation is found at https://www.shadertoy.com/view/wlyyDm, which unfortunately also doesn't have a source.
    _inline_all_archs_ static unsigned int hash_combine(unsigned int seed, unsigned int v) {
        return seed ^ (v + (seed << 6) + (seed >> 2));
    }

    _inline_all_archs_ static unsigned int nested_uniform_scramble_base2(unsigned int x, unsigned int seed) {
        x = reverse_bits(x);
        x = cessen_owen_hash(x, seed);
        x = reverse_bits(x);
        return x;
    }

    _inline_all_archs_ static Vector4ui sobol_sample4ui(unsigned int index) {
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

public:

    _inline_all_archs_ static Vector4ui sample4ui(unsigned int index, unsigned int seed) {
        index = nested_uniform_scramble_base2(index, seed);
        Vector4ui xs = sobol_sample4ui(index);
        xs.x = nested_uniform_scramble_base2(xs.x, hash_combine(seed, 0));
        xs.y = nested_uniform_scramble_base2(xs.y, hash_combine(seed, 1));
        xs.z = nested_uniform_scramble_base2(xs.z, hash_combine(seed, 2));
        xs.w = nested_uniform_scramble_base2(xs.w, hash_combine(seed, 3));
        return xs;
    }

    // Helper function for generating samples in a path tracer.
    _inline_all_archs_ static Vector4ui sample4ui(unsigned int accumulation_count, unsigned int pixel_hash, unsigned int dimension) {
        // Implemented according to section 5.3 'Use in a Path Tracer'.
        // Specifically we hash the pixel hash and dimension together using a high quality hash function.
        // The reason that this is favored over simply reseeding using an RNG is that this allows dimensions to be sampled out of order.
        unsigned int index = accumulation_count;
        unsigned int seed = pcg2d(pixel_hash, dimension).x;
        return sample4ui(index, seed);
    }

    // Helper function for generating samples in a path tracer.
    _inline_all_archs_ static Vector4f sample4f(unsigned int accumulation_count, unsigned int pixel_hash, unsigned int dimension) {
        return Vector4f(sample4ui(accumulation_count, pixel_hash, dimension)) * uint_normalizer;
    }
};

} // NS Bifrost::Math::RNG

#endif // _BIFROST_MATH_RNG_H_
