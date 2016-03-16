// OptiX random number generators.
// ---------------------------------------------------------------------------
// Copyright (C) 2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _OPTIXRENDERER_RNG_H_
#define _OPTIXRENDERER_RNG_H_

#include <OptiXRenderer/Defines.h>

#include <optixu/optixu_math_namespace.h>

namespace OptiXRenderer {
namespace RNG {

__inline_all__ float van_der_corput(unsigned int n, unsigned int scramble) {

    // Reverse bits of n.
#if (defined(__CUDACC__) || defined(__CUDABE__))
    n = __brev(n);
#else
    n = (n << 16) | (n >> 16);
    n = ((n & 0x00ff00ff) << 8) | ((n & 0xff00ff00) >> 8);
    n = ((n & 0x0f0f0f0f) << 4) | ((n & 0xf0f0f0f0) >> 4);
    n = ((n & 0x33333333) << 2) | ((n & 0xcccccccc) >> 2);
    n = ((n & 0x55555555) << 1) | ((n & 0xaaaaaaaa) >> 1);
#endif
    n ^= scramble;

    return float((n >> 8) & 0xffffff) / float(1 << 24);
}

__inline_all__ float sobol2(unsigned int n, unsigned int scramble) {

    for (unsigned int v = 1u << 31u; n != 0; n >>= 1u, v ^= v >> 1u)
        if (n & 0x1) scramble ^= v;

    return float((scramble >> 8) & 0xffffff) / float(1 << 24);
}

__inline_all__ optix::float2 sample02(unsigned int n, optix::uint2 scramble = optix::make_uint2(5569, 95597)) {
    return optix::make_float2(van_der_corput(n, scramble.x), sobol2(n, scramble.y));
}

class LinearCongruential {
private:
    static const unsigned int multiplier = 1664525u;
    static const unsigned int increment = 1013904223u;
    static const unsigned int max = 0xFFFFFFFFu; // uint32 max.

    unsigned int m_state;

public:
    __inline_all__ void seed(unsigned int seed) { m_state = seed; }
    __inline_all__ unsigned int get_seed() const { return m_state; }

    __inline_all__ unsigned int sample1ui() {
        m_state = multiplier * m_state + increment;
        return m_state;
    }

    __inline_all__ float sample1f() {
        const float inv_max = 1.0f / (float(max) + 1.0f);
        return float(sample1ui()) * inv_max;
    }

    __inline_all__ optix::float2 sample2f() {
        return optix::make_float2(sample1f(), sample1f());
    }

    __inline_all__ optix::float3 sample3f() {
        return optix::make_float3(sample1f(), sample1f(), sample1f());
    }

    __inline_all__ optix::float4 sample4f() {
        return optix::make_float4(sample1f(), sample1f(), sample1f(), sample1f());
    }
};

} // NS RNG
} // NS OptiXRenderer

#endif // _OPTIXRENDERER_RNG_H_