// OptiX random number generators.
// ---------------------------------------------------------------------------
// Copyright (C) 2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _OPTIXRENDERER_RNG_H_
#define _OPTIXRENDERER_RNG_H_

#include <OptiXRenderer/Shading/Defines.h>

#include <optixu/optixu_math_namespace.h>

namespace OptiXRenderer {
namespace RNG {

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