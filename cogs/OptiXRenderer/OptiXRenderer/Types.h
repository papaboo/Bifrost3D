// OptiX renderer POD types.
// ---------------------------------------------------------------------------
// Copyright (C) 2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _OPTIXRENDERER_TYPES_H_
#define _OPTIXRENDERER_TYPES_H_

#include <OptiXRenderer/Shading/Defines.h>
#include <OptiXRenderer/RNG.h>

#include <optixu/optixu_math_namespace.h>

namespace OptiXRenderer {

enum class RayTypes {
    MonteCarlo = 0,
    NormalVisualization,
    Shadow,
    Count
};

enum class EntryPoints {
    PathTracing = 0,
    NormalVisualization,
    Count
};

//----------------------------------------------------------------------------
// Per ray data.
//----------------------------------------------------------------------------

struct __align__(16) MonteCarloPRD{
    optix::float3 radiance;
    RNG::LinearCongruential rng;
    optix::float3 throughput;
};

struct ShadowPRD {
    optix::float3 attenuation;
};

//----------------------------------------------------------------------------
// Light source structs
//----------------------------------------------------------------------------

struct __align__(16) LightSample {
    optix::float3 radiance;
    float PDF;
    optix::float3 direction;
    float distance;

    __inline_all__ static LightSample None() {
        LightSample sample = {};
        return sample;
    }
};

struct __align__(16) PointLight{
    unsigned int flags;
    optix::float3 position;
    optix::float3 power;
    float radius;
};

} // NS OptiXRenderer

#endif // _OPTIXRENDERER_TYPES_H_