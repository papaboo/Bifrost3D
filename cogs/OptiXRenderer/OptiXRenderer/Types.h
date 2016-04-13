// OptiX renderer POD types.
// ---------------------------------------------------------------------------
// Copyright (C) 2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _OPTIXRENDERER_TYPES_H_
#define _OPTIXRENDERER_TYPES_H_

#include <OptiXRenderer/Defines.h>
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

namespace MeshFlags {
static const unsigned char None = 0u;
static const unsigned char Normals = 1u << 0u;
static const unsigned char Texcoords = 1u << 1u;
};

//----------------------------------------------------------------------------
// Base types.
//----------------------------------------------------------------------------

struct __align__(16) Sphere {
    optix::float3 center;
    float radius;

    __inline_all__ static Sphere make(optix::float3 center, float radius) {
        Sphere s = { center, radius };
        return s;
    }
};

//----------------------------------------------------------------------------
// Per ray data.
//----------------------------------------------------------------------------

struct __align__(16) MonteCarloPRD {
    optix::float3 radiance;
    RNG::LinearCongruential rng;
    optix::float3 throughput;
    unsigned int bounces;

    optix::float3 position;
    float bsdf_sample_pdf;
    optix::float3 direction;
};

struct ShadowPRD {
    optix::float3 attenuation;
};

//----------------------------------------------------------------------------
// Light source structs.
//----------------------------------------------------------------------------

struct __align__(16) LightSample {
    optix::float3 radiance;
    float PDF;
    optix::float3 direction;
    float distance;

    __inline_all__ static LightSample none() {
        LightSample sample = {};
        return sample;
    }

    __inline_all__ bool is_valid() { return PDF > 0.000001f; }
};

struct __align__(16) SphereLight {
    unsigned int flags;
    optix::float3 power;
    optix::float3 position;
    float radius;
};

//----------------------------------------------------------------------------
// Material type and sampling structs.
//----------------------------------------------------------------------------

struct __align__(16) BSDFSample {
    optix::float3 weight;
    float PDF;
    optix::float3 direction;
    float __padding;

    __inline_all__ static BSDFSample none() {
        BSDFSample sample = {};
        return sample;
    }

    __inline_all__ bool is_valid() { return PDF > 0.000001f; }
};


struct __align__(16) Material {
    optix::float3 base_tint;
    float base_roughness;
    float specularity;
    float metallic;
    optix::float2 __padding;
};

} // NS OptiXRenderer

#endif // _OPTIXRENDERER_TYPES_H_