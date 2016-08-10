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
    float bsdf_MIS_PDF; // If negative, then it indicates that MIS should not be used.
    optix::float3 direction;
    float path_PDF;

    float clamped_path_PDF; // The same as the path PDF, but the PDF's are clamped to 1 before being applied.
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
    optix::float3 direction_to_light;
    float distance;

    __inline_all__ static LightSample none() {
        LightSample sample = {};
        return sample;
    }
};

struct SphereLight {
    optix::float3 power;
    optix::float3 position;
    float radius;
};

struct DirectionalLight {
    optix::float3 radiance;
    optix::float3 direction;
    float __padding;
};

struct EnvironmentLight {
    int width;
    int height;
    int environment_map_ID;
    int marginal_CDF_ID;
    int conditional_CDF_ID;
    int per_pixel_PDF_ID;
};

struct __align__(16) Light {
    enum Flags {
        None = 0u,
        Sphere = 1u,
        Directional = 2u,
        Environment = 3u,
        TypeMask = 3u
    };

    union {
        SphereLight sphere;
        DirectionalLight directional;
        EnvironmentLight environment;
    };

    unsigned int flags;
    __inline_all__ unsigned int get_type() const { return flags & TypeMask; }
    __inline_all__ bool is_type(Flags light_type) { return get_type() == light_type; }
};

//----------------------------------------------------------------------------
// Material type and sampling structs.
//----------------------------------------------------------------------------

// NOTE the suboptimal alignment of 8 instead of 16 yields a tiny tiny performance benefit. I have no clue why.
struct __align__(8) BSDFResponse {
    optix::float3 weight;
    float PDF;

    __inline_all__ static BSDFResponse none() {
        BSDFResponse evaluation = {};
        return evaluation;
    }
};

struct __align__(16) BSDFSample {
    optix::float3 weight;
    float PDF;
    optix::float3 direction;
    float __padding;

    __inline_all__ static BSDFSample none() {
        BSDFSample sample = {};
        return sample;
    }
};

struct __align__(16) Material {
    optix::float3 base_tint;
    unsigned int base_tint_texture_ID;
    float base_roughness;
    float specularity;
    float metallic;
    float __padding;
};

} // NS OptiXRenderer

#endif // _OPTIXRENDERER_TYPES_H_