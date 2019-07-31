// OptiX renderer POD types.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _OPTIXRENDERER_TYPES_H_
#define _OPTIXRENDERER_TYPES_H_

#include <OptiXRenderer/Defines.h>
#include <OptiXRenderer/OctahedralNormal.h>
#include <OptiXRenderer/PublicTypes.h>
#include <OptiXRenderer/RNG.h>

#ifndef GPU_DEVICE
#include <optixu/optixpp_namespace.h>
#undef RGB
#endif // GPU_DEVICE
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_matrix_namespace.h>

#include <cuda_fp16.h>

namespace OptiXRenderer {

struct RayTypes {
    static const unsigned int MonteCarlo = 0;
    static const unsigned int Shadow = 1;
    static const unsigned int Count = 2;
};

struct EntryPoints {
    static const unsigned int PathTracing = 0;
    static const unsigned int AIDenoiserPathTracing = 1;
    static const unsigned int AIDenoiserCopyOutput = 2;
    static const unsigned int Albedo = 3;
    static const unsigned int Count = 4;
};

struct MeshFlags {
    static const unsigned char None = 0u;
    static const unsigned char Normals = 1u << 0u;
    static const unsigned char Texcoords = 1u << 1u;
};

//----------------------------------------------------------------------------
// Base types.
//----------------------------------------------------------------------------

struct __align__(16) Cone {
    optix::float3 direction;
    float cos_theta;

    __inline_all__ static Cone make(optix::float3 direction, float cos_theta) {
        Cone c = { direction, cos_theta };
        return c;
    }
};

struct __align__(16) Sphere {
    optix::float3 center;
    float radius;

    __inline_all__ static Sphere make(optix::float3 center, float radius) {
        Sphere s = { center, radius };
        return s;
    }
};

struct __align__(16) VertexGeometry {
    optix::float3 position;
    OctahedralNormal normal;
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
        sample.direction_to_light = { 0, 1, 0 };
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

    __inline_all__ static EnvironmentLight none() {
        EnvironmentLight light = {};
        return light;
    }
};

struct PresampledEnvironmentLight {
    int environment_map_ID; // Texture ID.
    int per_pixel_PDF_ID; // Texture ID.
    int samples_ID; // Buffer ID.
    int sample_count;

    __inline_all__ static PresampledEnvironmentLight none() {
        PresampledEnvironmentLight light = {};
        return light;
    }
};

struct __align__(16) Light {
    enum Flags {
        None = 0u,
        Sphere = 1u,
        Directional = 2u,
        Environment = 3u,
        PresampledEnvironment = 4u,
        TypeMask = 7u
    };

    union {
        SphereLight sphere;
        DirectionalLight directional;
        EnvironmentLight environment;
        PresampledEnvironmentLight presampled_environment;
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
    optix::float3 tint;
    int tint_roughness_texture_ID;
    float roughness;
    int roughness_texture_ID; // Should only be set if tint_roughness_texture_ID is 0. Can be packed with tint_roughness_texture_ID if needed and the most significant bits can be used to denote the type.
    float specularity;
    float metallic;
    int metallic_texture_ID;
    float coverage;
    int coverage_texture_ID;
    int __padding;
};

//----------------------------------------------------------------------------
// Ray payloads.
//----------------------------------------------------------------------------

struct PMJSamplerState {
public:
    static const unsigned int MAX_SAMPLE_COUNT = 4096;
    unsigned int iteration : 13;
    unsigned int dimension : 5;
    unsigned int x_offset : 7;
    unsigned int y_offset : 7;

    __inline_all__ static PMJSamplerState make(unsigned int iteration, float x_offset, float y_offset) {
        PMJSamplerState sampler;
        sampler.iteration = iteration;
        sampler.dimension = 0;
        sampler.x_offset = unsigned int(x_offset * (1 << 7) + 0.5f);
        sampler.y_offset = unsigned int(y_offset * (1 << 7) + 0.5f);
        return sampler;
    }

    __inline_all__ void set_dimension(unsigned int dim) { this->dimension = dim; }
    __inline_all__ int get_index() const {
        if (dimension == 0)
            return iteration % MAX_SAMPLE_COUNT;
        return (iteration ^ (iteration >> dimension)) % MAX_SAMPLE_COUNT;
    }

    __inline_all__ int get_index_1d() { int index = get_index(); dimension += 1; return index; }
    __inline_all__ int get_index_2d() { int index = get_index(); dimension += 2; return index; }
    __inline_all__ float scramble(float x) const { return fmodf(x + x_offset / 128.0f, 1); }
    __inline_all__ optix::float2 scramble(optix::float2 v) const { return{ fmodf(v.x + x_offset / 128.0f, 1), fmodf(v.y + y_offset / 128.0f, 1) }; }
};

struct __align__(16) MonteCarloPayload {
    optix::float3 radiance;
    PMJSamplerState pmj_rng_state;
    optix::float3 throughput;
    unsigned int bounces;

    optix::float3 position;
    float bsdf_MIS_PDF; // If negative, then it indicates that MIS should not be used.
    optix::float3 direction;
    int __padding0;

    optix::float3 shading_normal;
    int material_index;

    optix::float2 texcoord;
    optix::float2 __padding1;
};

struct ShadowPayload {
    optix::float3 attenuation;
};

//----------------------------------------------------------------------------
// Camera data uploaded to the GPU
//----------------------------------------------------------------------------

struct __align__(16) CameraStateGPU {
    optix::float4  camera_position;
    optix::Matrix4x4 inverted_view_projection_matrix;
    unsigned int accumulations;
    unsigned int max_bounce_count;
#ifdef DOUBLE_PRECISION_ACCUMULATION_BUFFER
    rtBufferId<optix::double4, 2> accumulation_buffer;
#else
    rtBufferId<optix::float4, 2> accumulation_buffer;
#endif
    rtBufferId<optix::ushort4, 2> output_buffer;
};

//----------------------------------------------------------------------------
// Scene data uploaded to the GPU
//----------------------------------------------------------------------------

struct __align__(16) SceneStateGPU {
#if PRESAMPLE_ENVIRONMENT_MAP
    PresampledEnvironmentLight environment_light; // Takes up 4 ints and 16 byte aligned.
    optix::float2 __padding;
#else
    EnvironmentLight environment_light; // Takes up 6 ints
#endif

    // -- Aligned to 8 byte from here --

    rtBufferId<Light, 1> light_buffer;
    unsigned int light_count;
    optix::float3 environment_tint;
    float ray_epsilon;
};

//----------------------------------------------------------------------------
// AI denoiser data uploaded to OptiX
//----------------------------------------------------------------------------

struct __align__(8) AIDenoiserStateGPU {
    unsigned int flags;
    rtBufferId<optix::float4, 2> noisy_pixels_buffer;
    rtBufferId<optix::float4, 2> denoised_pixels_buffer;
    rtBufferId<optix::float4, 2> albedo_buffer;
    rtBufferId<optix::float4, 2> normals_buffer;
};

} // NS OptiXRenderer

#endif // _OPTIXRENDERER_TYPES_H_