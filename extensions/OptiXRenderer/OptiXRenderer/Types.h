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
    static const unsigned int Depth = 3;
    static const unsigned int Albedo = 4;
    static const unsigned int Tint = 5;
    static const unsigned int Roughness = 6;
    static const unsigned int ShadingNormal = 7;
    static const unsigned int Count = 8;
};

struct MeshFlags {
    static const unsigned char None = 0u;
    static const unsigned char Normals = 1u << 0u;
    static const unsigned char Texcoords = 1u << 1u;
};

//-------------------------------------------------------------------------------------------------
// OptiX decoder for Bifrost::Math::OctahedralNormal.
//-------------------------------------------------------------------------------------------------
struct __align__(4) OctahedralNormal {

    optix::short2 encoding;

    __inline_all__ static float sign(float v) { return v >= 0.0f ? +1.0f : -1.0f; }

    __inline_all__ optix::float3 decode_unnormalized() const {
        optix::float2 p2 = optix::make_float2(encoding.x, encoding.y);
        optix::float3 n = optix::make_float3(p2, SHRT_MAX - fabsf(p2.x) - fabsf(p2.y));
        if (n.z < 0.0f) {
            float tmp_x = (SHRT_MAX - fabsf(n.y)) * sign(n.x);
            n.y = (SHRT_MAX - fabsf(n.x)) * sign(n.y);
            n.x = tmp_x;
        }
        return n;
    }
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

struct Disk {
    optix::float3 center;
    optix::float3 normal;
    float radius;

    __inline_all__ static Disk make(optix::float3 center, optix::float3 normal, float radius) {
        Disk d = { center, normal, radius };
        return d;
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

struct SpotLight {
    optix::float3 power;
    optix::float3 position;
    float radius;
    optix::float3 direction;
    float cos_angle;
};

struct DirectionalLight {
    optix::float3 radiance;
    optix::float3 direction;
    float __padding;
};

struct EnvironmentLight {
    int environment_map_ID;
    int marginal_CDF_ID;
    int conditional_CDF_ID;
    int per_pixel_PDF_ID;
    unsigned short tint_x; // Tint stored as fixedpoint
    unsigned short tint_y; // Tint stored as fixedpoint
    unsigned short tint_z; // Tint stored as fixedpoint
    unsigned short __padding; // Tint stored as fixedpoint
    unsigned short width;
    unsigned short height;

    __inline_all__ static EnvironmentLight empty(optix::float3 tint) {
        EnvironmentLight light = {};
        light.set_tint(tint);
        return light;
    }

    __inline_all__ void set_tint(optix::float3 tint) {
        tint_x = (unsigned short)fmaxf(65535.0f, tint.x * 65535.0f + 0.5f);
        tint_y = (unsigned short)fmaxf(65535.0f, tint.y * 65535.0f + 0.5f);
        tint_z = (unsigned short)fmaxf(65535.0f, tint.z * 65535.0f + 0.5f);
    }

    __inline_all__ optix::float3 get_tint() const {
        return optix::make_float3(tint_x / 65535.0f, tint_y / 65535.0f, tint_z / 65535.0f);
    }
};

struct PresampledEnvironmentLight {
    int environment_map_ID; // Texture ID.
    int per_pixel_PDF_ID; // Texture ID.
    int samples_ID; // Buffer ID.
    int sample_count;
    optix::float3 tint;

    __inline_all__ static PresampledEnvironmentLight empty(optix::float3 tint) {
        PresampledEnvironmentLight light = {};
        light.tint = tint;
        return light;
    }

    __inline_all__ void set_tint(optix::float3 t) { tint = t; }
    __inline_all__ optix::float3 get_tint() const { return tint; }
};

struct __align__(16) Light {
    enum Flags {
        None = 0u,
        Sphere = 1u,
        Directional = 2u,
        Environment = 3u,
        PresampledEnvironment = 4u,
        Spot = 5u,
        TypeMask = 7u
    };

    union {
        SphereLight sphere;
        DirectionalLight directional;
        EnvironmentLight environment;
        PresampledEnvironmentLight presampled_environment;
        SpotLight spot;
    };

    unsigned int flags;
    __inline_all__ unsigned int get_type() const { return flags & TypeMask; }
    __inline_all__ bool is_type(Flags light_type) { return get_type() == light_type; }
};

//----------------------------------------------------------------------------
// Material type and sampling structs.
//----------------------------------------------------------------------------

// PDF wrapper for multiple importance sampling.
// The wrapper contains a PDF and a boolean indicating if the PDF should be used for multiple importance sampling or not.
struct MisPDF {
private:
    float m_PDF;

public:
    __inline_all__ static MisPDF from_PDF(float pdf) {
        MisPDF mis_PDF;
        mis_PDF.m_PDF = pdf;
        return mis_PDF;
    }
    __inline_all__ static MisPDF delta_dirac() {
        MisPDF mis_PDF;
        mis_PDF.m_PDF = -FLT_MAX;
        return mis_PDF;
    }

    __inline_all__ bool use_for_MIS() const { return m_PDF > 0.0f; }
    __inline_all__ float PDF() const { return abs(m_PDF); }
};

// NOTE the suboptimal alignment of 8 instead of 16 yields a tiny tiny performance benefit. I have no clue why.
struct __align__(8) BSDFResponse {
    optix::float3 reflectance;
    float PDF;

    __inline_all__ static BSDFResponse none() {
        BSDFResponse evaluation = {};
        return evaluation;
    }
};

struct __align__(16) BSDFSample {
    optix::float3 reflectance;
    float PDF;
    optix::float3 direction;
    float __padding;

    __inline_all__ static BSDFSample none() {
        BSDFSample sample = {};
        return sample;
    }
};

struct __align__(16) Material {
    enum Flags {
        None = 0u,
        ThinWalled = 1u,
        Cutout = 2u,
    };

    int flags;
    optix::float3 tint;

    float roughness;
    int tint_roughness_texture_ID;
    int roughness_texture_ID; // Should only be set if tint_roughness_texture_ID is 0. Can be packed with tint_roughness_texture_ID if needed and the most significant bits can be used to denote the type.
    float specularity;

    float metallic;
    int metallic_texture_ID;
    float coverage;
    int coverage_texture_ID;

    float coat;
    float coat_roughness;
    optix::float2 __padding2;

    __inline_all__ bool is_thin_walled() const { return (flags & (Flags::Cutout | Flags::ThinWalled)) != 0; }
    __inline_all__ bool is_cutout() const { return (flags & Flags::Cutout) != 0; }
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
#if LCG_RNG
    RNG::LinearCongruential rng_state;
#elif PMJ_RNG
    PMJSamplerState rng_state;
#endif
    optix::float3 throughput;
    unsigned int bounces;

    optix::float3 position;
    MisPDF bsdf_MIS_PDF;
    optix::float3 direction;
    int __padding0;

    LightSample light_sample;

    optix::float3 shading_normal;
    int material_index;

    optix::float2 texcoord;
    optix::float2 __padding1;
};

struct ShadowPayload {
    optix::float3 radiance;
};

//----------------------------------------------------------------------------
// Camera data uploaded to the GPU
//----------------------------------------------------------------------------

struct __align__(16) CameraStateGPU {
    optix::Matrix3x3 world_to_view_rotation;
    optix::Matrix4x4 inverse_view_projection_matrix;
    unsigned int accumulations;
    unsigned int max_bounce_count;
#ifdef DOUBLE_PRECISION_ACCUMULATION_BUFFER
    rtBufferId<optix::double4, 2> accumulation_buffer;
#else
    rtBufferId<optix::float4, 2> accumulation_buffer;
#endif
    rtBufferId<optix::ushort4, 2> output_buffer;

    float path_regularization_scale;
    optix::float3 _padding;
};

//----------------------------------------------------------------------------
// Scene data uploaded to the GPU
//----------------------------------------------------------------------------

struct __align__(16) SceneStateGPU {
#if PRESAMPLE_ENVIRONMENT_MAP
    static_assert(sizeof(PresampledEnvironmentLight) == sizeof(float) * 7, "PresampledEnvironmentLight expected to take up seven floats");
    PresampledEnvironmentLight environment_light;
#else
    static_assert(sizeof(EnvironmentLight) == sizeof(float) * 7, "EnvironmentLight expected to take up seven floats");
    EnvironmentLight environment_light; // Takes up 7 ints
#endif

    float ray_epsilon;

    // -- Aligned to 8'th byte from here --
    rtBufferId<Light, 1> light_buffer;
    unsigned int light_count;
};

//----------------------------------------------------------------------------
// AI denoiser data uploaded to OptiX
//----------------------------------------------------------------------------

struct __align__(8) AIDenoiserStateGPU {
    static const unsigned int ResetAlbedoAccumulation = 1 << 16;
    static const unsigned int ResetNormalAccumulation = 1 << 17;
    unsigned int flags;
    rtBufferId<optix::float4, 2> noisy_pixels_buffer;
    rtBufferId<optix::float4, 2> denoised_pixels_buffer;
    rtBufferId<optix::float4, 2> albedo_buffer;
    rtBufferId<optix::float4, 2> normals_buffer;
};

} // NS OptiXRenderer

#endif // _OPTIXRENDERER_TYPES_H_