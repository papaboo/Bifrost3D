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
    static const unsigned int PrimitiveID = 8;
    static const unsigned int Count = 9;
};

struct MeshFlags {
    static const unsigned char None = 0u;
    static const unsigned char Normals = 1u << 0u;
    static const unsigned char Texcoords = 1u << 1u;
    static const unsigned char Tints = 1u << 2u;
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

    __inline_all__ optix::float3 decode() const {
        optix::float3 decoded_unnormalized = decode_unnormalized();
        return optix::normalize(decoded_unnormalized);
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

struct InstanceID {
    enum class Type {
        MeshModel = 1,
        LightSource = 2
    };

    int id;

    __inline_all__ static InstanceID make(Type type, int instance_id) {
        InstanceID id = { int(type) << 30 | instance_id };
        return id;
    }

    __inline_all__ bool operator==(const InstanceID& rhs) const { return id == rhs.id; }

    // There is only one analytical light source instance, so we can hardcode the value
    __inline_all__ static InstanceID analytical_light_sources() { return InstanceID::make(InstanceID::Type::LightSource, 1); }
};

struct PrimitiveID {
    InstanceID instance_id;
    int primitive_id;

    __inline_all__ static PrimitiveID make(InstanceID instance_id, int primitive_id) {
        PrimitiveID id = { instance_id, primitive_id };
        return id;
    }

    __inline_all__ bool operator==(const PrimitiveID& rhs) const { return instance_id == rhs.instance_id && primitive_id == rhs.primitive_id; }
};

// PDF wrapper.
// The wrapper contains a PDF and a boolean indicating if the sample can be used for multiple importance sampling (MIS).
// Delta dirac function PDFs are represented by NaN.
__constant_all__ float MIN_VALID_PDF = 0.000001f;
struct PDF {
public:
    float m_PDF;

    PDF() = default;
    __inline_all__ PDF(float pdf) : m_PDF(pdf) {}

    __inline_all__ static PDF invalid() { return PDF(nanf("")); }
    __inline_all__ static PDF delta_dirac(float pdf = 1) { return PDF(-pdf); }

    __inline_all__ bool operator==(PDF rhs) const { return m_PDF == rhs.m_PDF; }
    __inline_all__ bool operator!=(PDF rhs) const { return m_PDF != rhs.m_PDF; }

    __inline_all__ float value() const { return abs(m_PDF); }
    __inline_all__ bool is_valid() const { return value() > MIN_VALID_PDF; }
    __inline_all__ bool is_delta_dirac() const { return !(m_PDF >= 0.0f); }
    __inline_all__ void disable_MIS() { if (m_PDF >= 0.0f) m_PDF = -m_PDF; }
    __inline_all__ bool is_valid_and_not_delta_dirac() const { return m_PDF > MIN_VALID_PDF; }
    __inline_all__ bool invalid_or_delta_dirac() const { return !(m_PDF > MIN_VALID_PDF); }
    __inline_all__ bool use_for_MIS() const { return is_valid_and_not_delta_dirac(); }

    __inline_all__ PDF& scale(float s) {
#ifdef _DEBUG
        if (s < 0.0f)
            THROW(OPTIX_NEGATIVE_PDF_SCALE_EXCEPTION);
#endif

        m_PDF *= s;
        return *this;
    }
    __inline_all__ PDF& operator*=(float s) { return scale(s); }
    __inline_all__ PDF operator*(float s) const { PDF copy = *this; return copy.scale(s); }

    __inline_all__ PDF& add(PDF rhs) {
#ifdef _DEBUG
        if (is_delta_dirac() || rhs.is_delta_dirac())
            THROW(OPTIX_DELTA_DIRAC_PDF_ADDITION_EXCEPTION);
#endif

        m_PDF += rhs.m_PDF;
        return *this;
    }
    __inline_all__ PDF& operator+=(PDF rhs) { return add(rhs); }
    __inline_all__ PDF operator+(PDF rhs) const { PDF copy = *this; return copy.add(rhs); }

    __inline_all__ static bool is_valid(float PDF) {
        return PDF > 0.000001f;
    }
};

//----------------------------------------------------------------------------
// Light source structs.
//----------------------------------------------------------------------------

struct __align__(16) LightSample {
    optix::float3 radiance;
    PDF PDF;
    optix::float3 direction_to_light;
    float distance;

    __inline_all__ static LightSample none() {
        LightSample sample = {};
        sample.direction_to_light = { 0, 1, 0 };
        sample.PDF = PDF::delta_dirac(0);
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
    unsigned short PDF_width;
    unsigned short PDF_height;

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

// NOTE the suboptimal alignment of 8 instead of 16 yields a tiny tiny performance benefit. I have no clue why.
struct __align__(8) BSDFResponse {
    optix::float3 reflectance;
    PDF PDF;

    __inline_all__ static BSDFResponse none() {
        BSDFResponse evaluation = {};
        return evaluation;
    }
};

struct __align__(16) BSDFSample {
    optix::float3 reflectance;
    PDF PDF;
    optix::float3 direction;
    float __padding;

    __inline_all__ static BSDFSample none() {
        BSDFSample sample = {};
        return sample;
    }
};

struct __align__(16) Material {
    enum Flags : unsigned short {
        None = 0u,
        ThinWalled = 1u,
        Cutout = 2u,
    };

    enum ShadingModel : unsigned short {
        Default = 0u,
        Diffuse = 1u,
        Transmissive = 2u,
    };

    Flags flags;
    ShadingModel shading_model;
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
    __inline_all__ bool is_transmissive() const { return shading_model == ShadingModel::Transmissive; }

#if GPU_DEVICE
    __inline_all__ optix::float4 get_tint_roughness(optix::float2 texcoord) const {
        optix::float4 tint_roughness = optix::make_float4(tint, roughness);
        if (tint_roughness_texture_ID)
            tint_roughness *= optix::rtTex2D<optix::float4>(tint_roughness_texture_ID, texcoord.x, texcoord.y);
        if (roughness_texture_ID)
            tint_roughness.w *= optix::rtTex2D<float>(roughness_texture_ID, texcoord.x, texcoord.y);
        return tint_roughness;
    }

    __inline_all__ float get_metallic(optix::float2 texcoord) const {
        if (metallic_texture_ID)
            return metallic * optix::rtTex2D<float>(metallic_texture_ID, texcoord.x, texcoord.y);
        else
            return metallic;
    }

    __inline_all__ float get_coverage(optix::float2 texcoord) const {
        float coverage_tex_sample = 1.0f;
        if (coverage_texture_ID)
            coverage_tex_sample = optix::rtTex2D<float>(coverage_texture_ID, texcoord.x, texcoord.y);

        if (is_cutout())
            return coverage_tex_sample < coverage ? 0 : 1;
        else
            return coverage * coverage_tex_sample;
    }
#endif // GPU_DEVICE
};

//----------------------------------------------------------------------------
// Ray payloads.
//----------------------------------------------------------------------------

struct __align__(16) MonteCarloPayload {
    optix::float3 radiance;
#if LCG_RNG
    RNG::LinearCongruential rng;
#elif PRACTICAL_SOBOL_RNG
    RNG::PracticalScrambledSobol rng;
#endif
    optix::float3 throughput;
    unsigned int bounces;

    optix::float3 position;
    float ray_min_t;
    optix::float3 direction;
    PDF bsdf_PDF;

    LightSample light_sample;
    optix::float3 light_sample_origin;
    optix::uchar4 tint_and_roughness_scale;

    optix::float3 shading_normal;
    int material_index;

    optix::float2 texcoord;
    PrimitiveID primitive_id;

    __inline_dev__ void debug_output(optix::float3 color) {
        throughput = { 0,0,0 };
        radiance = color;
    }
};

struct ShadowPayload {
    optix::float3 radiance;
};

//----------------------------------------------------------------------------
// Model state
//----------------------------------------------------------------------------

struct ModelState {
    InstanceID instance_id;
    int material_index;
};

//----------------------------------------------------------------------------
// Camera data uploaded to the GPU
//----------------------------------------------------------------------------

struct __align__(16) CameraStateGPU {
    optix::Matrix3x3 view_to_world_rotation;
    optix::Matrix4x4 inverse_projection_matrix;
    optix::Matrix4x4 inverse_view_projection_matrix;
    unsigned int accumulations;
    unsigned int max_bounce_count;
#ifdef DOUBLE_PRECISION_ACCUMULATION_BUFFER
    rtBufferId<optix::double4, 2> accumulation_buffer;
#else
    rtBufferId<optix::float4, 2> accumulation_buffer;
#endif
    rtBufferId<optix::ushort4, 2> output_buffer;

    float path_regularization_PDF_scale;
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

    int next_event_sample_count;

    // -- Aligned to 8'th word from here --
    rtBufferId<Light, 1> light_buffer;
    unsigned int light_count;

    optix::int2 __padding;
};

//----------------------------------------------------------------------------
// AI denoiser data uploaded to OptiX
//----------------------------------------------------------------------------

struct __align__(8) AIDenoiserStateGPU {
    unsigned int flags;
    rtBufferId<optix::float4, 2> noisy_pixels_buffer;
    rtBufferId<optix::float4, 2> denoised_pixels_buffer;
    rtBufferId<optix::float4, 2> albedo_buffer;
};

} // NS OptiXRenderer

#endif // _OPTIXRENDERER_TYPES_H_