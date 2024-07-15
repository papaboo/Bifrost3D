// Material model parameters.
// ------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _DX11_RENDERER_SHADERS_SHADING_MODELS_PARAMETERS_H_
#define _DX11_RENDERER_SHADERS_SHADING_MODELS_PARAMETERS_H_

#include <Utils.hlsl>

namespace ShadingModels {

// ------------------------------------------------------------------------------------------------
// Textures.
// ------------------------------------------------------------------------------------------------

Texture2D environment_tex : register(t0);
SamplerState environment_sampler : register(s0);

Texture2D coverage_tex : register(t1);
SamplerState coverage_sampler : register(s1);

Texture2D tint_roughness_tex : register(t2);
SamplerState tint_roughness_sampler : register(s2);

Texture2D metallic_tex : register(t3);
SamplerState metallic_sampler : register(s3);

Texture2D<float2> ggx_with_fresnel_rho_tex : register(t15);

// ------------------------------------------------------------------------------------------------
// Specular rho helpers.
// ------------------------------------------------------------------------------------------------
struct SpecularRho {
    static const int angle_sample_count = 32;
    static const int roughness_sample_count = 32;

    float base, full;
    float rho(float specularity) { return lerp(base, full, specularity); }
    float3 rho(float3 specularity) {
        return float3(rho(specularity.x), rho(specularity.y), rho(specularity.z));
    }

    // Compensate for lost energy due to multiple scattering.
    // Multiple-scattering microfacet BSDFs with the smith model, Heitz et al., 2016 and 
    // Practical multiple scattering compensation for microfacet models, Emmanuel Turquin, 2018
    // showed that multiple-scattered reflectance has roughly the same distribution as single-scattering reflectance.
    // We can therefore account for energy lost to multi-scattering by computing the ratio of lost energy of a fully specular material,
    // and then scaling the specular reflectance by that ratio during evaluation, which increases reflectance to account for energy lost.
    float energy_loss_adjustment() { return 1.0f / full; }

    static SpecularRho fetch(float abs_cos_theta, float roughness) {
        // Adjust UV coordinates to start sampling half a pixel into the texture, as the pixel values correspond to the boundaries of the rho function.
        float cos_theta_uv = lerp(0.5 / angle_sample_count, (angle_sample_count - 0.5) / angle_sample_count, abs_cos_theta);
        float roughness_uv = lerp(0.5 / roughness_sample_count, (roughness_sample_count - 0.5) / roughness_sample_count, roughness);

        float2 specular_rho = ggx_with_fresnel_rho_tex.SampleLevel(bilinear_sampler, float2(cos_theta_uv, roughness_uv), 0);
        SpecularRho res;
        res.base = specular_rho.r;
        res.full = specular_rho.g;
        return res;
    }
};

static float3 compute_specular_rho(float3 specularity, float abs_cos_theta, float roughness) {
    return SpecularRho::fetch(abs_cos_theta, roughness).rho(specularity);
}

// ------------------------------------------------------------------------------------------------
// Parameters.
// ------------------------------------------------------------------------------------------------

struct TextureBound {
    static const unsigned int None = 0;
    static const unsigned int Tint = 1 << 0;
    static const unsigned int Roughness = 1 << 1;
    static const unsigned int Tint_Roughness = Tint | Roughness;
    static const unsigned int Coverage = 1 << 2;
    static const unsigned int Metallic = 1 << 3;
};

struct Parameters {
    float3 m_tint;
    unsigned int m_textures_bound;
    float m_roughness;
    float m_specularity;
    float m_metallic;
    float m_coverage;
    float m_coat;
    float m_coat_roughness;
    int shading_model; // Shading model pixel shaders are set on the host, but this can be used to validate that the correct pixel shaders are used.
    float __padding;

    float4 tint_roughness(float2 texcoord) {
        float4 tint_roughness = float4(m_tint, m_roughness);
        if (m_textures_bound & TextureBound::Tint_Roughness) {
            float4 tex_scale = tint_roughness_tex.Sample(tint_roughness_sampler, texcoord);
            tex_scale.rgb = (m_textures_bound & TextureBound::Tint) == 0 ? float3(1, 1, 1) : tex_scale.rgb;
            tint_roughness *= tex_scale;
        }
        return tint_roughness;
    }

    float metallic(float2 texcoord) {
        float metallic = m_metallic;
        if (m_textures_bound & TextureBound::Metallic)
            metallic *= metallic_tex.Sample(metallic_sampler, texcoord).a;
        return metallic;
    }

    float coverage(float2 texcoord) {
        float coverage = m_coverage;
        if (m_textures_bound & TextureBound::Coverage)
            coverage *= coverage_tex.Sample(coverage_sampler, texcoord).a;
        return coverage;
    }

    bool discard_from_cutout(float2 texcoord) {
        if (m_textures_bound & TextureBound::Coverage)
            return coverage_tex.Sample(coverage_sampler, texcoord).a < m_coverage;
        else
            return false;
    }

    float coat_scale() { return m_coat; }
    float coat_roughness() { return m_coat_roughness; }
};

} // NS ShadingModels

#endif // _DX11_RENDERER_SHADERS_SHADING_MODELS_PARAMETERS_H_