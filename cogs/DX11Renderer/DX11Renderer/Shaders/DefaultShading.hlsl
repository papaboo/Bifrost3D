// Default shading.
// ---------------------------------------------------------------------------
// Copyright (C) 2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _DX11_RENDERER_SHADERS_DEFAULT_SHADING_H_
#define _DX11_RENDERER_SHADERS_DEFAULT_SHADING_H_

#include "BSDFs/Diffuse.hlsl"
#include "BSDFs/GGX.hlsl"
#include "Utils.hlsl"

//-----------------------------------------------------------------------------
// Textures.
//-----------------------------------------------------------------------------

Texture2D environment_tex : register(t0);
SamplerState environment_sampler : register(s0);

Texture2D color_tex : register(t1);
SamplerState color_sampler : register(s1);

Texture2D coverage_tex : register(t2);
SamplerState coverage_sampler : register(s2);

Texture2D ggx_with_fresnel_rho_tex : register(t15);
SamplerState precomputation2D_sampler : register(s15);

//-----------------------------------------------------------------------------
// Default shading.
//-----------------------------------------------------------------------------

struct DefaultShading {
    float3 m_tint;
    unsigned int m_tint_texture_index;
    float m_roughness;
    float m_specularity;
    float m_metallic;
    float m_coverage;
    unsigned int m_coverage_texture_index;
    int3 __padding;

    float compute_specularity(float specularity, float metalness) {
        float dielectric_specularity = specularity * 0.08f; // See Physically-Based Shading at Disney bottom of page 8.
        float metal_specularity = specularity * 0.2f + 0.6f;
        return lerp(dielectric_specularity, metal_specularity, metalness);
    }

    float compute_specular_rho(float specularity, float abs_cos_theta, float roughness) {
        float base_specular_rho = ggx_with_fresnel_rho_tex.Sample(precomputation2D_sampler, float2(abs_cos_theta, roughness)).r;
        float full_specular_rho = 1.0f; // TODO This is wrong. GGX doesn't have a rho of one. Try to use the actual GGX rho instead.
        return lerp(base_specular_rho, full_specular_rho, specularity);
    }

    float coverage(float2 texcoord) {
        float c = m_coverage;
        if (m_coverage_texture_index != 0)
            c *= coverage_tex.Sample(coverage_sampler, texcoord).a;
        return c;
    }

    float roughness() { return m_roughness; }

    void evaluate_tints(float3 wo, float3 wi, float2 texcoord, 
                        out float3 diffuse_tint, out float3 specular_tint) {
        bool is_same_hemisphere = wi.z * wo.z >= 0.00000001f;
        if (!is_same_hemisphere) {
            diffuse_tint = specular_tint = float3(0.0f, 0.0f, 0.0f);
            return;
        }

        // Flip directions if on the backside of the material.
        if (wo.z < 0.0f) {
            wi.z = -wi.z;
            wo.z = -wo.z;
        }

        float3 tint = m_tint;
        if (m_tint_texture_index != 0)
            tint *= color_tex.Sample(color_sampler, texcoord).rgb;

        float specularity = compute_specularity(m_specularity, m_metallic);

        diffuse_tint = tint * (1.0f - compute_specular_rho(specularity, wo.z, m_roughness));

        float3 halfway = normalize(wo + wi);
        float fresnel = schlick_fresnel(specularity, dot(wo, halfway));
        specular_tint = fresnel * lerp(float3(1.0f, 1.0f, 1.0f), m_tint, m_metallic);
    }

    float3 evaluate(float3 wo, float3 wi, float2 texcoord) {
        float3 diffuse_tint, specular_tint;
        evaluate_tints(wo, wi, texcoord, diffuse_tint, specular_tint);
        float3 diffuse = diffuse_tint * BSDFs::Lambert::evaluate();
        float ggx_alpha = BSDFs::GGX::alpha_from_roughness(m_roughness);
        float3 halfway = normalize(wo + wi);
        float3 specular = specular_tint * BSDFs::GGX::evaluate(ggx_alpha, wo, wi, halfway);
        return diffuse + specular;
    }

    // Apply the shading model to the IBL.
    // TODO Take the current LOD and pixel density into account before choosing sample LOD.
    //      See http://casual-effects.blogspot.dk/2011/08/plausible-environment-lighting-in-two.html 
    //      for how to derive the LOD level for cubemaps.
    float3 IBL(float3 wo, float3 normal, float2 texcoord) {
        float3 tint = m_tint;
        if (m_tint_texture_index != 0)
            tint *= color_tex.Sample(color_sampler, texcoord).rgb;

        float ggx_alpha = BSDFs::GGX::alpha_from_roughness(m_roughness);
        float3 wi = BSDFs::GGX::approx_off_specular_peak(ggx_alpha, wo, normal);

        float specularity = compute_specularity(m_specularity, m_metallic);
        float abs_cos_theta = abs(dot(wi, normal));
        float specular_rho = compute_specular_rho(specularity, abs_cos_theta, m_roughness);
        float3 specular_tint = lerp(float3(1.0f, 1.0f, 1.0f), tint, m_metallic) * specular_rho;
        float3 diffuse_tint = tint * (1.0f - specular_rho);

        float width, height, mip_count;
        environment_tex.GetDimensions(0, width, height, mip_count);

        float2 diffuse_tc = direction_to_latlong_texcoord(normal);
        float3 diffuse = diffuse_tint * environment_tex.SampleLevel(environment_sampler, diffuse_tc, mip_count).rgb;

        float2 specular_tc = direction_to_latlong_texcoord(wi);
        float3 specular = specular_tint * environment_tex.SampleLevel(environment_sampler, specular_tc, mip_count * m_roughness).rgb;

        return diffuse + specular;
    }
};

#endif // _DX11_RENDERER_SHADERS_DEFAULT_SHADING_H_