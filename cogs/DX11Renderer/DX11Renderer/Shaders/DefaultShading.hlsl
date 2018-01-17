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

struct MaterialParams {
    float3 m_tint;
    unsigned int m_tint_texture_index;
    float m_roughness;
    float m_specularity;
    float m_metallic;
    float m_coverage;
    unsigned int m_coverage_texture_index;
    int3 __padding;

    float coverage(float2 texcoord) {
        float coverage = m_coverage;
        if (m_coverage_texture_index != 0)
            coverage *= coverage_tex.Sample(coverage_sampler, texcoord).a;
        return coverage;
    }
};

//-----------------------------------------------------------------------------
// Default shading.
//-----------------------------------------------------------------------------
// TODO Make immutable by only exposing through accesors.
struct DefaultShading {
    float3 m_diffuse_tint;
    float m_coverage;
    float3 m_specular_tint; // Without fresnel
    float m_specularity;
    float3 m_off_specular_peak;
    float m_roughness;

    // --------------------------------------------------------------------------------------------
    // Helpers
    // --------------------------------------------------------------------------------------------
    static float compute_specularity(float specularity, float metalness) {
        float dielectric_specularity = specularity * 0.08f; // See Physically-Based Shading at Disney bottom of page 8.
        float metal_specularity = specularity * 0.2f + 0.6f;
        return lerp(dielectric_specularity, metal_specularity, metalness);
    }

    static float compute_specular_rho(float specularity, float abs_cos_theta, float roughness) {
        float base_specular_rho = ggx_with_fresnel_rho_tex.Sample(precomputation2D_sampler, float2(abs_cos_theta, roughness)).r;
        float full_specular_rho = 1.0f; // TODO This is wrong. GGX doesn't have a rho of one. Try to use the actual GGX rho instead.
        return lerp(base_specular_rho, full_specular_rho, specularity);
    }

    // --------------------------------------------------------------------------------------------
    // Factory function constructing a shading from a constant buffer.
    // --------------------------------------------------------------------------------------------
    static DefaultShading from_constants(uniform MaterialParams material_params, float3 wo, float2 texcoord) {
        DefaultShading shading;

        // Coverage
        shading.m_coverage = material_params.m_coverage;
        if (material_params.m_coverage_texture_index != 0)
            shading.m_coverage *= coverage_tex.Sample(coverage_sampler, texcoord).a;

        // Roughness
        shading.m_roughness = material_params.m_roughness;

        // Metallic
        float metallic = material_params.m_metallic;

        // Specularity
        shading.m_specularity = compute_specularity(material_params.m_specularity, metallic);

        // Diffuse and specular tint
        float3 tint = material_params.m_tint;
        if (material_params.m_tint_texture_index != 0)
            tint *= color_tex.Sample(color_sampler, texcoord).rgb;
        float abs_cos_theta = abs(wo.z);
        float specular_rho = compute_specular_rho(shading.m_specularity, abs_cos_theta, shading.m_roughness);
        shading.m_diffuse_tint = tint * (1.0f - specular_rho);
        shading.m_specular_tint = lerp(float3(1.0f, 1.0f, 1.0f), tint, metallic);

        // Off specular peak
        float ggx_alpha = BSDFs::GGX::alpha_from_roughness(shading.m_roughness);
        shading.m_off_specular_peak = BSDFs::GGX::approx_off_specular_peak(ggx_alpha, wo);

        return shading;
    }

    // --------------------------------------------------------------------------------------------
    // Getters.
    // --------------------------------------------------------------------------------------------
    float coverage() { return m_coverage; }
    float roughness() { return m_roughness; }

    // --------------------------------------------------------------------------------------------
    // Evaluate the material.
    // --------------------------------------------------------------------------------------------
    bool evaluate_tints(float3 wo, float3 wi, out float3 diffuse_tint, out float3 specular_tint) {
        bool is_same_hemisphere = wi.z * wo.z >= 0.00000001f;
        if (!is_same_hemisphere) {
            diffuse_tint = specular_tint = float3(0.0f, 0.0f, 0.0f);
            return false;
        }

        // Flip directions if on the backside of the material.
        if (wo.z < 0.0f) {
            wi.z = -wi.z;
            wo.z = -wo.z;
        }

        diffuse_tint = m_diffuse_tint;

        float3 halfway = normalize(wo + wi);
        float fresnel = schlick_fresnel(m_specularity, dot(wo, halfway));
        specular_tint = fresnel * m_specular_tint;

        return true;
    }

    float3 evaluate(float3 wo, float3 wi) {
        float3 diffuse_tint, specular_tint;
        if (!evaluate_tints(wo, wi, diffuse_tint, specular_tint))
            return float3(0, 0, 0);

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
    float3 IBL(float3 wo, float3 normal) {
        float width, height, mip_count;
        environment_tex.GetDimensions(0, width, height, mip_count);

        float2 diffuse_tc = direction_to_latlong_texcoord(normal);
        float3 diffuse = m_diffuse_tint * environment_tex.SampleLevel(environment_sampler, diffuse_tc, mip_count).rgb;

        float ggx_alpha = BSDFs::GGX::alpha_from_roughness(m_roughness);
        float3 wi = BSDFs::GGX::approx_off_specular_peak(ggx_alpha, wo, normal);
        float abs_cos_theta = abs(dot(wo, normal));
        float specular_rho = compute_specular_rho(m_specularity, abs_cos_theta, m_roughness);
        float3 specular_tint = m_specular_tint * specular_rho;
        float2 specular_tc = direction_to_latlong_texcoord(wi);
        float3 specular = specular_tint * environment_tex.SampleLevel(environment_sampler, specular_tc, mip_count * m_roughness).rgb;

        return diffuse + specular;
    }

};

#endif // _DX11_RENDERER_SHADERS_DEFAULT_SHADING_H_