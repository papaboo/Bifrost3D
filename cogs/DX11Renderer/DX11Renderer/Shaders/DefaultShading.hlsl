// Default shading.
// ---------------------------------------------------------------------------
// Copyright (C) 2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#include "BSDFs/Diffuse.hlsl"
#include "BSDFs/GGX.hlsl"
#include "Utils.hlsl"

//-----------------------------------------------------------------------------
// Textures.
//-----------------------------------------------------------------------------

Texture2D environmentTex : register(t0);
SamplerState environmentSampler : register(s0);

Texture2D colorTex : register(t1);
SamplerState colorSampler : register(s1);

Texture2D coverageTex : register(t2);
SamplerState coverageSampler : register(s2);

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

    float coverage(float2 texcoord) {
        float c = m_coverage;
        if (m_coverage_texture_index != 0)
            c *= coverageTex.Sample(coverageSampler, texcoord).a;
        return c;
    }

    float3 evaluate(float3 wo, float3 wi, float2 texcoord) {
        bool is_same_hemisphere = wi.z * wo.z >= 0.00000001f;
        if (!is_same_hemisphere)
            return float3(0.0f, 0.0f, 0.0f);

        // Flip directions if on the backside of the material.
        if (wo.z < 0.0f) {
            wi.z = -wi.z;
            wo.z = -wo.z;
        }

        float3 tint = m_tint;
        if (m_tint_texture_index != 0)
            tint *= colorTex.Sample(colorSampler, texcoord).rgb;
        float3 halfway = normalize(wo + wi);
        float specularity = lerp(m_specularity, 1.0f, m_metallic);
        float fresnel = schlick_fresnel(specularity, dot(wo, halfway));
        float3 specular_tint = lerp(float3(1.0f, 1.0f, 1.0f), tint, m_metallic);
        float ggx_alpha = BSDFs::GGX::alpha_from_roughness(m_roughness);
        float3 specular = specular_tint * BSDFs::GGX::evaluate(ggx_alpha, wo, wi, halfway);
        float3 diffuse = tint * BSDFs::Lambert::evaluate();
        return lerp(diffuse, specular, fresnel);
    }

    // This is a horrible hack, just to get some approximate IBL in.
    // In order to do this right, the material model needs to be fleshed out and rho for GGX needs to be approximated.
    // TODO Take the current LOD and pixel density into account before choosing sample LOD.
    // See http://casual-effects.blogspot.dk/2011/08/plausible-environment-lighting-in-two.html 
    // for how to derive the LOD level for cubemaps.
    float3 IBL(float3 normal, float3 wi, float2 texcoord) {
        float3 tint = m_tint;
        if (m_tint_texture_index != 0)
            tint *= colorTex.Sample(colorSampler, texcoord).rgb;

        float abs_cos_theta = abs(dot(wi, normal));

        float specularity = lerp(m_specularity, 1.0f, m_metallic);
        float fresnel = schlick_fresnel(specularity, pow(abs_cos_theta, 0.25f)); // Ugh, that fresnel approximation again.

        float width, height, mip_count;
        environmentTex.GetDimensions(0, width, height, mip_count);

        float2 diffuse_tc = direction_to_latlong_texcoord(normal);
        float3 diffuse = tint * environmentTex.SampleLevel(environmentSampler, diffuse_tc, mip_count - 3).rgb;

        float2 specular_tc = direction_to_latlong_texcoord(wi);
        float3 specular = tint * environmentTex.SampleLevel(environmentSampler, specular_tc, (mip_count - 3) * m_roughness * m_roughness).rgb;

        return lerp(diffuse, specular, fresnel);
    }
};

