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
// Samplers.
//-----------------------------------------------------------------------------

Texture2D colorTex : register(t0);
SamplerState colorSampler : register(s0);

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
        float3 specular_tint = lerp(float3(1.0f, 1.0f, 1.0f), m_tint, m_metallic);
        float ggx_alpha = BSDFs::GGX::alpha_from_roughness(m_roughness);
        float3 specular = specular_tint * BSDFs::GGX::evaluate(ggx_alpha, wo, wi, halfway);
        float3 diffuse = m_tint * BSDFs::Burley::evaluate(m_roughness, wo, wi, halfway);
        return lerp(diffuse, specular, fresnel);
    }

};

