// Diffuse BRDFs.
// ---------------------------------------------------------------------------
// Copyright (C) 2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _DX11_RENDERER_SHADERS_BSDFS_DIFFUSE_H_
#define _DX11_RENDERER_SHADERS_BSDFS_DIFFUSE_H_

#include "Utils.hlsl"

namespace BSDFs {

//-----------------------------------------------------------------------------
// Lambert.
//-----------------------------------------------------------------------------

namespace Lambert {

float evaluate() {
    return RECIP_PI;
}

} // NS Lambert

//-----------------------------------------------------------------------------
// Burley.
//-----------------------------------------------------------------------------

namespace Burley {

float evaluate(float roughness, float3 wo, float3 wi, float3 halfway) {
    float wi_dot_halfway = dot(wi, halfway);
    float fd90 = 0.5f + 2.0f * wi_dot_halfway * wi_dot_halfway * roughness;
    float fresnel_wo = schlick_fresnel(wo.z);
    float fresnel_wi = schlick_fresnel(wi.z);
    float normalizer = lerp(1.0f / 0.969371021f, 1.0f / 1.04337633f, roughness); // Burley isn't energy conserving, so we normalize by a 'good enough' constant here.
    return lerp(1.0f, fd90, fresnel_wo) * lerp(1.0f, fd90, fresnel_wi) * RECIP_PI * normalizer;
}

} // NS Burley

//-----------------------------------------------------------------------------
// Oren Nayar.
//-----------------------------------------------------------------------------

namespace OrenNayar {

float evaluate(float roughness, float3 wo, float3 wi) {
    float2 cos_theta = float2(abs(wi.z), abs(wo.z));
    float sin_theta_sqrd = (1.0f - cos_theta.x * cos_theta.x) * (1.0f - cos_theta.y * cos_theta.y);
    float sin_theta = sqrt(max(0.0f, sin_theta_sqrd));

    const float3 normal = float3(0.0f, 0.0f, 1.0f);
    float3 light_plane = normalize(wi - cos_theta.x * normal);
    float3 view_plane = normalize(wo - cos_theta.y * normal);
    float cos_phi = clamp(dot(light_plane, view_plane), 0.0f, 1.0f);

    float sigma_sqrd = roughness * roughness;
    float A = 1.0f - (sigma_sqrd / (2.0f * sigma_sqrd + 0.66f));
    float B = 0.45f * sigma_sqrd / (sigma_sqrd + 0.09f);

    return (A + B * cos_phi * sin_theta / max(cos_theta.x, cos_theta.y)) * RECIP_PI;
}

// As seen in Unreal Engine and based on
// Beyond a Simple Physically Based Blinn-Phong Model in Real-Time, Gotanda 2012.
float evaluate_approx(float roughness, float wo_dot_n, float wi_dot_n, float wo_dot_h) {
    float a = roughness * roughness;
    float s = a;// / ( 1.29 + 0.5 * a );
    float s2 = s * s;
    float VoL = 2 * wo_dot_h * wo_dot_h - 1; // double angle identity
    float Cosri = VoL - wo_dot_n * wo_dot_n;
    float C1 = 1 - 0.5 * s2 / (s2 + 0.33);
    float C2 = 0.45 * s2 / (s2 + 0.09) * Cosri * (Cosri >= 0 ? rcp(max(wi_dot_n, wi_dot_n)) : 1);
    return (C1 + C2) * (1 + roughness * 0.5) / PI;
}

} // NS OrenNayar

} // NS BSDFs

#endif // _DX11_RENDERER_SHADERS_BSDFS_DIFFUSE_H_