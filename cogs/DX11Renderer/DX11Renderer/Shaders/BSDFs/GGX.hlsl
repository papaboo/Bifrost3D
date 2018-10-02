// GGX BRDF.
// ---------------------------------------------------------------------------
// Copyright (C) 2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _DX11_RENDERER_SHADERS_BSDFS_SPECULAR_H_
#define _DX11_RENDERER_SHADERS_BSDFS_SPECULAR_H_

#include "Utils.hlsl"

namespace BSDFs {
namespace GGX {

    float alpha_from_roughness(float roughness) {
        return max(0.00000000001f, roughness * roughness);
    }

    float D(float alpha, float abs_cos_theta) {
        float alpha_sqrd = alpha * alpha;
        float cos_theta_sqrd = abs_cos_theta * abs_cos_theta;
        float tan_theta_sqrd = max(1.0f - cos_theta_sqrd, 0.0f) / cos_theta_sqrd;
        float cos_theta_cubed = cos_theta_sqrd * cos_theta_sqrd;
        float foo = alpha_sqrd + tan_theta_sqrd; // No idea what to call this.
        return alpha_sqrd / (PI * cos_theta_cubed * foo * foo);
    }

    // Samples the GGX distribution and returns a direction and PDF.
    float4 sample(float alpha, float2 random_sample) {
        float phi = random_sample.y * (2.0f * PI);

        float tan_theta_sqrd = alpha * alpha * random_sample.x / (1.0f - random_sample.x);
        float cos_theta = 1.0f / sqrt(1.0f + tan_theta_sqrd);

        float r = sqrt(max(1.0f - cos_theta * cos_theta, 0.0f));

        float4 res;
        res.xyz = float3(cos(phi) * r, sin(phi) * r, cos_theta);
        res.w = D(alpha, cos_theta) * cos_theta; // We have to be able to inline this to reuse some temporaries.
        return res;
    }

    float PDF(float alpha, float abs_cos_theta) {
        return D(alpha, abs_cos_theta) * abs_cos_theta;
    }

    // Understanding the Masking - Shadowing Function in Microfacet - Based BRDFs, Heitz 14, equation 72.
    float height_correlated_smith_delta(float alpha, float3 w) {
        float cos_theta_sqrd = w.z * w.z;
        float tan_theta_sqrd = max(1.0f - cos_theta_sqrd, 0.0f) / cos_theta_sqrd;
        float recip_a_sqrd = alpha * alpha * tan_theta_sqrd;
        return 0.5 * (-1.0f + sqrt(1.0f + recip_a_sqrd));
    }

    // Height correlated smith geometric term.
    // Understanding the Masking-Shadowing Function in Microfacet-Based BRDFs, Heitz 14, equation 99. 
    float height_correlated_smith_G(float alpha, float3 wo, float3 wi, float3 halfway) {
        float numerator = heaviside(dot(wo, halfway)) * heaviside(dot(wi, halfway));
        return numerator / (1.0f + height_correlated_smith_delta(alpha, wo) + height_correlated_smith_delta(alpha, wi));
    }

    float evaluate(float alpha, float3 wo, float3 wi, float3 halfway) {
        float g = height_correlated_smith_G(alpha, wo, wi, halfway);
        float d = D(alpha, halfway.z);
        float f = 1.0f; // No fresnel.
        return (d * f * g) / (4.0f * wo.z * wi.z);
    }

    float3 approx_off_specular_peak(float alpha, float3 wo) {
        float3 reflection = float3(-wo.x, -wo.y, wo.z);
        // reflection = lerp(float3(0, 0, 1), reflection, (1 - alpha) * (sqrt(1 - alpha) + alpha)); // UE4 implementation
        reflection = lerp(reflection, float3(0, 0, 1), alpha);
        return normalize(reflection);
    }

    float3 approx_off_specular_peak(float alpha, float3 wo, float3 normal) {
        float3 reflection = reflect(-wo, normal);
        // reflection = lerp(normal, reflection, (1 - alpha) * (sqrt(1 - alpha) + alpha)); // UE4 implementation
        reflection = lerp(reflection, normal, alpha);
        return normalize(reflection);
    }

} // GGX
} // BSDFs

#endif // _DX11_RENDERER_SHADERS_BSDFS_SPECULAR_H_