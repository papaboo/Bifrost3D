// GGX BRDF.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _DX11_RENDERER_SHADERS_BSDFS_SPECULAR_H_
#define _DX11_RENDERER_SHADERS_BSDFS_SPECULAR_H_

#include "Utils.hlsl"

namespace BSDFs {
namespace GGX {

    float alpha_from_roughness(float roughness) {
        return max(0.00000000001f, roughness * roughness);
    }

    // Sampling the GGX Distribution of Visible Normals, equation 1.
    float D(float alpha_x, float alpha_y, float3 halfway) {
        float m = pow2(halfway.x / alpha_x) + pow2(halfway.y / alpha_y) + pow2(halfway.z);
        return 1 / (PI * alpha_x * alpha_y * pow2(m));
    }

    float D(float alpha, float abs_cos_theta) {
        float alpha_sqrd = alpha * alpha;
        float cos_theta_sqrd = abs_cos_theta * abs_cos_theta;
        float tan_theta_sqrd = max(1.0f - cos_theta_sqrd, 0.0f) / cos_theta_sqrd;
        float foo = alpha_sqrd + tan_theta_sqrd; // No idea what to call this.
        return alpha_sqrd / (PI * pow2(cos_theta_sqrd * foo));
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

    // Sampling the GGX Distribution of Visible Normals, equation 2.
    float lambda(float alpha_x, float alpha_y, float3 w) {
        return 0.5f * (-1 + sqrt(1 + (pow2(alpha_x * w.x) + pow2(alpha_y * w.y)) / pow2(w.z)));
    }
    float lambda(float alpha, float3 w) { return lambda(alpha, alpha, w); }

    // Height correlated smith geometric term.
    // Understanding the Masking-Shadowing Function in Microfacet-Based BRDFs, Heitz 14, equation 99. 
    float G2(float alpha, float3 wo, float3 wi, float3 halfway) {
        float numerator = 1.0; // heaviside(dot(wo, halfway)) * heaviside(dot(wi, halfway)); // Should always be one.
        return numerator / (1.0f + lambda(alpha, wo) + lambda(alpha, wi));
    }

    float3 evaluate(float alpha, float3 specularity, float3 wo, float3 wi) {
        float3 halfway = normalize(wo + wi);
        float g = G2(alpha, wo, wi, halfway);
        float d = D(alpha, halfway.z);
        float3 f = schlick_fresnel(specularity, dot(wo, halfway));
        return f * (d * g / (4.0f * wo.z * wi.z));
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