// GGX BRDF.
// ---------------------------------------------------------------------------
// Copyright (C) 2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

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

    // Understanding the Masking - Shadowing Function in Microfacet - Based BRDFs, Heitz 14, equation 72.
    float height_correlated_smith_delta(float alpha, float3 w, float3 halfway) {
        float cos_theta_sqrd = w.z * w.z;
        float tan_theta_sqrd = max(1.0f - cos_theta_sqrd, 0.0f) / cos_theta_sqrd;
        float a_sqrd = 1.0f / (alpha * alpha * tan_theta_sqrd);
        return (-1.0f + sqrt(1.0f + 1.0f / a_sqrd)) / 2.0f;
    }

    // Height correlated smith geometric term.
    // Understanding the Masking-Shadowing Function in Microfacet-Based BRDFs, Heitz 14, equation 99. 
    float height_correlated_smith_G(float alpha, float3 wo, float3 wi, float3 halfway) {
        float numerator = heaviside(dot(wo, halfway)) * heaviside(dot(wi, halfway));
        return numerator / (1.0f + height_correlated_smith_delta(alpha, wo, halfway) + height_correlated_smith_delta(alpha, wi, halfway));
    }

    float evaluate(float alpha, float3 wo, float3 wi, float3 halfway) {
        float g = height_correlated_smith_G(alpha, wo, wi, halfway);
        float d = D(alpha, halfway.z);
        float f = 1.0f; // No fresnel.
        return (d * f * g) / (4.0f * wo.z * wi.z);
    }
}
}
