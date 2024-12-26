// Diffuse BRDFs.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
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

float E_FON_approx(float cos_theta, float roughness, float A, float B) {
    float mucomp = 1.0 - cos_theta;
    // Rewritten from Listing 1 to perform fewer multiplications.
    float GoverPi = mucomp * (-0.332181442 + mucomp * 0.0714429953);
    GoverPi = mucomp * (0.491881867 + GoverPi);
    GoverPi = mucomp * (0.0571085289 + GoverPi);
    return A + B * GoverPi;
}

// Evaluate the energy-preserving Oren-Nayar diffuse BRDF.
// Based of Listing 1 in EON: A practical energy-preserving rough diffuse BRDF
// As explained on page 12 in the paper, we here choose to avoid the color shifting multi-scattering term.
// Instead we evaluate the BRDF as if rho=1 and then multiply by albedo after if tinting is required.
// This makes rho trivial to compute, as it is just the input albedo, and makes the BRDF color tweaking a linear operation.
float evaluate(float roughness, float3 wo, float3 wi) {
    const float constant1_FON = 0.5 - 2.0 / (3.0 * PI);
    const float constant2_FON = 2.0 / 3.0 - 28.0 / (15.0 * PI);

    float cos_theta_i = wi.z; // input angle cos
    float cos_theta_o = wo.z; // output angle cos
    float s = dot(wi, wo) - cos_theta_i * cos_theta_o; // QON s term
    float s_over_t = s > 0.0 ? s / max(cos_theta_i, cos_theta_o) : s; // FON s/t
    float A = 1.0 / (1.0 + constant1_FON * roughness); // FON A coeff.
    float B = roughness * A; // FON B coeff.

    float f_single_scatter = RECIP_PI * A * (1.0 + roughness * s_over_t); // single-scatter

    // Always use the approximate E term in DX11
    float EF_o = E_FON_approx(cos_theta_o, roughness, A, B); // FON wo albedo (approx)
    float EF_i = E_FON_approx(cos_theta_i, roughness, A, B); // FON wi albedo (approx)
    float average_EF = A * (1.0 + constant2_FON * roughness); // avg. albedo
    float multi_scatter_rho = average_EF / (1.0 - (1.0 - average_EF));
    float f_multi_scatter = (multi_scatter_rho * RECIP_PI) * abs(1.0 - EF_o) * abs(1.0 - EF_i) // Replaced max(eps) from the paper with abs
        / max(1.0e-7, 1.0 - average_EF); // multi-scatter lobe
    return f_single_scatter + f_multi_scatter;
}

} // NS OrenNayar

} // NS BSDFs

#endif // _DX11_RENDERER_SHADERS_BSDFS_DIFFUSE_H_