// OptiX renderer functions for the Oren-Nayar BSDF.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _OPTIXRENDERER_BSDFS_ORENNAYAR_H_
#define _OPTIXRENDERER_BSDFS_ORENNAYAR_H_

#include <OptiXRenderer/Distributions.h>
#include <OptiXRenderer/Types.h>

namespace OptiXRenderer {
namespace Shading {
namespace BSDFs {

//----------------------------------------------------------------------------
// Implementation of the energy-preserving Oren-Nayar refectance model.
// EON: A practical energy-preserving rough diffuse BRDF https://arxiv.org/pdf/2410.18026
//----------------------------------------------------------------------------
namespace OrenNayar {

using namespace optix;

__constant_all__ float constant1_FON = 0.5f - 2.0f / (3.0f * PIf);

__inline_all__ float E_FON_exact(float cos_theta, float roughness, float A, float B) {
    float Si = sqrt(1.0f - (cos_theta * cos_theta));
    float G = Si * (acos(cos_theta) - Si * cos_theta)
        + (2.0f / 3.0f) * ((Si / cos_theta) * (1.0f - (Si * Si * Si)) - Si);
    return A + B * G * RECIP_PIf;
}
__inline_all__ float E_FON_exact(float cos_theta, float roughness) {
    float A = 1.0f / (1.0f + constant1_FON * roughness); // FON A coeff.
    float B = roughness * A; // FON B coeff.
    return E_FON_exact(cos_theta, roughness, A, B);
}

__inline_all__ float E_FON_approx(float cos_theta, float roughness, float A, float B) {
    float mucomp = 1.0f - cos_theta;
    // Rewritten from Listing 1 to perform fewer multiplications.
    float GoverPi = 0.0f;
    for (float g : {0.0714429953f, -0.332181442f, 0.491881867f, 0.0571085289f})
        GoverPi = mucomp * (g + GoverPi);
    return A + B * GoverPi;
}

__inline_all__ float E_FON_approx(float cos_theta, float roughness) {
    float A = 1.0f / (1.0f + constant1_FON * roughness); // FON A coeff.
    float B = roughness * A; // FON B coeff.
    return E_FON_approx(cos_theta, roughness, A, B);
}

// Evaluate the energy-preserving Oren-Nayar diffuse BRDF.
// Based of Listing 1 in EON: A practical energy-preserving rough diffuse BRDF
// As explained on page 12 in the paper, we here choose to avoid the color shifting multi-scattering term.
// Instead we evaluate the BRDF as if rho=1 and then multiply by albedo after if tinting is required.
// This makes rho trivial to compute, as it is just the input albedo, and makes the BRDF color tweaking a linear operation.
__inline_all__ float evaluate(float roughness, float3 wo, float3 wi, bool exact = false) {
    const float constant2_FON = 2.0f / 3.0f - 28.0f / (15.0f * PIf);

    float cos_theta_i = wi.z; // input angle cos
    float cos_theta_o = wo.z; // output angle cos
    float s = dot(wi, wo) - cos_theta_i * cos_theta_o; // QON s term
    float s_over_t = s > 0.0f ? s / fmaxf(cos_theta_i, cos_theta_o) : s; // FON s/t
    float A = 1.0f / (1.0f + constant1_FON * roughness); // FON A coeff.
    float B = roughness * A; // FON B coeff.

    float f_single_scatter = RECIP_PIf * A * (1.0f + roughness * s_over_t); // single-scatter

    float EF_o = exact ? E_FON_exact(cos_theta_o, roughness, A, B) : // FON wo albedo (exact)
                         E_FON_approx(cos_theta_o, roughness, A, B); // FON wo albedo (approx)
    float EF_i = exact ? E_FON_exact(cos_theta_i, roughness, A, B) : // FON wi albedo (exact)
                         E_FON_approx(cos_theta_i, roughness, A, B); // FON wi albedo (approx)
    float average_EF = A * (1.0f + constant2_FON * roughness); // avg. albedo
    float multi_scatter_rho = average_EF / (1.0f - (1.0f - average_EF));
    float f_multi_scatter = (multi_scatter_rho * RECIP_PIf) * abs(1.0f - EF_o) * abs(1.0f - EF_i) // Replaced max(eps) from the paper with abs
        / fmaxf(1.0e-7f, 1.0f - average_EF); // multi-scatter lobe
    return f_single_scatter + f_multi_scatter;
}

__inline_all__ float3 evaluate(float3 albedo, float roughness, float3 wo, float3 wi, bool exact = false) {
    return albedo * evaluate(roughness, wo, wi, exact);
}

__inline_all__ PDF pdf(float roughness, float3 wo, float3 wi) {
    float cos_theta = wo.z;
    float uniform_probability = pow(roughness, 0.1f) * (0.162925f + cos_theta * (-0.372058f + (0.538233f - 0.290822f*cos_theta)*cos_theta));
    float cltc_probability = 1.0f - uniform_probability;
    float cltc_PDF = Distributions::OrenNayerCLTC::PDF(roughness, wo, wi);
    float uniform_PDF = Distributions::UniformHemisphere::PDF();
    return uniform_probability * uniform_PDF + cltc_probability * cltc_PDF;
}

__inline_all__ BSDFResponse evaluate_with_PDF(float3 albedo, float roughness, float3 wo, float3 wi, bool exact = false) {
    auto reflectance = evaluate(albedo, roughness, wo, wi, exact);
    auto PDF = pdf(roughness, wo, wi);
    return { reflectance, PDF };
}

__inline_all__ BSDFSample sample(float3 albedo, float roughness, float3 wo, float2 random_sample, bool exact = false) {
    float cos_theta = wo.z;
    float uniform_probability = pow(roughness, 0.1f) * (0.162925f + cos_theta * (-0.372058f + (0.538233f - 0.290822f*cos_theta)*cos_theta));
    float cltc_probability = 1.0f - uniform_probability;

    Distributions::DirectionalSample cosine_sample;
    float cltc_PDF;
    if (random_sample.x <= uniform_probability) {
        random_sample.x = random_sample.x / uniform_probability;
        cosine_sample = Distributions::UniformHemisphere::sample(random_sample); // sample wi from uniform lobe
        cltc_PDF = Distributions::OrenNayerCLTC::PDF(roughness, wo, cosine_sample.direction); // evaluate CLTC PDF at wi
    } else {
        random_sample.x = (random_sample.x - uniform_probability) / cltc_probability;
        cosine_sample = Distributions::OrenNayerCLTC::sample(roughness, wo, random_sample); // sample wi from CLTC lobe
        cltc_PDF = cosine_sample.PDF;
    }
    float uniform_PDF = Distributions::UniformHemisphere::PDF();
    cosine_sample.PDF = uniform_probability * uniform_PDF + cltc_probability * cltc_PDF; // MIS PDF of wi

    BSDFSample bsdf_sample;
    bsdf_sample.direction = cosine_sample.direction;
    bsdf_sample.PDF = cosine_sample.PDF;
    bsdf_sample.reflectance = evaluate(albedo, roughness, wo, bsdf_sample.direction, exact);
    return bsdf_sample;
}

} // NS OrenNayar
} // NS BSDFs
} // NS Shading
} // NS OptiXRenderer

#endif // _OPTIXRENDERER_BSDFS_ORENNAYAR_H_