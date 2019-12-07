// OptiX renderer functions for microfacet models.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _OPTIXRENDERER_BSDFS_MICROFACET_H_
#define _OPTIXRENDERER_BSDFS_MICROFACET_H_

#include <OptiXRenderer/Distributions.h>
#include <OptiXRenderer/Types.h>
#include <OptiXRenderer/Utils.h>

namespace OptiXRenderer {
namespace Shading {
namespace BSDFs {

//----------------------------------------------------------------------------
// Implementation of GGX's D and G term for the Microfacet model.
// Sources:
// * Walter et al 07.
// * Understanding the Masking-Shadowing Function in Microfacet-Based BRDFs, Heitz 14.
// * https://github.com/tunabrain/tungsten/blob/master/src/core/bsdfs/Microfacet.hpp
//----------------------------------------------------------------------------
namespace GGX {

using namespace optix;

__inline_all__ float alpha_from_roughness(float roughness) {
    return fmaxf(0.00000000001f, roughness * roughness);
}

__inline_all__ float roughness_from_alpha(float alpha) {
    return sqrt(alpha);
}

// Understanding the Masking-Shadowing Function in Microfacet-Based BRDFs, Heitz 14.
// Height correlated smith geometric term. Equation 99. 
__inline_all__ float height_correlated_smith_G(float alpha, float3 wo, float3 wi) {
#if _DEBUG
    float3 halfway = normalize(wo + wi);
    float heavisided = heaviside(dot(wo, halfway)) * heaviside(dot(wi, halfway));
    if (heavisided != 1.0f)
        THROW(OPTIX_SHADING_WRONG_HEMISPHERE_EXCEPTION);
#endif
    return 1.0f / (1.0f + Distributions::GGX_VNDF::lambda(alpha, wo) + Distributions::GGX_VNDF::lambda(alpha, wi));
}

//----------------------------------------------------------------------------
// GGX BSDF, Walter et al 07.
// Here we have seperated the BRDF from the BTDF. 
// This makes sampling slightly less performant, but allows for a full 
// overview of the individual components, which can then later be composed 
// in a new material/BSDF.
//----------------------------------------------------------------------------

//----------------------------------------------------------------------------
// GGX BRDF, Walter et al 07.
//----------------------------------------------------------------------------

__inline_all__ float evaluate(float alpha, float specularity, float3 wo, float3 wi) {
    float3 halfway = normalize(wo + wi);
    float G = height_correlated_smith_G(alpha, wo, wi);
    float D = Distributions::GGX_VNDF::D(alpha, halfway);
    float F = schlick_fresnel(specularity, dot(wo, halfway));
    return (D * F * G) / (4.0f * wo.z * wi.z);
}

__inline_all__ float3 evaluate(float alpha, float3 specularity, float3 wo, float3 wi, float3 halfway) {
    float G = height_correlated_smith_G(alpha, wo, wi);
    float D = Distributions::GGX_VNDF::D(alpha, halfway);
    float3 F = schlick_fresnel(specularity, dot(wo, halfway));
    return F * (D * G / (4.0f * wo.z * wi.z));
}

__inline_all__ float3 evaluate(float alpha, float3 specularity, float3 wo, float3 wi) {
    float3 halfway = normalize(wo + wi);
    return evaluate(alpha, specularity, wo, wi, halfway);
}

__inline_all__ float PDF(float alpha, float3 wo, float3 halfway) {
#if _DEBUG
    if (dot(wo, halfway) < 0.0f || halfway.z < 0.0f)
        THROW(OPTIX_SHADING_WRONG_HEMISPHERE_EXCEPTION);
#endif

    return Distributions::GGX_VNDF::PDF(alpha, wo, halfway) / (4.0f * dot(wo, halfway));
}

__inline_all__ BSDFResponse evaluate_with_PDF(float alpha, float3 specularity, float3 wo, float3 wi, float3 halfway) {
    float lambda_wo = Distributions::GGX_VNDF::lambda(alpha, wo);
    float lambda_wi = Distributions::GGX_VNDF::lambda(alpha, wi);

    float quater_D = 0.25f * Distributions::GGX_VNDF::D(alpha, halfway);

    float3 F = schlick_fresnel(specularity, dot(wo, halfway));

    float recip_G2 = 1.0f + lambda_wo + lambda_wi; // reciprocal height_correlated_smith_G
    float3 reflectance = F * (quater_D / (recip_G2 * wo.z * wi.z));

    BSDFResponse res;
    res.reflectance = reflectance;
    float recip_G1 = 1.0f + lambda_wo;
    res.PDF = quater_D / (recip_G1 * wo.z);
    return res;
}

__inline_all__ BSDFResponse evaluate_with_PDF(float alpha, float3 specularity, float3 wo, float3 wi) {
    float3 halfway = normalize(wo + wi);
    return evaluate_with_PDF(alpha, specularity, wo, wi, halfway);
}

__inline_all__ BSDFResponse evaluate_with_PDF(float alpha, float specularity, float3 wo, float3 wi) {
    return evaluate_with_PDF(alpha, make_float3(specularity), wo, wi);
}

__inline_all__ float3 approx_off_specular_peak(float alpha, float3 wo) {
    float3 reflection = make_float3(-wo.x, -wo.y, wo.z);
    // reflection = lerp(make_float3(0, 0, 1), reflection, (1 - alpha) * (sqrt(1 - alpha) + alpha)); // UE4 implementation
    reflection = lerp(reflection, make_float3(0, 0, 1), alpha);
    return normalize(reflection);
}

__inline_all__ BSDFSample sample(float alpha, float3 specularity, float3 wo, float2 random_sample) {
    BSDFSample bsdf_sample;

    float3 halfway = Distributions::GGX_VNDF::sample_halfway(alpha, wo, random_sample);
    bsdf_sample.direction = reflect(-wo, halfway);

    BSDFResponse response = evaluate_with_PDF(alpha, specularity, wo, bsdf_sample.direction, halfway);
    bsdf_sample.PDF = response.PDF;
    bsdf_sample.reflectance = response.reflectance;

    bool discardSample = bsdf_sample.PDF < 0.00001f || bsdf_sample.direction.z < 0.00001f; // Discard samples if the pdf is too low (precision issues) or if the new direction points into the surface (energy loss).
    return discardSample ? BSDFSample::none() : bsdf_sample;
}
__inline_all__ BSDFSample sample(float alpha, float specularity, float3 wo, float2 random_sample) { return sample(alpha, make_float3(specularity), wo, random_sample); }

} // NS GGX

} // NS BSDFs
} // NS Shading
} // NS OptiXRenderer

#endif // _OPTIXRENDERER_BSDFS_MICROFACET_H_