// OptiX renderer functions for microfacet models.
// ---------------------------------------------------------------------------
// Copyright (C) 2015-2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
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
// Separable smith geometric term. Equation 98. 
__inline_all__ float separable_smith_G(float alpha, const float3& wo, const float3& wi, const float3& halfway) {
#if _DEBUG
    float heavisided = heaviside(dot(wo, halfway)) * heaviside(dot(wi, halfway));
    if (heavisided != 1.0f)
        THROW(OPTIX_GGX_WRONG_HEMISPHERE_EXCEPTION);
#endif
    return 1.0f / ((1.0f + Distributions::VNDF_GGX::lambda(alpha, wo.z) * (1.0f + Distributions::VNDF_GGX::lambda(alpha, wi.z))));
}

// Understanding the Masking-Shadowing Function in Microfacet-Based BRDFs, Heitz 14.
// Height correlated smith geometric term. Equation 99. 
__inline_all__ float height_correlated_smith_G(float alpha, const float3& wo, const float3& wi, const float3& halfway) {
#if _DEBUG
    float heavisided = heaviside(dot(wo, halfway)) * heaviside(dot(wi, halfway));
    if (heavisided != 1.0f)
        THROW(OPTIX_GGX_WRONG_HEMISPHERE_EXCEPTION);
#endif
    return 1.0f / (1.0f + Distributions::VNDF_GGX::lambda(alpha, wo.z) + Distributions::VNDF_GGX::lambda(alpha, wi.z));
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

__inline_all__ float evaluate(float alpha, const float3& wo, const float3& wi, const float3& halfway) {
    float G = height_correlated_smith_G(alpha, wo, wi, halfway);
    float D = Distributions::GGX::D(alpha, halfway.z);
    float F = 1.0f; // No fresnel.
    return (D * F * G) / (4.0f * wo.z * wi.z);
}

__inline_all__ float3 evaluate(const float3& tint, float alpha, const float3& wo, const float3& wi, const float3& halfway) {
    return tint * evaluate(alpha, wo, wi, halfway);
}

__inline_all__ float PDF(float alpha, const float3& wo, const float3& wi, const float3& halfway) {
#if _DEBUG
    if (dot(wo, halfway) < 0.0f || halfway.z < 0.0f)
        THROW(OPTIX_GGX_WRONG_HEMISPHERE_EXCEPTION);
#endif

    return Distributions::GGX::PDF(alpha, halfway.z) / (4.0f * dot(wo, halfway));
}

__inline_all__ BSDFResponse evaluate_with_PDF(const float3& tint, float alpha, const float3& wo, const float3& wi, const float3& halfway) {
    float G = height_correlated_smith_G(alpha, wo, wi, halfway);
    float D = Distributions::GGX::D(alpha, halfway.z);
    float F = 1.0f; // No fresnel.
    float f = (D * F * G) / (4.0f * wo.z * wi.z);
    BSDFResponse res;
    res.weight = tint * f;
    res.PDF = (D * halfway.z) / (4.0f * dot(wo, halfway));
    return res;
}

__inline_all__ float3 approx_off_specular_peak(float alpha, const float3& wo) {
    float3 reflection = make_float3(-wo.x, -wo.y, wo.z);
    // reflection = lerp(make_float3(0, 0, 1), reflection, (1 - alpha) * (sqrt(1 - alpha) + alpha)); // UE4 implementation
    reflection = lerp(reflection, make_float3(0, 0, 1), alpha);
    return normalize(reflection);
}

__inline_all__ BSDFSample sample(const float3& tint, float alpha, const float3& wo, float2 random_sample) {

    BSDFSample bsdf_sample;

    Distributions::DirectionalSample halfway_sample = Distributions::GGX::sample(alpha, random_sample);
    bsdf_sample.direction = reflect(-wo, halfway_sample.direction);
    bool discardSample = halfway_sample.PDF < 0.00001f || bsdf_sample.direction.z < 0.00001f; // Discard samples if the pdf is too low (precision issues) or if the new direction points into the surface (energy loss).
    if (discardSample) return BSDFSample::none();

    bsdf_sample.PDF = halfway_sample.PDF / (4.0f * dot(wo, halfway_sample.direction));

    { // Evaluate using the already computed sample PDF, which is D * cos_theta.
        const float3& wi = bsdf_sample.direction; // alias for readability.
        float G = height_correlated_smith_G(alpha, wo, wi, halfway_sample.direction);
        float D = halfway_sample.PDF / halfway_sample.direction.z;
        float F = 1.0f;
        bsdf_sample.weight = tint * ((D * F * G) / (4.0f * wo.z * wi.z));
    }
    return bsdf_sample;
}

} // NS GGX

namespace GGXWithVNDF {

using namespace optix;

__inline_all__ float alpha_from_roughness(float roughness) {
    return GGX::alpha_from_roughness(roughness);
}

__inline_all__ float evaluate(float alpha, const float3& wo, const float3& wi, const float3& halfway) {
    return GGX::evaluate(alpha, wo, wi, halfway);
}

__inline_all__ float3 evaluate(const float3& tint, float alpha, const float3& wo, const float3& wi, const float3& halfway) {
    return tint * evaluate(alpha, wo, wi, halfway);
}

__inline_all__ float PDF(float alpha, const float3& wo, const float3& halfway) {
#if _DEBUG
    if (dot(wo, halfway) < 0.0f || halfway.z < 0.0f)
        THROW(OPTIX_GGX_WRONG_HEMISPHERE_EXCEPTION);
#endif

    return Distributions::VNDF_GGX::PDF(alpha, wo, halfway) / (4.0f * dot(wo, halfway));
}

__inline_all__ BSDFResponse evaluate_with_PDF(const float3& tint, float alpha, const float3& wo, const float3& wi, const float3& halfway) {
    float lambda_wo = Distributions::VNDF_GGX::lambda(alpha, wo.z);
    float lambda_wi = Distributions::VNDF_GGX::lambda(alpha, wi.z);

    float G1 = 1.0f / (1.0f + lambda_wo);
    float G2 = 1.0f / (1.0f + lambda_wo + lambda_wi); // height_correlated_smith_G

    float D_over_4 = Distributions::GGX::D(alpha, halfway.z) / 4.0f;

    float F = 1.0f; // No fresnel.
    float f = (D_over_4 * F * G2) / (wo.z * wi.z);

    BSDFResponse res;
    res.weight = tint * f;
    res.PDF = G1 * D_over_4 / wo.z;
    return res;
}

__inline_all__ BSDFSample sample(const float3& tint, float alpha, const float3& wo, float2 random_sample) {
    BSDFSample bsdf_sample;

    float3 halfway = Distributions::VNDF_GGX::sample_halfway(alpha, wo, random_sample);
    bsdf_sample.direction = reflect(-wo, halfway);

    BSDFResponse response = evaluate_with_PDF(tint, alpha, wo, bsdf_sample.direction, halfway);
    bsdf_sample.PDF = response.PDF;
    bsdf_sample.weight = response.weight;

    bool discardSample = bsdf_sample.PDF < 0.00001f || bsdf_sample.direction.z < 0.00001f; // Discard samples if the pdf is too low (precision issues) or if the new direction points into the surface (energy loss).
    return discardSample ? BSDFSample::none() : bsdf_sample;
}

} // NS GGXWithVNDF

} // NS BSDFs
} // NS Shading
} // NS OptiXRenderer

#endif // _OPTIXRENDERER_BSDFS_MICROFACET_H_