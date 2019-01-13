// OptiX renderer functions for microfacet models.
// ---------------------------------------------------------------------------
// Copyright (C) 2015-2016, Bifrost. See AUTHORS.txt for authors
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

// Smith G1 term. Equation 34 in Walter et al 07.
__inline_all__ float G1(float alpha, const float3& w, const float3& halfway) {
#if _DEBUG
    // Check if the w vector projected onto the halfway vector is in the same hemisphere as the halfway vector and the normal, otherwise something went horribly horribly wrong and we're leaking light.
    if (dot(w, halfway) * w.z <= 0.0f)
#if GPU_DEVICE
        rtThrow(OPTIX_GGX_WRONG_HEMISPHERE_EXCEPTION);
#else
        throw "OPTIX_GGX_WRONG_HEMISPHERE_EXCEPTION";
#endif
#endif

    float alpha_sqrd = alpha * alpha;
    float cos_theta_sqrd = w.z * w.z;
    float tan_theta_sqrd = fmaxf(1.0f - cos_theta_sqrd, 0.0f) / cos_theta_sqrd;
    return 2.0f / (1.0f + sqrt(1.0f + alpha_sqrd * tan_theta_sqrd));
}

__inline_all__ float smith_G(float alpha, const float3& wo, const float3& wi, const float3& halfway) {
    return G1(alpha, wo, halfway) * G1(alpha, wi, halfway);
}

// Height correlated smith geometric term.
// Understanding the Masking-Shadowing Function in Microfacet-Based BRDFs, Heitz 14, equation 99. 
__inline_all__ float height_correlated_smith_G(float alpha, const float3& wo, const float3& wi, const float3& halfway) {
    // TODO Use the heaviside function for numerator.
    return 1.0f / (1.0f + G1(alpha, wo, halfway) + G1(alpha, wi, halfway));
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
#if GPU_DEVICE
        rtThrow(OPTIX_GGX_WRONG_HEMISPHERE_EXCEPTION);
#else
        throw "OPTIX_GGX_WRONG_HEMISPHERE_EXCEPTION";
#endif
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

__inline_all__ BSDFSample sample(const float3& tint, float alpha, const float3& wo, float2 random_sample) {

    BSDFSample bsdf_sample;

    Distributions::DirectionalSample halfway_sample = Distributions::GGX::sample(alpha, random_sample);
    bsdf_sample.direction = reflect(-wo, halfway_sample.direction);
    bool discardSample = halfway_sample.PDF < 0.00001f || bsdf_sample.direction.z < 0.00001f; // Discard samples if the pdf is too low (precision issues) or if the new direction points into the surface (energy loss).
    if (discardSample) return BSDFSample::none();

    bsdf_sample.PDF = halfway_sample.PDF / (4.0f * dot(wo, halfway_sample.direction)); // TODO Discard based on this PDF, not the mirofacet normal PDF. And that aside, shouldn't the integrator do the discarding?

    { // Evaluate using the already computed sample PDF, which is D * cos_theta.
        const float3& wi = bsdf_sample.direction; // alias for readability.
        float G = height_correlated_smith_G(alpha, wo, wi, halfway_sample.direction);
        float D = halfway_sample.PDF / halfway_sample.direction.z;
        float F = 1.0f;
        bsdf_sample.weight = tint * ((D * F * G) / (4.0f * wo.z * wi.z));
    }
    return bsdf_sample;
}


//----------------------------------------------------------------------------
// GGX BTDF, Walter et al 07.
// Halfway is always on the same side as wi. TODO Is it??? Is halfway the normal halfway, or the halfway when both vectors have been flipped ot the same side??
//----------------------------------------------------------------------------

float3 refract(float no, float ni, const float3& wo, const float3& halfway) {
    float c = dot(wo, halfway);
    float n = ni / no;
    return (n * c - sign(wo.z) * sqrt(1.0f + n * (c * c - 1.0f))) * halfway - n * wo;
}

float jacobian_BTDF(float no, float ni, float wo_dot_h, float wi_dot_h) {
    float sqrt_d = ni * wi_dot_h + no * wo_dot_h;
    return no * no * abs(wo_dot_h) / (sqrt_d * sqrt_d); // TODO dot(wo, h) is always negative or? Hence the abs. Then use dot(wi, h) instead.
}

__inline_all__ float evaluate_BTDF(float alpha, float no, float ni, const float3& wo, const float3& wi, const float3& halfway) {
    float G = height_correlated_smith_G(alpha, wo, -wi, -halfway);
    float D = Distributions::GGX::D(alpha, halfway.z);
    float F = 1.0f; // No fresnel.

    float jacobian = jacobian_BTDF(no, ni, dot(wo, halfway), dot(wi, halfway));

    return (abs(dot(wi, halfway)) * D * F * G) / (abs(wo.z) * abs(wi.z)) * jacobian; // TODO Is the abs ever needed??
}

__inline_all__ float PDF_BTDF(float alpha, float no, float ni, const float3& wo, const float3& wi, const float3& halfway) {
#if _DEBUG
    if (dot(wi, halfway) < 0.0f || halfway.z < 0.0f) // TODO Is this correct?? Depends on how the halfway is defined.
#if GPU_DEVICE
        rtThrow(OPTIX_GGX_WRONG_HEMISPHERE_EXCEPTION);
#else
        throw "OPTIX_GGX_WRONG_HEMISPHERE_EXCEPTION";
#endif
#endif

    return Distributions::GGX::PDF(alpha, halfway.z) * jacobian_BTDF(no, ni, dot(wo, halfway), dot(wi, halfway));
}

__inline_all__ BSDFSample sample_BTDF(const float3& tint, float alpha, float no, float ni, const float3& wo, float2 random_sample) {

    BSDFSample bsdf_sample;

    Distributions::DirectionalSample halfway_sample = Distributions::GGX::sample(alpha, random_sample);
    bsdf_sample.direction = refract(no, ni, -wo, halfway_sample.direction);
    bool discardSample = halfway_sample.PDF < 0.00001f || bsdf_sample.direction.z < 0.00001f; // Discard samples if the pdf is too low (precision issues) or if the new direction points into the surface (energy loss).
    if (discardSample) return BSDFSample::none();

    const float3& h = halfway_sample.direction;
    const float3& wi = bsdf_sample.direction;

    bsdf_sample.PDF = halfway_sample.PDF * jacobian_BTDF(no, ni, dot(wo, h), dot(wi, h));
    bsdf_sample.weight = tint * evaluate_BTDF(alpha, no, no, wo, wi, h);
    
    return bsdf_sample;
}

} // NS GGX
} // NS BSDFs
} // NS Shading
} // NS OptiXRenderer

#endif // _OPTIXRENDERER_BSDFS_MICROFACET_H_