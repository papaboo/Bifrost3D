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

// Understanding the Masking-Shadowing Function in Microfacet-Based BRDFs, Heitz 14.
namespace Heitz {

// Equation 72.
__inline_all__ float lambda(float alpha, const float3& w) {
    // NOTE At grazing angles the masking function is going to return very low values. 
    //      This is generally not a problem, unless alpha is low as well, 
    //      meaning that the in and out vectors will always be at grazing angles 
    //      and therefore the masking is consistently underestimated. 
    //      We could fix this be adding a very specific scaling for this one case.
    //      Check the GGX Rho computation for validity.
    float cos_theta_sqrd = w.z * w.z;
    float tan_theta_sqrd = fmaxf(1.0f - cos_theta_sqrd, 0.0f) / cos_theta_sqrd;
    float a_sqrd = 1.0f / (alpha * alpha * tan_theta_sqrd);
    return (-1.0f + sqrt(1.0f + 1.0f / a_sqrd)) / 2.0f;
}

__inline_all__ float masking(float alpha, const float3& wo) {
    return 1.0f / (1.0f + lambda(alpha, wo));
}

// Height correlated smith geometric term. Equation 98. 
__inline_all__ float separable_smith_G(float alpha, const float3& wo, const float3& wi, const float3& halfway) {
    float headysided = heaviside(dot(wo, halfway)) * heaviside(dot(wi, halfway)); // TODO Should always evaluate to 1.0f. But if we replace it by a constant then guard it by an exception that catches cases where it isn't 1.0f in debug builds.
    return headysided / ((1.0f + lambda(alpha, wo) * (1.0f + lambda(alpha, wi))));
}

// Height correlated smith geometric term. Equation 99. 
__inline_all__ float height_correlated_smith_G(float alpha, const float3& wo, const float3& wi, const float3& halfway) {
    float headysided = heaviside(dot(wo, halfway)) * heaviside(dot(wi, halfway)); // TODO Should always evaluate to 1.0f. But if we replace it by a constant then guard it by an exception that catches cases where it isn't 1.0f in debug builds.
    return headysided / (1.0f + lambda(alpha, wo) + lambda(alpha, wi));
}

} // NS Heitz

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
    float G = Heitz::height_correlated_smith_G(alpha, wo, wi, halfway);
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
    float G = Heitz::height_correlated_smith_G(alpha, wo, wi, halfway);
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
        float G = Heitz::height_correlated_smith_G(alpha, wo, wi, halfway_sample.direction);
        float D = halfway_sample.PDF / halfway_sample.direction.z;
        float F = 1.0f;
        bsdf_sample.weight = tint * ((D * F * G) / (4.0f * wo.z * wi.z));
    }
    return bsdf_sample;
}

} // NS GGX
} // NS BSDFs
} // NS Shading
} // NS OptiXRenderer

#endif // _OPTIXRENDERER_BSDFS_MICROFACET_H_