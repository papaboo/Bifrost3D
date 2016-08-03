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

namespace OptiXRenderer {
namespace Shading {
namespace BSDFs {

//----------------------------------------------------------------------------
// Implementation of GGX's D and G term for the Microfacet model.
// Sources:
// * Walter et al 07.
// * https://github.com/tunabrain/tungsten/blob/master/src/core/bsdfs/Microfacet.hpp
// Future work
// * Reference equations in Walter07 and implement different G terms.
// * Create an optimized version where some expressions have been cancelled out.
//----------------------------------------------------------------------------
namespace GGX {

__inline_all__ float alpha_from_roughness(float roughness) {
    return optix::fmaxf(0.00000000001f, roughness * roughness);
}

__inline_all__ float roughness_from_alpha(float alpha) {
    return sqrt(alpha);
}

// TODO This isn't the smith or schlick geometric term. Which is it?
__inline_all__ float G1(float alpha, const optix::float3& w, const optix::float3& halfway) {
    // Check if the w vector projected onto the halfway vector is in the same hemisphere as the halfway vector and the normal, otherwise the response is black.
    // TODO We know that this is always the case when sampling, so ignore the checks. What about evaluation? It's not guaranteed, but it would be silly evaluate incompatible directions.
    if (optix::dot(w, halfway) * w.z <= 0.0f)
        return 0.0f;

    float alpha_sqrd = alpha * alpha;
    float cos_theta_sqrd = w.z * w.z;
    float tan_theta_sqrd = optix::fmaxf(1.0f - cos_theta_sqrd, 0.0f) / cos_theta_sqrd;
    return 2.0f / (1.0f + sqrt(1.0f + alpha_sqrd * tan_theta_sqrd));
}

__inline_all__ float G(float alpha, const optix::float3& wo, const optix::float3& wi, const optix::float3& halfway) {
    return G1(alpha, wo, halfway) * G1(alpha, wi, halfway);
}

//----------------------------------------------------------------------------
// GGX BSDF, Walter et al 07.
//----------------------------------------------------------------------------

__inline_all__ float evaluate(float alpha, const optix::float3& wo, const optix::float3& wi, const optix::float3& halfway) {
    float G = GGX::G(alpha, wo, wi, halfway);
    float D = Distributions::GGX::D(alpha, halfway.z);
    float F = 1.0f; // No fresnel.
    return (D * F * G) / (4.0f * wo.z * wi.z);
}

__inline_all__ optix::float3 evaluate(const optix::float3& tint, float alpha, const optix::float3& wo, const optix::float3& wi, const optix::float3& halfway) {
    return tint * evaluate(alpha, wo, wi, halfway);
}

__inline_all__ optix::float3 evaluate(const optix::float3& tint, float alpha, const optix::float3& wo, const optix::float3& wi) {
    const optix::float3 halfway = optix::normalize(wi + wo);
    return evaluate(tint, alpha, wo, wi, halfway);
}

__inline_all__ BSDFSample sample(const optix::float3& tint, float alpha, const optix::float3& wo, optix::float2 random_sample) {

    BSDFSample bsdf_sample;

    Distributions::DirectionalSample halfway_sample = Distributions::GGX::sample(alpha, random_sample);
    bsdf_sample.direction = optix::reflect(-wo, halfway_sample.direction);
    bool discardSample = halfway_sample.PDF < 0.00001f || bsdf_sample.direction.z < 0.00001f; // Discard samples if the pdf is too low (precision issues) or if the new direction points into the surface (energy loss).
    if (discardSample) return BSDFSample::none();

    bsdf_sample.PDF = halfway_sample.PDF / (4.0f * optix::dot(wo, halfway_sample.direction));

    { // Evaluate using the already computed sample PDF, which is D * cos_theta.
        const optix::float3& wi = bsdf_sample.direction; // alias for readability.
        float G = GGX::G(alpha, wo, wi, halfway_sample.direction);
        float D = halfway_sample.PDF / halfway_sample.direction.z;
        float F = 1.0f;
        bsdf_sample.weight = tint * ((D * F * G) / (4.0f * wo.z * wi.z));
    }
    return bsdf_sample;
}

__inline_all__ float PDF(float alpha, const optix::float3& wo, const optix::float3& wi) {
    const optix::float3 halfway = optix::normalize(wo + wi);

    // NOTE Do we need these checks? What happens when the BSDF is a BTDF and the last one fails?
    if (optix::dot(wo, halfway) < 0.0f || halfway.z < 0.0f)
        return 0.0f;

    return Distributions::GGX::PDF(alpha, halfway.z) / (4.0f * optix::dot(wo, halfway));
}

} // NS GGX
} // NS BSDFs
} // NS Shading
} // NS OptiXRenderer

#endif // _OPTIXRENDERER_BSDFS_MICROFACET_H_