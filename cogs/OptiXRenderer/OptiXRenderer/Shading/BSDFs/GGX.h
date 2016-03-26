// OptiX renderer functions for microfacet models.
// ---------------------------------------------------------------------------
// Copyright (C) 2015-2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _OPTIXRENDERER_BSDFS_MICROFACET_H_
#define _OPTIXRENDERER_BSDFS_MICROFACET_H_

#include <OptiXRenderer/Types.h>

namespace OptiXRenderer {
namespace Shading {
namespace BSDFs {

//----------------------------------------------------------------------------
// Implementation of GGX's D and G term for the Microfacet model.
// Sources:
// * Walter et al 07.
// * https://github.com/tunabrain/tungsten/blob/master/src/core/bsdfs/Microfacet.hpp
// TODO
// * Create an optimized version where some expressions have been cancelled out.
//----------------------------------------------------------------------------
namespace GGX {

__inline_all__ float alpha_from_roughness(float roughness) {
    return roughness;
}

__inline_all__ float roughness_from_alpha(float alpha) {
    return alpha;
}

__inline_all__ float square(float v) {
    return v * v;
}

__inline_all__ float D(float alpha, const optix::float3& halfway) {
    // TODO Do we need this check here? Can we move it further out?
    if (halfway.z < 0.0f)
        return 0.0f;

    float alpha_sqrd = alpha * alpha;
    float cos_theta_sqrd = halfway.z * halfway.z;
    float tan_theta_sqrd = optix::fmaxf(1.0f - cos_theta_sqrd, 0.0f) / cos_theta_sqrd;
    float cos_theta_cubed = cos_theta_sqrd * cos_theta_sqrd;
    return (alpha_sqrd / PIf) / (cos_theta_cubed * square(alpha_sqrd + tan_theta_sqrd)); // TODO Combine divisions into a single one.
}

__inline_all__ float G1(float alpha, const optix::float3& w, const optix::float3& halfway) {
    if (optix::dot(w, halfway) * w.z <= 0.0f) // TODO Really? Can't I just check that they are on the same side, i.e w.z * halfway.z >= 0.0f. Perhaps even mvoe the check out to G().
        return 0.0f;

    float alpha_sqrd = alpha * alpha;
    float cos_theta_sqrd = w.z * w.z;
    float tan_theta_sqrd = optix::fmaxf(1.0f - cos_theta_sqrd, 0.0f) / cos_theta_sqrd;
    return 2.0f / (1.0f + sqrt(1.0f + alpha_sqrd * tan_theta_sqrd));
}

__inline_all__ float G(float alpha, const optix::float3& wo, const optix::float3& wi, const optix::float3& halfway) {
    return G1(alpha, wo, halfway) * G1(alpha, wi, halfway);
}

namespace Distribution {

__inline_all__ float PDF(float alpha, const optix::float3& halfway) {
    return D(alpha, halfway) * halfway.z;
}

__inline_all__ optix::float3 sample(float alpha, optix::float2 random_sample) {
    float phi = random_sample.y * (2.0f * PIf);

    float tan_theta_sqrd = alpha * alpha * random_sample.x / (1.0f - random_sample.x);
    float cos_theta = 1.0f / sqrt(1.0f + tan_theta_sqrd);
    
    float r = sqrt(optix::fmaxf(1.0f - cos_theta * cos_theta, 0.0f));
    return optix::make_float3(cos(phi) * r, sin(phi) * r, cos_theta);
}

} // NS Distribution

//----------------------------------------------------------------------------
// GGX BSDF, Walter et al 07.
//----------------------------------------------------------------------------

__inline_all__ float evaluate(float alpha, const optix::float3& wo, const optix::float3& wi, const optix::float3& halfway) {
    float G = GGX::G(alpha, wo, wi, halfway);
    float D = GGX::D(alpha, halfway);
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

    // Distribution::DirectionalSample halfway_sample = sample(random_sample); TODO
    optix::float3 halfway = Distribution::sample(alpha, random_sample);
    float PDF = Distribution::PDF(alpha, halfway);
    bsdf_sample.direction = optix::reflect(-wo, halfway);
    bool discardSample = PDF < 0.00001f || bsdf_sample.direction.z < 0.00001f; // Discard samples if the pdf is too low (precision issues) or if the new direction points into the surface (energy loss).
    if (discardSample) return BSDFSample::none();

    bsdf_sample.weight = evaluate(tint, alpha, wo, bsdf_sample.direction, halfway);
    bsdf_sample.PDF = PDF / (4.0f * abs(optix::dot(wo, halfway)));

    return bsdf_sample;
}

__inline_all__ float PDF(float alpha, const optix::float3& wo, const optix::float3& halfway) {
    return Distribution::PDF(alpha, halfway) / (4.0f * abs(optix::dot(wo, halfway)));
}

} // NS GGX
} // NS BSDFs
} // NS Shading
} // NS OptiXRenderer

#endif // _OPTIXRENDERER_BSDFS_MICROFACET_H_