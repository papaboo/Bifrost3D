// OptiX renderer functions for the Burley BSDF.
// ---------------------------------------------------------------------------
// Copyright (C) 2015-2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _OPTIXRENDERER_BSDFS_BURLEY_H_
#define _OPTIXRENDERER_BSDFS_BURLEY_H_

#include <OptiXRenderer/Distributions.h>
#include <OptiXRenderer/Types.h>

namespace OptiXRenderer {
namespace Shading {
namespace BSDFs {

//----------------------------------------------------------------------------
// Implementation of the Burley refectance model.
// Sources:
// * Physically-Based Shading at Disney, Burley et al., 12.
// * https://github.com/wdas/brdf/blob/master/src/brdfs/disney.brdf
// Future work
// * Improve sampling. Perhaps GGX or PowerCosine distribution will make a better fit.
//----------------------------------------------------------------------------
namespace Burley {

using namespace optix;

__inline_all__ static float schlick_fresnel(float abs_cos_theta) {
    float m = clamp(1.0f - abs_cos_theta, 0.0f, 1.0f);
    float m2 = m * m;
    return m * m2 * m2; // pow(m, 5)
}

__inline_all__ float evaluate(float roughness, const float3& wo, const float3& wi, const float3& halfway) {
    float wi_dot_halfway = dot(wi, halfway);
    float fd90 = 0.5f + 2.0f * wi_dot_halfway * wi_dot_halfway * roughness;
    float fresnel_wo = schlick_fresnel(wo.z);
    float fresnel_wi = schlick_fresnel(wi.z);
    float normalizer = 1.0f / lerp(0.969371021f, 1.04337633f, roughness); // Burley isn't energy conserving, so we normalize by a 'good enough' constant here.
    return lerp(1.0f, fd90, fresnel_wo) * lerp(1.0f, fd90, fresnel_wi) * RECIP_PIf *normalizer;
}

__inline_all__ float3 evaluate(const float3& tint, float roughness, const float3& wo, const float3& wi, const float3& halfway) {
    return tint * evaluate(roughness, wo, wi, halfway);
}

__inline_all__ float3 evaluate(const float3& tint, float roughness, const float3& wo, const float3& wi) {
    float3 halfway = normalize(wi + wo);
    return tint * evaluate(roughness, wo, wi, halfway);
}

__inline_all__ float PDF(float roughness, const float3& wo, const float3& wi) {
    return Distributions::Cosine::PDF(wi.z);
}

__inline_all__ BSDFResponse evaluate_with_PDF(const float3& tint, float roughness, const float3& wo, const float3& wi) {
    BSDFResponse response;
    response.weight = evaluate(tint, roughness, wo, wi);
    response.PDF = PDF(roughness, wo, wi);
    return response;
}

__inline_all__ BSDFSample sample(const float3& tint, float roughness, const float3& wo, float2 random_sample) {
    // Sampling can potentially be improved by combining a uniform and cosine distribution, based on roughness.
    Distributions::DirectionalSample cosine_sample = Distributions::Cosine::sample(random_sample);
    BSDFSample bsdf_sample;
    bsdf_sample.direction = cosine_sample.direction;
    bsdf_sample.PDF = cosine_sample.PDF;
    bsdf_sample.weight = evaluate(tint, roughness, wo, bsdf_sample.direction);
    return bsdf_sample;
}

} // NS OrenNayar
} // NS BSDFs
} // NS Shading
} // NS OptiXRenderer

#endif // _OPTIXRENDERER_BSDFS_ORENNAYAR_H_