// OptiX renderer functions for the Oren-Nayar BSDF.
// ---------------------------------------------------------------------------
// Copyright (C) 2015-2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _OPTIXRENDERER_BSDFS_ORENNAYAR_H_
#define _OPTIXRENDERER_BSDFS_ORENNAYAR_H_

#include <OptiXRenderer/Distributions.h>
#include <OptiXRenderer/Types.h>

namespace OptiXRenderer {
namespace Shading {
namespace BSDFs {

//----------------------------------------------------------------------------
// Implementation of the Oren-Nayar refectance model.
// Sources:
// * http://www1.cs.columbia.edu/CAVE/publications/pdfs/Oren_SIGGRAPH94.pdf
// * https://en.wikipedia.org/wiki/Oren%E2%80%93Nayar_reflectance_model
// * https://github.com/tunabrain/tungsten/blob/master/src/core/bsdfs/OrenNayarBsdf.cpp
//----------------------------------------------------------------------------
namespace OrenNayar {

using namespace optix;

struct State {
    float A;
    float B;
};

__inline_all__ State compute_state(float roughness) {
    float sigma_sqrd = roughness * roughness;
    State s;
    s.A = 1.0f - (sigma_sqrd / (2.0f * sigma_sqrd + 0.66f));
    s.B = 0.45f * sigma_sqrd / (sigma_sqrd + 0.09f);
    return s;
}

__inline_all__ float evaluate(float roughness, const float3& wo, const float3& wi) {
    float2 cos_theta = make_float2(abs(wi.z), abs(wo.z));
    const float sin_theta_sqrd = (1.0f - cos_theta.x * cos_theta.x) * (1.0f - cos_theta.y * cos_theta.y);
    float sin_theta = sqrt(fmaxf(0.0f, sin_theta_sqrd)); // AVH Hmm this is not sin_theta to the above cos_theta. Is it sin_theta to the halfwayvector? Must investigate!

    const float3 normal = make_float3(0.0f, 0.0f, 1.0f);
    float3 light_plane = normalize(wi - cos_theta.x * normal);
    float3 view_plane = normalize(wo - cos_theta.y * normal);
    float cos_phi = clamp(dot(light_plane, view_plane), 0.0f, 1.0f);

    State state = compute_state(roughness);
    return (state.A + state.B * cos_phi * sin_theta / fmaxf(cos_theta.x, cos_theta.y)) * RECIP_PIf;
}

__inline_all__ float3 evaluate(const float3& tint, float roughness, const float3& wo, const float3& wi) {
    return tint * evaluate(roughness, wo, wi);
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