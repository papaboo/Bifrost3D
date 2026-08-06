// Bifrost functions for the Burley BSDF.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _BIFROST_ASSETS_SHADING_BSDFS_BURLEY_H_
#define _BIFROST_ASSETS_SHADING_BSDFS_BURLEY_H_

#include <Bifrost/Assets/Shading/Utils.h>
#include <Bifrost/Math/Distributions.h>
#include <Bifrost/Math/Utils.h>

namespace Bifrost::Assets::Shading::BSDFs {

//----------------------------------------------------------------------------
// Implementation of the Burley refectance model.
// Sources:
// * Physically-Based Shading at Disney, Burley et al., 12.
// * https://github.com/wdas/brdf/blob/master/src/brdfs/disney.brdf
//----------------------------------------------------------------------------
namespace Burley {

using namespace Bifrost::Math;

_inline_all_archs_ static float schlick_fresnel(float abs_cos_theta) {
    float m = fmaxf(1.0f - abs_cos_theta, 0.0f);
    return pow5(m);
}

_inline_all_archs_ float evaluate(float roughness, Vector3f wo, Vector3f wi, Vector3f halfway) {
    float wi_dot_halfway = dot(wi, halfway);
    float fd90 = 0.5f + 2.0f * wi_dot_halfway * wi_dot_halfway * roughness;
    float fresnel_wo = schlick_fresnel(wo.z);
    float fresnel_wi = schlick_fresnel(wi.z);
    float normalizer = 1.0f / lerp(0.969371021f, 1.04337633f, roughness); // Burley isn't energy conserving, so we normalize by a 'good enough' constant here.
    return lerp(1.0f, fd90, fresnel_wo) * lerp(1.0f, fd90, fresnel_wi) * RECIP_PIf * normalizer;
}

_inline_all_archs_ RGB evaluate(RGB tint, float roughness, Vector3f wo, Vector3f wi, Vector3f halfway) {
    return tint * evaluate(roughness, wo, wi, halfway);
}

_inline_all_archs_ RGB evaluate(RGB tint, float roughness, Vector3f wo, Vector3f wi) {
    Vector3f halfway = normalize(wi + wo);
    return tint * evaluate(roughness, wo, wi, halfway);
}

_inline_all_archs_ PDF pdf(float roughness, Vector3f wo, Vector3f wi) {
    return Distributions::Cosine::PDF(wi.z);
}

_inline_all_archs_ BSDFResponse evaluate_with_PDF(RGB tint, float roughness, Vector3f wo, Vector3f wi) {
    auto reflectance = evaluate(tint, roughness, wo, wi);
    auto PDF = pdf(roughness, wo, wi);
    return { reflectance, PDF };
}

_inline_all_archs_ BSDFSample sample(RGB tint, float roughness, Vector3f wo, Vector2f random_sample) {
    // Sampling can potentially be improved by combining a uniform and cosine distribution, based on roughness.
    auto cosine_sample = Math::Distributions::Cosine::sample(random_sample);
    BSDFSample bsdf_sample;
    bsdf_sample.direction = cosine_sample.direction;
    bsdf_sample.PDF = cosine_sample.PDF;
    bsdf_sample.reflectance = evaluate(tint, roughness, wo, bsdf_sample.direction);
    return bsdf_sample;
}

} // NS Burley
} // NS Bifrost::Assets::Shading::BSDFs

#endif // _BIFROST_ASSETS_SHADING_BSDFS_BURLEY_H_