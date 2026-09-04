// OptiX shading model utils.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _BIFROST_ASSETS_SHADING_SHADING_MODELS_UTILS_H_
#define _BIFROST_ASSETS_SHADING_SHADING_MODELS_UTILS_H_

#include <Bifrost/Assets/Shading/Constants.h>
#include <Bifrost/Assets/Shading/Fittings.h>
#include <Bifrost/Math/Color.h>

namespace Bifrost::Assets::Shading::ShadingModels {

// Scales the roughness of a material placed underneath a rough coat layer.
// This is done to simulate how a wider lobe from the rough transmission would perceptually widen the specular lobe of the underlying material.
// The implementation is based on equation 86 in the Roughening chapter of the OpenPBR course notes for Physically Based Shading 2025.
// https://blog.selfshadow.com/publications/s2025-shading-course/
_inline_all_archs_ float modulate_roughness_under_coat(float base_roughness, float coat_roughness) {
    auto pow4 = [](float v) -> float { float vv = v * v; return vv * vv; };

    float x_coat = 1 - air_ior / coat_ior;
    float adjusted_roughness4 = fminf(1, pow4(base_roughness) + 2.0f * x_coat * pow4(coat_roughness));
    return pow(adjusted_roughness4, 0.25f);
}

// ------------------------------------------------------------------------------------------------
// Specular rho helper.
// ------------------------------------------------------------------------------------------------
struct SpecularRho {
    float base, full;
    _inline_all_archs_ float rho(float specularity) { return Math::lerp(base, full, specularity); }
    _inline_all_archs_ Math::RGB rho(Math::RGB specularity) {
        return { rho(specularity.r), rho(specularity.g), rho(specularity.b) };
    }

    // Compensate for lost energy due to multiple scattering.
    // Multiple-scattering microfacet BSDFs with the smith model, Heitz et al., 2016 and 
    // Practical multiple scattering compensation for microfacet models, Emmanuel Turquin, 2018
    // showed that multiple-scattered reflectance has roughly the same distribution as single-scattering reflectance.
    // We can therefore account for energy lost to multi-scattering by computing the ratio of lost energy of a fully specular material,
    // and then scaling the specular reflectance by that ratio during evaluation, which increases reflectance to account for energy lost.
    _inline_all_archs_ float energy_loss_adjustment() const { return 1.0f / full; }

    _inline_all_archs_ static SpecularRho fetch(float abs_cos_theta, float roughness) {
        return { Rho::sample_GGX_with_fresnel(abs_cos_theta, roughness), Rho::sample_GGX(abs_cos_theta, roughness) };
    }
};

} // NS Bifrost::Assets::Shading::ShadingModels

#endif // _BIFROST_ASSETS_SHADING_SHADING_MODELS_UTILS_H_