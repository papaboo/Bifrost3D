// Bifrost diffuse shading model.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _BIFROST_ASSETS_SHADING_SHADING_MODELS_DIFFUSE_SHADING_H_
#define _BIFROST_ASSETS_SHADING_SHADING_MODELS_DIFFUSE_SHADING_H_

#include <Bifrost/Assets/Shading/BSDFs/OrenNayar.h>

namespace Bifrost::Assets::Shading::ShadingModels {

// ---------------------------------------------------------------------------
// The diffuse shading model.
// ---------------------------------------------------------------------------
class DiffuseShading {
private:
    Math::RGB m_tint;
    float m_roughness;

public:

    _inline_all_archs_ DiffuseShading(Math::RGB tint, float roughness)
        : m_tint(tint), m_roughness(roughness) { }

    _inline_all_archs_ BSDFResponse evaluate_with_PDF(Math::Vector3f wo, Math::Vector3f wi) const {
        // Return no contribution if the light is on the backside.
        if (wo.z < 0.000001f || wi.z < 0.000001f)
            return BSDFResponse::none();

        return Bifrost::Assets::Shading::BSDFs::OrenNayar::evaluate_with_PDF(m_tint, m_roughness, wo, wi);
    }

    _inline_all_archs_ BSDFSample sample(Math::Vector3f wo, Math::Vector3f random_sample) const {
        // Don't sample material from behind.
        if (wo.z < 0.000001f)
            return BSDFSample::none();

        return Bifrost::Assets::Shading::BSDFs::OrenNayar::sample(m_tint, m_roughness, wo, { random_sample.x, random_sample.y });
    }

    _inline_all_archs_ Math::RGB rho(float abs_cos_theta) const {
        return m_tint;
    }
};

} // NS Bifrost::Assets::Shading::ShadingModels

#endif // _BIFROST_ASSETS_SHADING_SHADING_MODELS_DIFFUSE_SHADING_H_