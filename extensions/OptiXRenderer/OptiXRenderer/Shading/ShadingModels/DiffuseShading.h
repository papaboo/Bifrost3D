// OptiX renderer diffuse shading model.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _OPTIXRENDERER_SHADING_MODEL_DIFFUSE_SHADING_H_
#define _OPTIXRENDERER_SHADING_MODEL_DIFFUSE_SHADING_H_

#include <Bifrost/Assets/Shading/BSDFs/OrenNayar.h>

#include <OptiXRenderer/Types.h>

namespace OptiXRenderer::Shading::ShadingModels {

// ---------------------------------------------------------------------------
// The diffuse shading model.
// ---------------------------------------------------------------------------
class DiffuseShading {
private:
    Bifrost::Math::RGB m_tint;
    float m_roughness;

public:

    __inline_all__ DiffuseShading(optix::float3 tint, float roughness)
        : m_tint(to_rgb(tint)), m_roughness(roughness) { }

    __inline_all__ BSDFResponse evaluate_with_PDF(optix::float3 wo, optix::float3 wi) const {
        // Return no contribution if the light is on the backside.
        if (wo.z < 0.000001f || wi.z < 0.000001f)
            return BSDFResponse::none();

        auto bsdf_response = Bifrost::Assets::Shading::BSDFs::OrenNayar::evaluate_with_PDF(m_tint, m_roughness, to_vector3f(wo), to_vector3f(wi));
        return bsdf_response;
    }

    __inline_all__ BSDFSample sample(optix::float3 wo, optix::float3 random_sample) const {
        // Don't sample material from behind.
        if (wo.z < 0.000001f)
            return BSDFSample::none();

        auto bsdf_sample = Bifrost::Assets::Shading::BSDFs::OrenNayar::sample(m_tint, m_roughness, to_vector3f(wo), { random_sample.x, random_sample.y });
        return bsdf_sample;
    }

    __inline_all__ optix::float3 rho(float abs_cos_theta) const {
        return to_float3(m_tint);
    }
};

} // NS OptiXRenderer::Shading::ShadingModels

#endif // _OPTIXRENDERER_SHADING_MODEL_DIFFUSE_SHADING_H_