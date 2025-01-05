// OptiX renderer diffuse shading model.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _OPTIXRENDERER_SHADING_MODEL_DIFFUSE_SHADING_H_
#define _OPTIXRENDERER_SHADING_MODEL_DIFFUSE_SHADING_H_

#include <OptiXRenderer/Shading/BSDFs/OrenNayar.h>

namespace OptiXRenderer {
namespace Shading {
namespace ShadingModels {

// ---------------------------------------------------------------------------
// The diffuse shading model.
// ---------------------------------------------------------------------------
class DiffuseShading {
private:
    optix::float3 m_tint;
    float m_roughness;

public:

    __inline_all__ DiffuseShading(optix::float3 tint, float roughness)
        : m_tint(tint), m_roughness(roughness) { }

    __inline_all__ BSDFResponse evaluate_with_PDF(optix::float3 wo, optix::float3 wi) const {
        // Return no contribution if the light is on the backside.
        if (wo.z < 0.000001f || wi.z < 0.000001f)
            return BSDFResponse::none();

        return BSDFs::OrenNayar::evaluate_with_PDF(m_tint, m_roughness, wo, wi);
    }

    __inline_all__ BSDFSample sample(optix::float3 wo, optix::float3 random_sample) const {
        // Don't sample material from behind.
        if (wo.z < 0.000001f)
            return BSDFSample::none();

        return BSDFs::OrenNayar::sample(m_tint, m_roughness, wo, make_float2(random_sample));
    }

    __inline_all__ optix::float3 rho(float abs_cos_theta) const {
        return m_tint;
    }
};

} // NS ShadingModels
} // NS Shading
} // NS OptiXRenderer

#endif // _OPTIXRENDERER_SHADING_MODEL_DIFFUSE_SHADING_H_