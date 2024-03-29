// OptiX renderer diffuse shading model.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _OPTIXRENDERER_SHADING_MODEL_DIFFUSE_SHADING_H_
#define _OPTIXRENDERER_SHADING_MODEL_DIFFUSE_SHADING_H_

#include <OptiXRenderer/Shading/BSDFs/Lambert.h>

namespace OptiXRenderer {
namespace Shading {
namespace ShadingModels {

// ---------------------------------------------------------------------------
// The diffuse shading model.
// ---------------------------------------------------------------------------
class DiffuseShading {
private:
    optix::float3 m_tint;

public:

    __inline_all__ DiffuseShading(optix::float3 tint)
        : m_tint(tint) { }

    __inline_all__ optix::float3 evaluate(optix::float3 wo, optix::float3 wi) const {
        if (!same_hemisphere(wo, wi))
            return { 0, 0, 0 };
        return BSDFs::Lambert::evaluate(m_tint, wo, wi);
    }

    __inline_all__ float PDF(optix::float3 wo, optix::float3 wi) const {
        if (!same_hemisphere(wo, wi))
            return 0;
        return BSDFs::Lambert::PDF(wo, wi);
    }

    __inline_all__ BSDFResponse evaluate_with_PDF(optix::float3 wo, optix::float3 wi) const {
        if (!same_hemisphere(wo, wi))
            return BSDFResponse::none();
        return BSDFs::Lambert::evaluate_with_PDF(m_tint, wo, wi);
    }

    __inline_all__ BSDFSample sample(optix::float3 wo, optix::float3 random_sample) const {
        auto res = BSDFs::Lambert::sample(m_tint, make_float2(random_sample));
        if (wo.z < 0.0f)
            res.direction.z = -res.direction.z;
        return res;
    }

    __inline_all__ optix::float3 rho(float abs_cos_theta) const {
        return m_tint;
    }
};

} // NS ShadingModels
} // NS Shading
} // NS OptiXRenderer

#endif // _OPTIXRENDERER_SHADING_MODEL_DIFFUSE_SHADING_H_