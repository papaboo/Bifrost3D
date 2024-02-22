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
// The diffuse shading material.
// ---------------------------------------------------------------------------
class DiffuseShading {
private:
    optix::float3 m_tint;

public:

    __inline_all__ DiffuseShading(const Material& material)
        : m_tint(material.tint) { }

#if GPU_DEVICE
    __inline_all__ DiffuseShading(const Material& material, optix::float2 texcoord) {
        m_tint = material.tint;
        if (material.tint_roughness_texture_ID)
            m_tint *= make_float3(optix::rtTex2D<optix::float4>(material.tint_roughness_texture_ID, texcoord.x, texcoord.y));
    }
#endif

    __inline_all__ optix::float3 evaluate(optix::float3 wo, optix::float3 wi) const {
        return BSDFs::Lambert::evaluate(m_tint, wo, wi);
    }

    __inline_all__ float PDF(optix::float3 wo, optix::float3 wi) const {
        return BSDFs::Lambert::PDF(wo, wi);
    }

    __inline_all__ BSDFResponse evaluate_with_PDF(optix::float3 wo, optix::float3 wi) const {
        return BSDFs::Lambert::evaluate_with_PDF(m_tint, wo, wi);
    }

    __inline_all__ BSDFSample sample(optix::float3 wo, optix::float3 random_sample) const {
        return BSDFs::Lambert::sample(m_tint, make_float2(random_sample));
    }
};

} // NS ShadingModels
} // NS Shading
} // NS OptiXRenderer

#endif // _OPTIXRENDERER_SHADING_MODEL_DIFFUSE_SHADING_H_