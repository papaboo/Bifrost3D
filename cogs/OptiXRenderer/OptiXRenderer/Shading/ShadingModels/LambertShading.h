// OptiX renderer lambert shading model.
// ---------------------------------------------------------------------------
// Copyright (C) 2015-2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _OPTIXRENDERER_SHADING_MODEL_LAMBERT_SHADING_H_
#define _OPTIXRENDERER_SHADING_MODEL_LAMBERT_SHADING_H_

#include <OptiXRenderer/Shading/BSDFs/Lambert.h>
#include <OptiXRenderer/Utils.h>

namespace OptiXRenderer {
namespace Shading {
namespace ShadingModels {

// ---------------------------------------------------------------------------
// The lambert shading material.
// ---------------------------------------------------------------------------
class LambertShading {
private:
    const Material& m_material;
    optix::float3 m_base_tint;

    // TODO Template with return type? Might be hell on the CPU.
    // TODO Move to utils.
    __inline_all__ static optix::float4 tex2D(unsigned int texture_ID, optix::float2 texcoord) {
#if GPU_DEVICE
        optix::float4 texel = optix::rtTex2D<optix::float4>(texture_ID, texcoord.x, texcoord.y);
        return texel;
#else
        return optix::make_float4(1.0f);
#endif
    }

public:

    __inline_all__ LambertShading(const Material& material)
        : m_material(material)
        , m_base_tint(material.base_tint) { }

    __inline_all__ LambertShading(const Material& material, optix::float2 texcoord)
        : m_material(material)
    {
        m_base_tint = material.base_tint;
        if (material.base_tint_texture_ID)
            m_base_tint *= make_float3(tex2D(material.base_tint_texture_ID, texcoord));
    }

    __inline_all__ optix::float3 evaluate(optix::float3 wo, optix::float3 wi) const {
        return BSDFs::Lambert::evaluate(m_base_tint);
    }

    __inline_all__ BSDFSample sample_one(const optix::float3& wo, const optix::float3& random_sample) const {
        return BSDFs::Lambert::sample(m_base_tint, make_float2(random_sample));
    }

    __inline_all__ BSDFSample sample_all(const optix::float3& wo, const optix::float3& random_sample) const {
        return BSDFs::Lambert::sample(m_base_tint, make_float2(random_sample));
    }

    __inline_all__ float PDF(const optix::float3& wo, const optix::float3& wi) const {
        return BSDFs::Lambert::PDF(wo, wi);
    }

}; // NS DefaultShading

} // NS ShadingModels
} // NS Shading
} // NS OptiXRenderer

#endif // _OPTIXRENDERER_SHADING_MODEL_LAMBERT_SHADING_H_