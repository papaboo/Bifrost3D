// OptiX renderer rough glass shading model.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _OPTIXRENDERER_SHADING_MODEL_ROUGH_GLASS_SHADING_H_
#define _OPTIXRENDERER_SHADING_MODEL_ROUGH_GLASS_SHADING_H_

#include <OptiXRenderer/Shading/BSDFs/GGX.h>
#include <OptiXRenderer/Utils.h>

namespace OptiXRenderer {
namespace Shading {
namespace ShadingModels {

// ---------------------------------------------------------------------------
// The rough glass shading material.
// ---------------------------------------------------------------------------
class RoughGlassShading {
private:
    optix::float3 m_tint;
    float m_specularity;
    float m_ggx_alpha;
    float m_ior_i_over_o;

public:

    __inline_all__ void setup_shading(optix::float3 tint, float roughness, float specularity, float cos_theta) {
        m_tint = tint;
        m_specularity = specularity;
        m_ggx_alpha = BSDFs::GGX::alpha_from_roughness(roughness);
        
        float medium_ior = dielectric_ior_from_specularity(m_specularity);
        bool entering = cos_theta >= 0.0f;
        float ior_o = entering ? 1.0f : medium_ior;
        float ior_i = entering ? medium_ior : 1.0f;
        m_ior_i_over_o = ior_i / ior_o;
    }

    __inline_all__ RoughGlassShading(const Material& material, float cos_theta) {
        setup_shading(material.tint, material.roughness, material.specularity, cos_theta);
    }

#if GPU_DEVICE
    __inline_all__ RoughGlassShading(const Material& material, optix::float2 texcoord, float cos_theta) {
        using namespace optix;

        // Tint and roughness
        float4 tint_roughness = make_float4(material.tint, material.roughness);
        if (material.tint_roughness_texture_ID)
            tint_roughness *= rtTex2D<float4>(material.tint_roughness_texture_ID, texcoord.x, texcoord.y);
        else if (material.roughness_texture_ID)
            tint_roughness.w *= rtTex2D<float>(material.roughness_texture_ID, texcoord.x, texcoord.y);
        float3 tint = make_float3(tint_roughness);
        float roughness = tint_roughness.w;

        setup_shading(tint, roughness, material.specularity, cos_theta);
    }
#endif

    __inline_all__ BSDFResponse evaluate_with_PDF(optix::float3 wo, optix::float3 wi) const {
        return BSDFs::GGX::evaluate_with_PDF(m_tint, m_ggx_alpha, m_specularity, m_ior_i_over_o, wo, wi);
    }

    __inline_all__ BSDFSample sample(optix::float3 wo, optix::float3 random_sample) const {
        return BSDFs::GGX::sample(m_tint, m_ggx_alpha, m_specularity, m_ior_i_over_o, wo, random_sample);
    }

};

} // NS ShadingModels
} // NS Shading
} // NS OptiXRenderer

#endif // _OPTIXRENDERER_SHADING_MODEL_ROUGH_GLASS_SHADING_H_