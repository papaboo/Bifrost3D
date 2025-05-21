// OptiX renderer transmissive shading model.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _OPTIXRENDERER_SHADING_MODEL_TRANSMISSIVE_SHADING_H_
#define _OPTIXRENDERER_SHADING_MODEL_TRANSMISSIVE_SHADING_H_

#include <OptiXRenderer/Shading/BSDFs/GGX.h>
#include <OptiXRenderer/Utils.h>

namespace OptiXRenderer {
namespace Shading {
namespace ShadingModels {

// ---------------------------------------------------------------------------
// The transmissive shading material.
// ---------------------------------------------------------------------------
class TransmissiveShading {
private:
    optix::float3 m_transmitted_tint;
    float m_specularity;
    float m_ggx_alpha;
    float m_ior_i_over_o;
    float m_energy_loss_adjustment;

public:

    __inline_all__ void setup_shading(optix::float3 transmitted_tint, float roughness, float specularity, float cos_theta) {
        m_transmitted_tint = transmitted_tint;
        m_specularity = specularity;
        m_ggx_alpha = BSDFs::GGX::alpha_from_roughness(roughness);
        
        float medium_ior = dielectric_ior_from_specularity(m_specularity);
        bool entering = cos_theta >= 0.0f;
        float ior_o = entering ? 1.0f : medium_ior;
        float ior_i = entering ? medium_ior : 1.0f;
        m_ior_i_over_o = ior_i / ior_o;

        // Compensate for energy lost due to internal scattering.
        float rho = DielectricRho::fetch(abs(cos_theta), roughness, m_specularity).total_rho;
        m_energy_loss_adjustment = 1.0f / rho;
    }

    __inline_all__ TransmissiveShading(const Material& material, float cos_theta) {
        setup_shading(material.tint, material.roughness, material.specularity, cos_theta);
    }

#if GPU_DEVICE
    __inline_all__ TransmissiveShading(const Material& material, optix::float2 texcoord, float4 tint_and_roughness_scale, float cos_theta, float min_roughness = 0.0f) {
        using namespace optix;

        // Tint and roughness
        float4 tint_roughness = material.get_tint_roughness(texcoord) * tint_and_roughness_scale;
        float3 transmitted_tint = make_float3(tint_roughness);
        float roughness = max(tint_roughness.w, min_roughness);

        setup_shading(transmitted_tint, roughness, material.specularity, cos_theta);
    }

    __inline_all__ static TransmissiveShading initialize_with_max_PDF_hint(const Material& material, optix::float2 texcoord, float4 tint_and_roughness_scale, float cos_theta_o, PDF max_PDF_hint) {
        float min_roughness = GGXMinimumRoughness::from_PDF(abs(cos_theta_o), max_PDF_hint);
        return TransmissiveShading(material, texcoord, tint_and_roughness_scale, cos_theta_o, min_roughness);
    }
#endif

    __inline_all__ BSDFResponse evaluate_with_PDF(optix::float3 wo, optix::float3 wi) const {
        // Return no contribution if the material is viewed from behind.
        if (wo.z < 0.000001f)
            return BSDFResponse::none();

        BSDFResponse response = BSDFs::GGX::evaluate_with_PDF(m_transmitted_tint, m_ggx_alpha, m_specularity, m_ior_i_over_o, wo, wi);
        response.reflectance *= m_energy_loss_adjustment;
        return response;
    }

    __inline_all__ BSDFSample sample(optix::float3 wo, optix::float3 random_sample) const {
        // Don't sample material from behind.
        if (wo.z < 0.000001f)
            return BSDFSample::none();

        BSDFSample sample = BSDFs::GGX::sample(m_transmitted_tint, m_ggx_alpha, m_specularity, m_ior_i_over_o, wo, random_sample);
        sample.reflectance *= m_energy_loss_adjustment;
        return sample;
    }

    // Estimate the directional-hemispherical reflectance function.
    __inline_all__ optix::float3 rho(float abs_cos_theta) const {
        THROW(OPTIX_NOT_IMPLEMENTED);
        return optix::make_float3(1, 0, 1);
    }
};

} // NS ShadingModels
} // NS Shading
} // NS OptiXRenderer

#endif // _OPTIXRENDERER_SHADING_MODEL_TRANSMISSIVE_SHADING_H_