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
    optix::float3 m_transmission_tint;
    float m_specularity;
    float m_ggx_alpha;
    float m_ior_i_over_o;
    float m_energy_loss_adjustment;

public:

    __inline_all__ void setup_shading(optix::float3 transmission_tint, float roughness, float specularity, float cos_theta_o) {
        m_transmission_tint = transmission_tint;
        m_specularity = specularity;
        m_ggx_alpha = BSDFs::GGX::alpha_from_roughness(roughness);
        
        float medium_ior = dielectric_ior_from_specularity(m_specularity);
        bool entering = cos_theta_o >= 0.0f;
        float ior_o = entering ? AIR_IOR : medium_ior;
        float ior_i = entering ? medium_ior : AIR_IOR;
        m_ior_i_over_o = ior_i / ior_o;

        // Compensate for energy lost due to internal scattering.
        float rho = DielectricRho::fetch(abs(cos_theta_o), roughness, m_ior_i_over_o).total_rho;
        m_energy_loss_adjustment = 1.0f / rho;
    }

    __inline_all__ TransmissiveShading(const Material& material, float cos_theta_o) {
        setup_shading(material.tint, material.roughness, material.specularity, cos_theta_o);
    }

#if GPU_DEVICE
    __inline_all__ TransmissiveShading(const Material& material, optix::float2 texcoord, float4 tint_and_roughness_scale, float cos_theta_o, float min_roughness = 0.0f) {
        using namespace optix;

        // Tint and roughness
        float4 tint_roughness = material.get_tint_roughness(texcoord) * tint_and_roughness_scale;
        float3 transmission_tint = make_float3(tint_roughness);
        float roughness = max(tint_roughness.w, min_roughness);

        setup_shading(transmission_tint, roughness, material.specularity, cos_theta_o);
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

        BSDFResponse response = BSDFs::GGX::evaluate_with_PDF(m_transmission_tint, m_ggx_alpha, m_specularity, m_ior_i_over_o, wo, wi);
        response.reflectance *= m_energy_loss_adjustment;
        return response;
    }

    __inline_all__ BSDFSample sample(optix::float3 wo, optix::float3 random_sample) const {
        // Don't sample material from behind.
        if (wo.z < 0.000001f)
            return BSDFSample::none();

        BSDFSample sample = BSDFs::GGX::sample(m_transmission_tint, m_ggx_alpha, m_specularity, m_ior_i_over_o, wo, random_sample);
        sample.reflectance *= m_energy_loss_adjustment;
        return sample;
    }

    // Estimate the directional-hemispherical reflectance function.
    __inline_all__ optix::float3 rho(float abs_cos_theta_o) const {
        float roughness = BSDFs::GGX::roughness_from_alpha(m_ggx_alpha);
        DielectricRho rho = DielectricRho::fetch(abs_cos_theta_o, roughness, m_ior_i_over_o);
        float reflection = rho.reflected_rho / rho.total_rho;
        return reflection + (1 - reflection) * m_transmission_tint;
    }
};

} // NS ShadingModels
} // NS Shading
} // NS OptiXRenderer

#endif // _OPTIXRENDERER_SHADING_MODEL_TRANSMISSIVE_SHADING_H_