// OptiX renderer GGX reflection shading layer.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _OPTIXRENDERER_SHADING_MODELS_LAYERS_GGX_REFLECTION_H_
#define _OPTIXRENDERER_SHADING_MODELS_LAYERS_GGX_REFLECTION_H_

#include <OptiXRenderer/Shading/BSDFs/GGX.h>
#include <OptiXRenderer/Shading/ShadingModels/Utils.h>

namespace OptiXRenderer {
namespace Shading {
namespace ShadingModels {
namespace Layers {

// ---------------------------------------------------------------------------
// A layer that reflects light according to the GGX microfacet distribution.
// Any light not reflected is considered transmitted
// ---------------------------------------------------------------------------
template <typename TransmissionShading>
class GGXReflection {
private:
    typedef optix::float3 float3;

    float m_alpha;
    float3 m_specularity;
    float m_reflection_energy_loss_adjustment;
    
    // Can fade the GGX reflection in and out by setting the sub-pixel opacity of the reflection.
    // An opacity of zero means the reflection is ignored, an opacity of one means the reflection is fully enabled.
    // This is useful to simulate sub-pixel reflections where a part of the pixel should have a reflection and part of it shouldn't.
    float m_layer_opacity;

    float3 m_transmission_scale;
    TransmissionShading m_transmission_shading_model;

    // The probability of choosing the reflection BRDF instead of the BRDF after transmission.
    float m_reflection_probability;

public:

    GGXReflection() = default;

    __inline_all__ GGXReflection(float roughness, optix::float3 specularity, float abs_cos_theta_o, TransmissionShading& transmission_shading_model, float opacity = 1.0f) {
        m_alpha = BSDFs::GGX::alpha_from_roughness(roughness);
        m_specularity = specularity;
        m_layer_opacity = opacity;
        m_transmission_shading_model = transmission_shading_model;

        SpecularRho specular_rho = SpecularRho::fetch(abs_cos_theta_o, roughness);
        m_reflection_energy_loss_adjustment = 1.0f / specular_rho.full;

        float3 single_bounce_rho = m_layer_opacity * specular_rho.rho(m_specularity);
        float3 lossless_rho = single_bounce_rho * m_reflection_energy_loss_adjustment;
        m_transmission_scale = 1.0f - lossless_rho;

        // Compute the probability to select reflection over transmission.
        // Ensure that the divisor is non-zero to avoid NaN's in the edge case that both BSDFs are black.
        float reflection_rho_sum = sum(compute_reflection_rho(abs_cos_theta_o));
        float transmission_rho_sum = sum(compute_transmission_rho(abs_cos_theta_o));
        m_reflection_probability = reflection_rho_sum / fmaxf(reflection_rho_sum + transmission_rho_sum, nextafterf(0, 1));
    }

    __inline_all__ BSDFResponse evaluate_with_PDF(optix::float3 wo, optix::float3 wi) const {
        // Return no contribution if viewed from behind or the light is on the backside.
        if (wo.z < 0.000001f || wi.z < 0.000001f)
            return BSDFResponse::none();
        
        BSDFResponse transmission = m_transmission_shading_model.evaluate_with_PDF(wo, wi);
        transmission.reflectance *= m_transmission_scale;
        BSDFResponse reflection = BSDFs::GGX_R::evaluate_with_PDF(m_alpha, m_specularity, wo, wi);
        reflection.reflectance *= m_reflection_energy_loss_adjustment * m_layer_opacity;

        // Merge the two bsdf responses.
        BSDFResponse res;
        res.reflectance = transmission.reflectance + reflection.reflectance;
        res.PDF = optix::lerp(transmission.PDF, reflection.PDF, m_reflection_probability);
        return res;
    }

    __inline_all__ BSDFSample sample(optix::float3 wo, optix::float3 random_sample) const {
        // Don't sample material from behind.
        if (wo.z < 0.000001f)
            return BSDFSample::none();

        bool sample_reflection = random_sample.z < m_reflection_probability;
        
        // Sample selected BRDF.
        BSDFSample bsdf_sample;
        if (!sample_reflection) {
            // Scale the random number used for selecting BRDF.
            // TODO Would be simpler if the probability was for transmission.
            random_sample.z = (random_sample.z - m_reflection_probability) / (1 - m_reflection_probability);

            bsdf_sample = m_transmission_shading_model.sample(wo, random_sample);
            bsdf_sample.reflectance *= m_transmission_scale;
            bsdf_sample.PDF *= 1 - m_reflection_probability;
        } else {
            bsdf_sample = BSDFs::GGX_R::sample(m_alpha, m_specularity, wo, make_float2(random_sample));
            bsdf_sample.reflectance *= m_reflection_energy_loss_adjustment * m_layer_opacity;
            bsdf_sample.PDF *= m_reflection_probability;
        }

        // Break if an invalid sample is produced.
        if (!is_PDF_valid(bsdf_sample.PDF))
            return bsdf_sample;
        
        // Compute contribution of the material components not sampled.
        if (sample_reflection) {
            // Evaluate BRDF underneath.
            BSDFResponse transmission_response = m_transmission_shading_model.evaluate_with_PDF(wo, bsdf_sample.direction);
            bsdf_sample.reflectance += transmission_response.reflectance * m_transmission_scale;
            bsdf_sample.PDF += (1 - m_reflection_probability) * transmission_response.PDF;
        } else {
            // Evaluate reflection layer.
            BSDFResponse reflection_response = BSDFs::GGX_R::evaluate_with_PDF(m_alpha, m_specularity, wo, bsdf_sample.direction);
            bsdf_sample.reflectance += reflection_response.reflectance * m_reflection_energy_loss_adjustment * m_layer_opacity;
            bsdf_sample.PDF += m_reflection_probability * reflection_response.PDF;
        }

        return bsdf_sample;
    }

    __inline_all__ optix::float3 rho(float abs_cos_theta_o) const {
        return compute_reflection_rho(abs_cos_theta_o) + compute_transmission_rho(abs_cos_theta_o);
    }

    __inline_all__ optix::float3 compute_reflection_rho(float abs_cos_theta_o) const {
        float roughness = BSDFs::GGX::roughness_from_alpha(m_alpha);
        return m_reflection_energy_loss_adjustment * m_layer_opacity * SpecularRho::fetch(abs_cos_theta_o, roughness).rho(m_specularity);
    }

    __inline_all__ optix::float3 compute_transmission_rho(float abs_cos_theta_o) const {
        return m_transmission_scale * m_transmission_shading_model.rho(abs_cos_theta_o);
    }
};

} // NS Layers
} // NS ShadingModels
} // NS Shading
} // NS OptiXRenderer

#endif // _OPTIXRENDERER_SHADING_MODELS_LAYERS_GGX_REFLECTION_H_