// OptiX renderer default shading model.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _OPTIXRENDERER_SHADING_MODEL_DEFAULT_SHADING_H_
#define _OPTIXRENDERER_SHADING_MODEL_DEFAULT_SHADING_H_

#include <OptiXRenderer/Shading/BSDFs/OrenNayar.h>
#include <OptiXRenderer/Shading/BSDFs/GGX.h>
#include <OptiXRenderer/Shading/ShadingModels/Utils.h>
#include <OptiXRenderer/Utils.h>

namespace OptiXRenderer {
namespace Shading {
namespace ShadingModels {

// ---------------------------------------------------------------------------
// The default shading model.
// Default shading consist of a diffuse base with a specular layer on top. It can optionally be coated.
// * The metallic parameter defines an interpolation between the evaluation of a dielectric material and the evaluation of a conductive material with the given parameters.
//   However, the same effect can be achieved by interpolating the two materials' diffuse tint and specularity, saving one material evaluation.
// * The specularity is a linear interpolation of the dielectric specularity and the conductor/metal specularity.
//   The dielectric specularity is defined by the material's specularity.
//   The conductor specularity is simply the tint of the material, as the tint describes the color of the metal when viewed head on.
// The diffuse and specular component of the base layer are set up in the following way.
// * The dielectric diffuse tint is computed as tint * dielectric_specular_transmission and the metallic diffuse tint is black. 
//   The diffuse tint of the materials is found by a linear interpolation of the dielectric and metallic tint based on metalness.
// If the coat is enabled then a microfacet coat on top of the base layer is initialized as well.
// * The base roughness is scaled by the coat roughness to simulate how light would scatter through the coat and 
//   perceptually widen the highlight of the underlying material.
// * The transmission through the coat is computed and baked into the diffuse tint and the specular scale
// Future work:
// * Reintroduce the Helmholtz reciprocity. See Revisiting Physically Based Shading at Imageworks for scale by precomputed rho.
// ---------------------------------------------------------------------------
#define USHORT_MAX 65535.0f

class DefaultShading {
private:
    optix::float3 m_diffuse_tint;
    float m_roughness;
    optix::float3 m_specularity;
    float m_specular_scale;
    float m_coat_scale;
    float m_coat_alpha;
    unsigned short m_specular_probability;
    unsigned short m_coat_probability;

    __inline_all__ static float compute_specular_properties(float roughness, float specularity, float scale, float abs_cos_theta_o,
        float& alpha, float& reflection_scale, float& transmission_scale) {
        alpha = BSDFs::GGX::alpha_from_roughness(roughness);
        SpecularRho rho_computation = SpecularRho::fetch(abs_cos_theta_o, roughness);

        // Compensate for lost energy due to multi-scattering and scale by the strength of the specular reflection.
        reflection_scale = scale * rho_computation.energy_loss_adjustment();
        float specular_rho = rho_computation.rho(specularity) * reflection_scale;

        // The amount of energy not reflected by the specular reflection is transmitted further into the material.
        transmission_scale = 1.0f - specular_rho;

        return specular_rho;
    }

    // Sets up the specular microfacet and the diffuse reflection as described in setup_base_layer().
    __inline_all__ void setup_shading(optix::float3 tint, float roughness, float dielectric_specularity, float metallic, float coat_scale, float coat_roughness,
        float cos_theta_o, float& coat_rho) {
        using namespace optix;

        float abs_cos_theta_o = abs(cos_theta_o);

        m_roughness = roughness;
        float3 conductor_specularity = tint;

        // Adjust parameters if coat is enabled.
        if (coat_scale > 0) {
            // Adjust specular reflection roughness
            float coat_modulated_roughness = modulate_roughness_under_coat(roughness, coat_roughness);
            m_roughness = lerp(roughness, coat_modulated_roughness, coat_scale);

            // Adjust base material specularity if coat is enabled
            // The specularity can become NaN if the input specularity is white, which is physically impossible, but doable in the data model.
            if (dielectric_specularity < 1.0f) {
                float coated_dielectric_specularity = adjust_dielectric_specularity_to_exterior_medium(COAT_IOR, dielectric_specularity);
                dielectric_specularity = lerp(dielectric_specularity, coated_dielectric_specularity, coat_scale);
            }

            if (metallic > 0) {
                // Not all extinction coefficients are valid for all specularities and some combinations will result in NANs in the adjusted specularity.
                // To avoid these issues we use zero extinction, which results in the same adjustment as to dielectric specularity.
                float3 metal_extinction_coefficient = optix::make_float3(0, 0, 0);
                float3 coated_conductor_specularity = adjust_conductor_specularity_to_exterior_medium({ COAT_IOR, COAT_IOR, COAT_IOR }, conductor_specularity, metal_extinction_coefficient);
                conductor_specularity = lerp(conductor_specularity, coated_conductor_specularity, coat_scale);

                // The specularity can become NaN if the input specularity is white, which is physically impossible, but doable in the data model.
                // In this case we simply reset the specularity to 1.
                conductor_specularity.x = isnan(conductor_specularity.x) ? 1.0f : conductor_specularity.x;
                conductor_specularity.y = isnan(conductor_specularity.y) ? 1.0f : conductor_specularity.y;
                conductor_specularity.z = isnan(conductor_specularity.z) ? 1.0f : conductor_specularity.z;
            }
        }

        // Dielectric material parameters
        float specular_alpha, dielectric_specular_transmission;
        compute_specular_properties(m_roughness, dielectric_specularity, 1.0f, abs_cos_theta_o,
            specular_alpha, m_specular_scale, dielectric_specular_transmission);
        float3 dielectric_tint = tint * dielectric_specular_transmission;

        // Interpolate between dieletric and conductor parameters based on the metallic parameter.
        // Conductor diffuse component is black, so interpolation amounts to scaling.
        m_specularity = lerp(make_float3(dielectric_specularity), conductor_specularity, metallic);
        m_diffuse_tint = dielectric_tint * (1.0f - metallic);

        // Setup clear coat
        if (coat_scale > 0) {
            // Clear coat with fixed index of refraction of 1.5 / specularity of 0.04, representative of polyurethane and glass.
            float coat_transmission;
            coat_rho = compute_specular_properties(coat_roughness, COAT_SPECULARITY, coat_scale, abs_cos_theta_o,
                m_coat_alpha, m_coat_scale, coat_transmission);

            // Scale specular and diffuse component by the coat transmission.
            m_specular_scale *= coat_transmission;
            m_diffuse_tint *= coat_transmission;
        } else {
            // No coat
            coat_rho = 0;
            m_coat_scale = 0;
            m_coat_alpha = 0;
        }
    }

    __inline_all__ void setup_sampling_probabilities(float abs_cos_theta_o, float coat_rho) {
        float diffuse_rho_sum = sum(diffuse_rho(abs_cos_theta_o));
        float specular_rho_sum = sum(specular_rho(abs_cos_theta_o));
        float coat_rho_sum = 3 * coat_rho;
        float recip_total_rho = 1.0f / (diffuse_rho_sum + specular_rho_sum + coat_rho_sum);

        float specular_probability = specular_rho_sum * recip_total_rho;
        m_specular_probability = unsigned short(specular_probability * USHORT_MAX + 0.5f);

        float coat_probability = coat_rho_sum * recip_total_rho;
        m_coat_probability = unsigned short(coat_probability * USHORT_MAX + 0.5f);
    }

public:

    __inline_all__ DefaultShading(const Material& material, float abs_cos_theta_o) {
        float coat_rho;
        setup_shading(material.tint, material.roughness, material.specularity, material.metallic, material.coat, material.coat_roughness, abs_cos_theta_o, coat_rho);
        setup_sampling_probabilities(abs_cos_theta_o, coat_rho);
    }

#if GPU_DEVICE
    __inline_all__ DefaultShading(const Material& material, optix::float2 texcoord, float4 tint_and_roughness_scale, float abs_cos_theta_o, float min_roughness = 0.0f) {
        using namespace optix;

        // Coat
        float coat_roughness = max(material.coat_roughness, min_roughness);

        // Metallic
        float metallic = material.get_metallic(texcoord);

        // Tint and roughness
        float4 tint_roughness = material.get_tint_roughness(texcoord) * tint_and_roughness_scale;
        float3 tint = make_float3(tint_roughness);
        float roughness = max(tint_roughness.w, min_roughness);

        float coat_rho;
        setup_shading(tint, roughness, material.specularity, metallic, material.coat, coat_roughness, abs_cos_theta_o, coat_rho);
        setup_sampling_probabilities(abs_cos_theta_o, coat_rho);
    }

    __inline_all__ static DefaultShading initialize_with_max_PDF_hint(const Material& material, optix::float2 texcoord, float4 tint_and_roughness_scale, float abs_cos_theta_o, PDF max_PDF_hint) {
        float min_roughness = GGXMinimumRoughness::from_PDF(abs_cos_theta_o, max_PDF_hint);
        return DefaultShading(material, texcoord, tint_and_roughness_scale, abs_cos_theta_o, min_roughness);
    }
#endif

    __inline_all__ float get_roughness() const { return m_roughness; }
    __inline_all__ float get_specular_alpha() const { return BSDFs::GGX::alpha_from_roughness(m_roughness); }
    __inline_all__ optix::float3 get_specularity() const { return m_specularity; }

    __inline_all__ float get_diffuse_probability() const { return 1.0f - (m_specular_probability + m_coat_probability) / USHORT_MAX; }
    __inline_all__ float get_specular_probability() const { return m_specular_probability / USHORT_MAX; }
    __inline_all__ float get_coat_probability() const { return m_coat_probability / USHORT_MAX; }

    __inline_all__ BSDFResponse evaluate_with_PDF(optix::float3 wo, optix::float3 wi) const {
        using namespace optix;

        // Return no contribution if the light is on the backside.
        if (wo.z < 0.000001f || wi.z < 0.000001f)
            return BSDFResponse::none();

        BSDFResponse diffuse_response = BSDFs::OrenNayar::evaluate_with_PDF(m_diffuse_tint, m_roughness, wo, wi);
        BSDFResponse specular_response = BSDFs::GGX_R::evaluate_with_PDF(get_specular_alpha(), m_specularity, wo, wi);
        specular_response.reflectance *= m_specular_scale;

        BSDFResponse response;
        response.reflectance = diffuse_response.reflectance + specular_response.reflectance;

        float diffuse_probability = get_diffuse_probability();
        float specular_probability = get_specular_probability();
        response.PDF = diffuse_response.PDF * diffuse_probability + specular_response.PDF * specular_probability;

        if (m_coat_scale > 0) {
            float coat_probability = get_coat_probability();
            BSDFResponse coat_response = BSDFs::GGX_R::evaluate_with_PDF(m_coat_alpha, COAT_SPECULARITY, wo, wi);
            response.reflectance += m_coat_scale * coat_response.reflectance;
            response.PDF += coat_response.PDF * coat_probability;
        }

        return response;
    }

    // Sample all BSDF based on the contribution of each BRDF.
    __inline_all__ BSDFSample sample(optix::float3 wo, optix::float3 random_sample) const {
        using namespace optix;

        // Don't sample material from behind.
        if (wo.z < 0.000001f)
            return BSDFSample::none();

        float specular_probability = get_specular_probability();
        float coat_probability = get_coat_probability();
        float diffuse_probability = 1 - coat_probability - specular_probability;

        // Pick a BRDF
        bool sample_coat = random_sample.z < coat_probability;
        bool sample_specular = !sample_coat && random_sample.z < (coat_probability + specular_probability);
        bool sample_diffuse = !sample_coat && !sample_specular;

        // Sample selected BRDF.
        BSDFSample bsdf_sample;
        if (sample_diffuse) {
            bsdf_sample = BSDFs::OrenNayar::sample(m_diffuse_tint, m_roughness, wo, make_float2(random_sample));
            bsdf_sample.PDF *= diffuse_probability;
        } else if (sample_specular) {
            bsdf_sample = BSDFs::GGX_R::sample(get_specular_alpha(), m_specularity, wo, make_float2(random_sample));
            bsdf_sample.reflectance *= m_specular_scale;
            bsdf_sample.PDF *= specular_probability;
        } else {
            bsdf_sample = BSDFs::GGX_R::sample(m_coat_alpha, COAT_SPECULARITY, wo, make_float2(random_sample));
            bsdf_sample.reflectance *= m_coat_scale;
            bsdf_sample.PDF *= coat_probability;
        }

        // Break if a sample with an invalid PDF is produced.
        if (bsdf_sample.PDF.invalid_or_delta_dirac())
            return bsdf_sample;

        // Compute contribution of the material components not sampled.
        if (!sample_diffuse) {
            // Evaluate diffuse layer as well.
            BSDFResponse diffuse_response = BSDFs::OrenNayar::evaluate_with_PDF(m_diffuse_tint, m_roughness, wo, bsdf_sample.direction);
            if (diffuse_response.PDF.is_valid_and_not_delta_dirac()) {
                bsdf_sample.reflectance += diffuse_response.reflectance;
                bsdf_sample.PDF += diffuse_response.PDF * diffuse_probability;
            }
        }
        if (!sample_specular) {
            // Evaluate specular layer as well.
            BSDFResponse specular_response = BSDFs::GGX_R::evaluate_with_PDF(get_specular_alpha(), m_specularity, wo, bsdf_sample.direction);
            if (specular_response.PDF.is_valid_and_not_delta_dirac()) {
                bsdf_sample.reflectance += specular_response.reflectance * m_specular_scale;
                bsdf_sample.PDF += specular_response.PDF * specular_probability;
            }
        }
        if (!sample_coat && m_coat_scale > 0) {
            // Evaluate coat layer as well.
            BSDFResponse coat_response = BSDFs::GGX_R::evaluate_with_PDF(m_coat_alpha, COAT_SPECULARITY, wo, bsdf_sample.direction);
            if (coat_response.PDF.is_valid_and_not_delta_dirac()) {
                bsdf_sample.reflectance += m_coat_scale * coat_response.reflectance;
                bsdf_sample.PDF += coat_response.PDF * coat_probability;
            }
        }

        return bsdf_sample;
    }

    // Estimate the directional-hemispherical reflectance function.
    __inline_all__ optix::float3 rho(float abs_cos_theta) const {
        optix::float3 radiance = diffuse_rho(abs_cos_theta) + specular_rho(abs_cos_theta);
        if (m_coat_scale > 0.0f)
            radiance = radiance + coat_rho(abs_cos_theta);
        return radiance;
    }

    __inline_all__ optix::float3 diffuse_rho(float abs_cos_theta) const { return m_diffuse_tint; }
    __inline_all__ optix::float3 specular_rho(float abs_cos_theta) const {
        return SpecularRho::fetch(abs_cos_theta, m_roughness).rho(m_specularity) * m_specular_scale;
    }
    __inline_all__ float coat_rho(float abs_cos_theta) const {
        float coat_roughness = BSDFs::GGX::roughness_from_alpha(m_coat_alpha);
        return SpecularRho::fetch(abs_cos_theta, coat_roughness).rho(COAT_SPECULARITY) * m_coat_scale;
    }
};

} // NS ShadingModels
} // NS Shading
} // NS OptiXRenderer

#endif // _OPTIXRENDERER_SHADING_MODEL_DEFAULT_SHADING_H_