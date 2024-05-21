// OptiX renderer default shading model.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _OPTIXRENDERER_SHADING_MODEL_DEFAULT_SHADING_H_
#define _OPTIXRENDERER_SHADING_MODEL_DEFAULT_SHADING_H_

#include <OptiXRenderer/Shading/BSDFs/Lambert.h>
#include <OptiXRenderer/Shading/BSDFs/GGX.h>
#include <OptiXRenderer/Shading/ShadingModels/Utils.h>
#include <OptiXRenderer/Utils.h>

namespace OptiXRenderer {
namespace Shading {
namespace ShadingModels {

// ---------------------------------------------------------------------------
// The default shading model.
// Default shading consist of a diffuse base with a specular layer on top.
// The diffuse tint is weighted by contribution, or rho, of the specular term.
// Future work:
// * Reintroduce the Helmholtz reciprocity.
//   Basically the diffuse tint should depend on rho from both wo.z and wi.z. Perhaps average specular rho before multiplying onto diffuse tint.
// ---------------------------------------------------------------------------
#define COAT_SPECULARITY 0.04f
#define USHORT_MAX 65535.0f

class DefaultShading {
private:
    optix::float3 m_diffuse_tint;
    float m_specular_alpha;
    optix::float3 m_specularity;
    float m_specular_scale;
    float m_coat_scale;
    float m_coat_alpha;
    unsigned short m_specular_probability;
    unsigned short m_coat_probability;

    __inline_all__ static float compute_specular_probability(optix::float3 diffuse_rho, optix::float3 specular_rho) {
        float diffuse_weight = sum(diffuse_rho);
        float specular_weight = sum(specular_rho);
        return specular_weight / (diffuse_weight + specular_weight);
    }

    __inline_all__ static float compute_coat_probability(optix::float3 base_rho, float coat_rho) {
        float base_weight = sum(base_rho);
        float coat_weight = 3 * coat_rho;
        return coat_weight / (base_weight + coat_weight);
    }

    // Sets up the diffuse and specular component on the base layer.
    // * The dielectric diffuse tint is computed as tint * dielectric_specular_transmission and the metallic diffuse tint is black. 
    //   The diffuse tint of the materials is found by a linear interpolation of the dielectric and metallic tint based on metalness.
    // * The metallic parameter defines an interpolation between a dielectric material and a conductive material with the given parameters.
    //   As both materials are described as an independent linear combination of diffuse and microfacet BRDFs, 
    //   the interpolation between the materials can be computed by simply interpolating the material parameters, 
    //   ie. lerp(dielectric.evaluate(...), conductor.evaluate(...), metallic) can be expressed as lerp_params(dielectric, conductor, metallic)evaluate(...)
    // * The specularity is a linear interpolation of the dielectric specularity and the conductor/metal specularity.
    //   The dielectric specularity is defined by the material's specularity.
    //   The conductor specularity is simply the tint of the material, as the tint describes the color of the metal when viewed head on.
    // * Microfacet interreflection is approximated from the principle that white-is-white and energy lost 
    //   from not simulating multiple scattering events can be computed as 1 - white_specular_rho = full_specular_interreflection.
    //   The specular interreflection with the given specularity can be found by scaling with rho of the given specularity,
    //   specular_interreflection = full_specular_interreflection / specular_rho.
    __inline_all__ void setup_base_layer(optix::float3 tint, float roughness, float specularity, float metallic, float abs_cos_theta_o) {
        using namespace optix;

        // No coat
        m_coat_scale = 0;
        m_coat_alpha = 0;
        m_coat_probability = 0;

        // Specular reflection alpha
        m_specular_alpha = BSDFs::GGX::alpha_from_roughness(roughness);

        // Specularity
        float dielectric_specularity = specularity;
        float3 conductor_specularity = tint;
        m_specularity = lerp(make_float3(dielectric_specularity), conductor_specularity, metallic);

        // Specular directional-hemispherical reflectance function, rho.
        SpecularRho specular_rho = SpecularRho::fetch(abs_cos_theta_o, roughness);

        // Compensate for lost energy due to multi-scattering.
        // Multiple-scattering microfacet BSDFs with the smith model, Heitz et al., 2016 and 
        // Practical multiple scattering compensation for microfacet models, Emmanuel Turquin, 2018
        // showed that multi-scattered reflectance has roughly the same distribution as single-scattering reflectance.
        // Here we use that result to account for multi-scattering by computing the ratio of lost energy of a fully specular material,
        // and then scaling the specular reflectance by that ratio during evaluation, which increases reflectance to account for energy lost.
        float specular_energy_loss_adjustment = 1.0f / specular_rho.full;
        m_specular_scale = specular_energy_loss_adjustment;

        // Dielectric tint.
        float dielectric_specular_rho = specular_rho.rho(dielectric_specularity);
        float dielectric_lossless_specular_rho = dielectric_specular_rho / specular_rho.full;
        float dielectric_specular_transmission = 1.0f - dielectric_lossless_specular_rho;
        float3 dielectric_tint = tint * dielectric_specular_transmission;
        m_diffuse_tint = dielectric_tint * (1.0f - metallic);
    }

    // Sets up the specular microfacet and the diffuse reflection as described in setup_base_layer().
    // If the coat is enabled then a microfacet coat on top of the base layer is initialized as well.
    // * The base roughness is scaled by the coat roughness to simulate how light would scatter through the coat and 
    //   perceptually widen the highlight of the underlying material.
    // * The transmission through the coat is computed and baked into the diffuse tint.
    //   The contribution from coat interreflection is then added to the diffuse tint.
    // * The transmission cannot be baked into the specularity, as specularity only represents the contribution from the specular layer 
    //   when viewed head on and we need the transmission to affect the reflectance at grazing angles as well.
    //   Instead we store the transmission and scale the base layer's specular contribution by it when evaluating or sampling.
    __inline_all__ void setup_shading(optix::float3 tint, float roughness, float specularity, float metallic, float coat_scale, float coat_roughness, float abs_cos_theta_o,
                                      float& coat_single_bounce_rho) {
        using namespace optix;

        float coat_modulated_roughness = modulate_roughness_under_coat(roughness, coat_roughness);
        roughness = lerp(roughness, coat_modulated_roughness, coat_scale);

        setup_base_layer(tint, roughness, specularity, metallic, abs_cos_theta_o);

        coat_single_bounce_rho = 0.0f;
        if (coat_scale > 0) {
            // Clear coat with fixed index of refraction of 1.5 / specularity of 0.04, representative of polyurethane and glass.
            m_coat_alpha = BSDFs::GGX::alpha_from_roughness(coat_roughness);
            SpecularRho coat_rho = SpecularRho::fetch(abs_cos_theta_o, coat_roughness);
            coat_single_bounce_rho = coat_scale * coat_rho.rho(COAT_SPECULARITY);
            float lossless_coat_rho = coat_single_bounce_rho / coat_rho.full;
            float coat_transmission = 1.0f - lossless_coat_rho;

            // Compensate for lost energy due to multi-scattering.
            float coat_energy_loss_adjustment = 1.0f / coat_rho.full;
            m_coat_scale = coat_energy_loss_adjustment * coat_scale;

            // Scale specular and diffuse component by the coat transmission.
            m_specular_scale *= coat_transmission;
            m_diffuse_tint *= coat_transmission;
        }
    }

    __inline_all__ void setup_sampling_probabilities(float abs_cos_theta_o, float coat_rho) {
        // Compute the probability of sampling the specular layer instead of the diffuse layer.
        optix::float3 specular_rho = compute_specular_rho(m_specularity, abs_cos_theta_o, get_roughness()) * m_specular_scale;
        float specular_probability = compute_specular_probability(m_diffuse_tint, specular_rho);
        m_specular_probability = unsigned short(specular_probability * USHORT_MAX + 0.5f);
        // Compute the probability of sampling the coat instead of the base layer.
        float coat_probability = compute_coat_probability(m_diffuse_tint + specular_rho, coat_rho);
        m_coat_probability = unsigned short(coat_probability * USHORT_MAX + 0.5f);
    }

public:

    __inline_all__ DefaultShading(const Material& material, float abs_cos_theta_o) {
        float coat_rho;
        setup_shading(material.tint, material.roughness, material.specularity, material.metallic, material.coat, material.coat_roughness, abs_cos_theta_o, coat_rho);
        setup_sampling_probabilities(abs_cos_theta_o, coat_rho);
    }

#if GPU_DEVICE
    __inline_all__ DefaultShading(const Material& material, float abs_cos_theta_o, optix::float2 texcoord, float min_roughness = 0.0f) {
        using namespace optix;

        // Coat
        float coat_roughness = max(material.coat_roughness, min_roughness);

        // Metallic
        float metallic = material.get_metallic(texcoord);

        // Tint and roughness
        float4 tint_roughness = material.get_tint_roughness(texcoord);
        float3 tint = make_float3(tint_roughness);
        float roughness = tint_roughness.w;
        roughness = max(roughness, min_roughness);

        float coat_rho;
        setup_shading(tint, roughness, material.specularity, metallic, material.coat, coat_roughness, abs_cos_theta_o, coat_rho);
        setup_sampling_probabilities(abs_cos_theta_o, coat_rho);
    }

    __inline_all__ static DefaultShading initialize_with_max_PDF_hint(const Material& material, optix::float2 texcoord, float abs_cos_theta_o, float max_PDF_hint) {
        float min_roughness = GGXMinimumRoughness::from_PDF(abs_cos_theta_o, max_PDF_hint);
        return DefaultShading(material, abs_cos_theta_o, texcoord, min_roughness);
    }
#endif

    __inline_all__ float get_roughness() const { return BSDFs::GGX::roughness_from_alpha(m_specular_alpha); }

    __inline_all__ BSDFResponse evaluate_with_PDF(optix::float3 wo, optix::float3 wi) const {
        using namespace optix;

        // Return no contribution if the light is on the backside.
        if (wo.z < 0.000001f || wi.z < 0.000001f)
            return BSDFResponse::none();

        BSDFResponse specular_eval = BSDFs::GGX_R::evaluate_with_PDF(m_specular_alpha, m_specularity, wo, wi);
        BSDFResponse diffuse_eval = BSDFs::Lambert::evaluate_with_PDF(m_diffuse_tint, wo, wi);

        BSDFResponse res;
        res.reflectance = diffuse_eval.reflectance + specular_eval.reflectance * m_specular_scale;

        float specular_probability = m_specular_probability / USHORT_MAX;
        res.PDF = lerp(diffuse_eval.PDF, specular_eval.PDF, specular_probability);

        if (m_coat_scale > 0) {
            float coat_probability = m_coat_probability / USHORT_MAX;
            BSDFResponse coat_eval = BSDFs::GGX_R::evaluate_with_PDF(m_coat_alpha, COAT_SPECULARITY, wo, wi);
            res.reflectance += m_coat_scale * coat_eval.reflectance;
            res.PDF = lerp(res.PDF, coat_eval.PDF, coat_probability);
        }

        return res;
    }

    // Sample all BSDF based on the contribution of each BRDF.
    __inline_all__ BSDFSample sample(optix::float3 wo, optix::float3 random_sample) const {
        using namespace optix;

        // Don't sample material from behind.
        if (wo.z < 0.000001f)
            return BSDFSample::none();

        float specular_probability = m_specular_probability / USHORT_MAX;
        float coat_probability = m_coat_probability / USHORT_MAX;

        // Pick a BRDF
        bool sample_coat = random_sample.z < coat_probability;
        bool sample_specular = !sample_coat && (random_sample.z - coat_probability) / (1 - coat_probability) < specular_probability;
        bool sample_diffuse = !sample_coat && !sample_specular;

        // Sample selected BRDF.
        BSDFSample bsdf_sample;
        if (sample_diffuse) {
            bsdf_sample = BSDFs::Lambert::sample(m_diffuse_tint, make_float2(random_sample));
            bsdf_sample.PDF *= (1 - coat_probability) * (1 - specular_probability);
        } else if (sample_specular) {
            bsdf_sample = BSDFs::GGX_R::sample(m_specular_alpha, m_specularity, wo, make_float2(random_sample));
            bsdf_sample.reflectance *= m_specular_scale;
            bsdf_sample.PDF *= (1 - coat_probability) * specular_probability;
        } else {
            bsdf_sample = BSDFs::GGX_R::sample(m_coat_alpha, COAT_SPECULARITY, wo, make_float2(random_sample));
            bsdf_sample.reflectance *= m_coat_scale;
            bsdf_sample.PDF *= coat_probability;
        }

        // Break if an invalid sample is produced.
        if (!is_PDF_valid(bsdf_sample.PDF))
            return bsdf_sample;

        // Compute contribution of the material components not sampled.
        if (!sample_diffuse) {
            // Evaluate diffuse layer as well.
            BSDFResponse diffuse_response = BSDFs::Lambert::evaluate_with_PDF(m_diffuse_tint, wo, bsdf_sample.direction);
            bsdf_sample.reflectance += diffuse_response.reflectance;
            bsdf_sample.PDF += (1 - coat_probability) * (1 - specular_probability) * diffuse_response.PDF;
        }
        if (!sample_specular) {
            // Evaluate specular layer as well.
            BSDFResponse specular_response = BSDFs::GGX_R::evaluate_with_PDF(m_specular_alpha, m_specularity, wo, bsdf_sample.direction);
            bsdf_sample.reflectance += specular_response.reflectance * m_specular_scale;
            bsdf_sample.PDF += (1 - coat_probability) * specular_probability * specular_response.PDF;
        }
        if (!sample_coat && m_coat_scale > 0) {
            // Evaluate coat layer as well.
            BSDFResponse coat_response = BSDFs::GGX_R::evaluate_with_PDF(m_coat_alpha, COAT_SPECULARITY, wo, bsdf_sample.direction);
            bsdf_sample.reflectance += m_coat_scale * coat_response.reflectance;
            bsdf_sample.PDF += coat_probability * coat_response.PDF;
        }

        return bsdf_sample;
    }

    // Estimate the directional-hemispherical reflectance function.
    __inline_dev__ optix::float3 rho(float abs_cos_theta) const {
        float specular_roughness = BSDFs::GGX::roughness_from_alpha(m_specular_alpha);
        optix::float3 specular_rho = compute_specular_rho(m_specularity, abs_cos_theta, specular_roughness) * m_specular_scale;

        float coat_roughness = BSDFs::GGX::roughness_from_alpha(m_coat_alpha);
        float single_bounce_coat_rho = m_coat_scale * SpecularRho::fetch(abs_cos_theta, coat_roughness).rho(COAT_SPECULARITY);

        return m_diffuse_tint + specular_rho + single_bounce_coat_rho;
    }
};

} // NS ShadingModels
} // NS Shading
} // NS OptiXRenderer

#endif // _OPTIXRENDERER_SHADING_MODEL_DEFAULT_SHADING_H_