// OptiX renderer default shading model.
// ---------------------------------------------------------------------------
// Copyright (C) 2015-2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _OPTIXRENDERER_SHADING_MODEL_DEFAULT_SHADING_H_
#define _OPTIXRENDERER_SHADING_MODEL_DEFAULT_SHADING_H_

#include <OptiXRenderer/Shading/BSDFs/Lambert.h>
#include <OptiXRenderer/Shading/BSDFs/GGX.h>
#include <OptiXRenderer/Utils.h>

#if GPU_DEVICE
rtTextureSampler<float2, 2> ggx_with_fresnel_rho_texture;
#else
#include <Cogwheel/Assets/Shading/Fittings.h>
#endif

namespace OptiXRenderer {
namespace Shading {
namespace ShadingModels {

// ---------------------------------------------------------------------------
// The default shading material.
// Default shading consist of a diffuse base with a specular layer on top.
// The diffuse tint is weighted by contribution, or rho, of the specular term.
// Future work:
// * Reintroduce the Helmholtz reciprocity.
//   Basically the diffuse tint should depend on rho from both wo.z and wi.z. Perhaps average specular rho before mutiplying onto diffuse tint.
// ---------------------------------------------------------------------------
class DefaultShading {
private:
    optix::float3 m_diffuse_tint;
    float m_roughness;
    optix::float3 m_specularity;

    struct PrecomputedSpecularRho { 
        float base, full;
        __inline_all__ float rho(float specularity) { return optix::lerp(base, full, specularity); }
        __inline_all__ optix::float3 rho(optix::float3 specularity) { 
            return { rho(specularity.x), rho(specularity.y), rho(specularity.z) };
        }
    };

    __inline_all__ static PrecomputedSpecularRho fetch_specular_rho(float abs_cos_theta, float roughness) {
#if GPU_DEVICE
        float2 specular_rho = tex2D(ggx_with_fresnel_rho_texture, abs_cos_theta, roughness);
        return { specular_rho.x, specular_rho.y };
#else
        return { Cogwheel::Assets::Shading::Rho::sample_GGX_with_fresnel(abs_cos_theta, roughness),
                 Cogwheel::Assets::Shading::Rho::sample_GGX(abs_cos_theta, roughness) };
#endif
    }

    __inline_all__ static optix::float3 compute_specular_rho(optix::float3 specularity, float abs_cos_theta, float roughness) {
        auto precomputed_specular_rho = fetch_specular_rho(abs_cos_theta, roughness);
        return precomputed_specular_rho.rho(specularity);
    }

    __inline_all__ static float compute_specular_probability(optix::float3 diffuse_rho, optix::float3 specular_rho) {
        float diffuse_weight = sum(diffuse_rho);
        float specular_weight = sum(specular_rho);
        return specular_weight / (diffuse_weight + specular_weight);
    }

    // Computes the specularity of the specular microfacet and the tint of the diffuse reflection.
    // * The specularity is a linear interpolation of the dielectric specularity and the conductor/metal specularity.
    //   The dielectric specularity is found by multiplying the materials specularity by 0.08, as is described on page 8 of Physically-Based Shading at Disney.
    //   The conductor specularity is simply the tint of the material, as the tint describes the color of the metal when viewed head on.
    // * A multiple scattering microfacet is approximated from the principle that white-is-white and energy lost 
    //   from not simulating multiple scattering events can be computed as 1 - white_specular_rho = energy_lost. 
    //   The light scattered from the 2nd, 3rd and so on scattering event is assumed to be diffuse and its rho/tint is energy_lost * specularity.
    // * The dielectric diffuse tint is computed as tint * (1 - specular_rho) and the metallic diffuse tint is black. 
    //   The diffuse tint of the materials is found by a linear interpolation of the two based on metallicness.
    // * The metallic parameter defines an interpolation between a dielectric material and a conductor material. 
    //   As both materials are described as an independent combination of diffuse and microfacet BRDFs, 
    //   the interpolation between the materials can be computed by simply interpolating the material parameters, 
    //   ie. lerp(evaluate(dielectric, ...), evaluate(conductor, ...), metallic) can be expressed as evaluate(lerp(dielectric, conductor, metallic), ...)
    __inline_all__ static void compute_tints(const optix::float3& tint, float roughness, float specularity, float metallic, float abs_cos_theta,
                                             optix::float3& diffuse_tint, optix::float3& base_specularity) {
        using namespace optix;

        // Specularity
        float dielectric_specularity = specularity * 0.08f; // See Physically-Based Shading at Disney bottom of page 8.
        float3 conductor_specularity = tint;
        base_specularity = lerp(make_float3(dielectric_specularity), conductor_specularity, metallic);

        // Specular directional-hemispherical reflectance function.
        auto precomputed_specular_rho = fetch_specular_rho(abs_cos_theta, roughness);
        float dielectric_specular_rho = precomputed_specular_rho.rho(dielectric_specularity);
        float3 conductor_specular_rho = precomputed_specular_rho.rho(conductor_specularity);

        // Dielectric tint.
        float dielectric_lossless_specular_rho = dielectric_specular_rho / precomputed_specular_rho.full;
        float3 dielectric_tint = tint * (1.0f - dielectric_lossless_specular_rho);

        // Microfacet specular interreflection.
        float3 specular_rho = lerp(make_float3(dielectric_specular_rho), conductor_specular_rho, metallic);
        float3 lossless_specular_rho = specular_rho / precomputed_specular_rho.full;
        float3 specular_multiple_scattering_rho = lossless_specular_rho - specular_rho;

        // Combining dielectric diffuse tint with specular scattering.
        diffuse_tint = dielectric_tint * (1.0f - metallic) + specular_multiple_scattering_rho;
    }

public:

    __inline_all__ DefaultShading(const Material& material, float abs_cos_theta)
        : m_roughness(material.roughness) { 
        compute_tints(material.tint, m_roughness, material.specularity, material.metallic, abs_cos_theta,
                      m_diffuse_tint, m_specularity);
    }

#if GPU_DEVICE
    __inline_all__ DefaultShading(const Material& material, float abs_cos_theta, optix::float2 texcoord)
        : m_roughness(material.roughness) {
        // Material tint
        float3 tint = material.tint;
        if (material.tint_texture_ID)
            tint *= make_float3(optix::rtTex2D<optix::float4>(material.tint_texture_ID, texcoord.x, texcoord.y));

        compute_tints(tint, m_roughness, material.specularity, material.metallic, abs_cos_theta,
                      m_diffuse_tint, m_specularity);
    }
#endif

    __inline_all__ static float coverage(const Material& material, optix::float2 texcoord) {
        float coverage = material.coverage;
#if GPU_DEVICE
        if (material.coverage_texture_ID)
            coverage *= optix::rtTex2D<float>(material.coverage_texture_ID, texcoord.x, texcoord.y);
#endif
        return coverage;
    }

    __inline_all__ optix::float3 evaluate(optix::float3 wo, optix::float3 wi) const {
        using namespace optix;

        bool is_same_hemisphere = wi.z * wo.z >= 0.00000001f;
        if (!is_same_hemisphere)
            return make_float3(0.0f);

        // Flip directions if on the backside of the material.
        if (wo.z < 0.0f) {
            wi.z = -wi.z;
            wo.z = -wo.z;
        }

        float ggx_alpha = BSDFs::GGX::alpha_from_roughness(m_roughness);
        optix::float3 specular = BSDFs::GGX::evaluate(ggx_alpha, m_specularity, wo, wi);
        optix::float3 diffuse = BSDFs::Lambert::evaluate(m_diffuse_tint, wo, wi);
        return diffuse + specular;
    }

    __inline_all__ float PDF(const optix::float3& wo, const optix::float3& wi) const {
        using namespace optix;

        float abs_cos_theta = abs(wo.z);
        optix::float3 specular_rho = compute_specular_rho(m_specularity, abs_cos_theta, m_roughness);
        float specular_probability = compute_specular_probability(m_diffuse_tint, specular_rho);

        // Merge PDFs based on the specular probability.
        float diffuse_PDF = BSDFs::Lambert::PDF(wo, wi);
        float ggx_alpha = BSDFs::GGX::alpha_from_roughness(m_roughness);
        float specular_PDF = BSDFs::GGX::PDF(ggx_alpha, wo, normalize(wo + wi));
        return lerp(diffuse_PDF, specular_PDF, specular_probability);
    }

    __inline_all__ BSDFResponse evaluate_with_PDF(optix::float3 wo, optix::float3 wi) const {
        using namespace optix;

        bool is_same_hemisphere = wi.z * wo.z >= 0.00000001f;
        if (!is_same_hemisphere)
            return BSDFResponse::none();

        // Flip directions if on the backside of the material.
        if (wo.z < 0.0f) {
            wi.z = -wi.z;
            wo.z = -wo.z;
        }

        const float ggx_alpha = BSDFs::GGX::alpha_from_roughness(m_roughness);
        BSDFResponse specular_eval = BSDFs::GGX::evaluate_with_PDF(ggx_alpha, m_specularity, wo, wi);
        BSDFResponse diffuse_eval = BSDFs::Lambert::evaluate_with_PDF(m_diffuse_tint, wo, wi);

        BSDFResponse res;
        res.weight = diffuse_eval.weight + specular_eval.weight;

        float abs_cos_theta = wo.z;
        optix::float3 specular_rho = compute_specular_rho(m_specularity, abs_cos_theta, m_roughness);
        const float specular_probability = compute_specular_probability(m_diffuse_tint, specular_rho);
        res.PDF = lerp(diffuse_eval.PDF, specular_eval.PDF, specular_probability);

        return res;
    }

    __inline_all__ BSDFSample sample_one(const optix::float3& wo, const optix::float3& random_sample) const {
        using namespace optix;

        // Sample BSDFs based on the contribution of each BRDF.
        float abs_cos_theta = abs(wo.z);
        optix::float3 specular_rho = compute_specular_rho(m_specularity, abs_cos_theta, m_roughness);
        float specular_probability = compute_specular_probability(m_diffuse_tint, specular_rho);
        bool sample_specular = random_sample.z < specular_probability;

        // Sample selected BRDF.
        BSDFSample bsdf_sample;
        if (sample_specular) {
            float alpha = BSDFs::GGX::alpha_from_roughness(m_roughness);
            bsdf_sample = BSDFs::GGX::sample(alpha, m_specularity, wo, make_float2(random_sample));
            bsdf_sample.PDF *= specular_probability;
        } else {
            bsdf_sample = BSDFs::Lambert::sample(m_diffuse_tint, make_float2(random_sample));
            bsdf_sample.PDF *= (1.0f - specular_probability);
        }

        return bsdf_sample;
    }

    __inline_all__ BSDFSample sample_all(const optix::float3& wo, const optix::float3& random_sample) const {
        using namespace optix;

        // Sample BSDFs based on the contribution of each BRDF.
        float abs_cos_theta = abs(wo.z);
        optix::float3 specular_rho = compute_specular_rho(m_specularity, abs_cos_theta, m_roughness);
        float specular_probability = compute_specular_probability(m_diffuse_tint, specular_rho);
        bool sample_specular = random_sample.z < specular_probability;

        const float ggx_alpha = BSDFs::GGX::alpha_from_roughness(m_roughness);

        // Sample selected BRDF.
        BSDFSample bsdf_sample;
        if (sample_specular) {
            bsdf_sample = BSDFs::GGX::sample(ggx_alpha, m_specularity, wo, make_float2(random_sample));
            bsdf_sample.PDF *= specular_probability;
        } else {
            bsdf_sample = BSDFs::Lambert::sample(m_diffuse_tint, make_float2(random_sample));
            bsdf_sample.PDF *= (1.0f - specular_probability);
        }

        // Apply fresnel and compute contribution of the rest of the material.
        if (sample_specular) {
            // Evaluate diffuse layer as well.
            BSDFResponse diffuse_response = BSDFs::Lambert::evaluate_with_PDF(m_diffuse_tint, wo, bsdf_sample.direction);
            bsdf_sample.weight += diffuse_response.weight;
            bsdf_sample.PDF += (1.0f - specular_probability) * diffuse_response.PDF;
        } else {
            // Evaluate specular layer as well.
            BSDFResponse glossy_response = BSDFs::GGX::evaluate_with_PDF(ggx_alpha, m_specularity, wo, bsdf_sample.direction);
            bsdf_sample.weight += glossy_response.weight;
            bsdf_sample.PDF += specular_probability * glossy_response.PDF;
        }

        return bsdf_sample;
    }

    // Estimate the directional-hemispherical reflectance function.
    __inline_dev__ optix::float3 rho(float abs_cos_theta) const {
        optix::float3 specular_rho = compute_specular_rho(m_specularity, abs_cos_theta, m_roughness);
        return m_diffuse_tint + specular_rho;
    }
};

} // NS ShadingModels
} // NS Shading
} // NS OptiXRenderer

#endif // _OPTIXRENDERER_SHADING_MODEL_DEFAULT_SHADING_H_