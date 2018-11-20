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
// * All materials are white at grazing angles, even metals.
//   http://bitsquid.blogspot.dk/2017/07/validating-materials-and-lights-in.html
//   Instead of using specular tint and specularity, then use float3 m_specularity.
// ---------------------------------------------------------------------------
class DefaultShading {
private:
    optix::float3 m_diffuse_tint;
    float m_roughness;
    optix::float3 m_specular_tint;
    float m_specularity;

    __inline_all__ static float compute_specularity(float specularity, float metalness) {
        float dielectric_specularity = specularity * 0.08f; // See Physically-Based Shading at Disney bottom of page 8.
        float metal_specularity = specularity * 0.2f + 0.6f;
        return optix::lerp(dielectric_specularity, metal_specularity, metalness);
    } 

    __inline_all__ static float compute_specular_rho(float specularity, float abs_cos_theta, float roughness) {
#if GPU_DEVICE
        float2 specular_rho = tex2D(ggx_with_fresnel_rho_texture, abs_cos_theta, roughness);
        return optix::lerp(specular_rho.x, specular_rho.y, specularity);
#else
        float base_specular_rho = Cogwheel::Assets::Shading::Rho::sample_GGX_with_fresnel(abs_cos_theta, roughness);
        float full_specular_rho = Cogwheel::Assets::Shading::Rho::sample_GGX(abs_cos_theta, roughness);
        return optix::lerp(base_specular_rho, full_specular_rho, specularity);
#endif
    }

    // Compute BSDF sampling probabilities based on their tinted weight.
    __inline_all__ static float compute_specular_probability(optix::float3 tint, optix::float3 specular_tint, float specular_rho) {
        float diffuse_weight = sum(tint);
        float specular_weight = sum(specular_tint) * specular_rho;

        return specular_weight / (diffuse_weight + specular_weight);
    }

public:

    __inline_all__ DefaultShading(const Material& material, float abs_cos_theta)
        : m_roughness(material.roughness) { 

        // Metallic
        float metallic = material.metallic;

        // Specularity
        m_specularity = compute_specularity(material.specularity, metallic);

        // Diffuse and specular tint
        optix::float3 tint = material.tint;

        float specular_rho = compute_specular_rho(m_specularity, abs_cos_theta, m_roughness);
        m_diffuse_tint = tint * (1.0f - specular_rho);
        m_specular_tint = optix::lerp(optix::make_float3(1, 1, 1), tint, metallic);
    }

#if GPU_DEVICE
    __inline_all__ DefaultShading(const Material& material, float abs_cos_theta, optix::float2 texcoord)
        : m_roughness(material.roughness)
    {
        // Metallic
        float metallic = material.metallic;

        // Specularity
        m_specularity = compute_specularity(material.specularity, metallic);

        // Diffuse and specular tint
        optix::float3 tint = material.tint;
        if (material.tint_texture_ID)
            tint *= make_float3(optix::rtTex2D<optix::float4>(material.tint_texture_ID, texcoord.x, texcoord.y));

        float specular_rho = compute_specular_rho(m_specularity, abs_cos_theta, m_roughness);
        m_diffuse_tint = tint * (1.0 - specular_rho);
        m_specular_tint = optix::lerp(optix::make_float3(1, 1, 1), tint, metallic);
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
        float3 specular = m_specular_tint * BSDFs::GGX::evaluate(ggx_alpha, m_specularity, wo, wi);
        float3 diffuse = BSDFs::Lambert::evaluate(m_diffuse_tint, wo, wi);
        return diffuse + specular;
    }

    __inline_all__ float PDF(const optix::float3& wo, const optix::float3& wi) const {
        using namespace optix;

        float abs_cos_theta = abs(wo.z);
        float specular_rho = compute_specular_rho(m_specularity, abs_cos_theta, m_roughness);
        float specular_probability = compute_specular_probability(m_diffuse_tint, m_specular_tint, specular_rho);

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
        BSDFResponse specular_eval = BSDFs::GGX::evaluate_with_PDF(m_specular_tint, ggx_alpha, m_specularity, wo, wi);
        BSDFResponse diffuse_eval = BSDFs::Lambert::evaluate_with_PDF(m_diffuse_tint, wo, wi);

        BSDFResponse res;
        res.weight = diffuse_eval.weight + specular_eval.weight;

        float abs_cos_theta = wo.z;
        float specular_rho = compute_specular_rho(m_specularity, abs_cos_theta, m_roughness);
        const float specular_probability = compute_specular_probability(m_diffuse_tint, m_specular_tint, specular_rho);
        res.PDF = lerp(diffuse_eval.PDF, specular_eval.PDF, specular_probability);

        return res;
    }

    __inline_all__ BSDFSample sample_one(const optix::float3& wo, const optix::float3& random_sample) const {
        using namespace optix;

        // Sample BSDFs based on the contribution of each BRDF.
        float abs_cos_theta = abs(wo.z);
        float specular_rho = compute_specular_rho(m_specularity, abs_cos_theta, m_roughness);
        float specular_probability = compute_specular_probability(m_diffuse_tint, m_specular_tint, specular_rho);
        bool sample_specular = random_sample.z < specular_probability;

        // Sample selected BRDF.
        BSDFSample bsdf_sample;
        if (sample_specular) {
            float alpha = BSDFs::GGX::alpha_from_roughness(m_roughness);
            bsdf_sample = BSDFs::GGX::sample(m_specular_tint, alpha, m_specularity, wo, make_float2(random_sample));
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
        float specular_rho = compute_specular_rho(m_specularity, abs_cos_theta, m_roughness);
        float specular_probability = compute_specular_probability(m_diffuse_tint, m_specular_tint, specular_rho);
        bool sample_specular = random_sample.z < specular_probability;

        const float ggx_alpha = BSDFs::GGX::alpha_from_roughness(m_roughness);

        // Sample selected BRDF.
        BSDFSample bsdf_sample;
        if (sample_specular) {
            bsdf_sample = BSDFs::GGX::sample(m_specular_tint, ggx_alpha, m_specularity, wo, make_float2(random_sample));
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
            BSDFResponse glossy_response = BSDFs::GGX::evaluate_with_PDF(m_specular_tint, ggx_alpha, m_specularity, wo, bsdf_sample.direction);
            bsdf_sample.weight += glossy_response.weight;
            bsdf_sample.PDF += specular_probability * glossy_response.PDF;
        }

        return bsdf_sample;
    }

    // Estimate the directional-hemispherical reflectance function.
    __inline_dev__ optix::float3 rho(float abs_cos_theta) const {
        float specular_rho = compute_specular_rho(m_specularity, abs_cos_theta, m_roughness);
        return m_diffuse_tint + m_specular_tint * specular_rho;
    }
};

} // NS ShadingModels
} // NS Shading
} // NS OptiXRenderer

#endif // _OPTIXRENDERER_SHADING_MODEL_DEFAULT_SHADING_H_