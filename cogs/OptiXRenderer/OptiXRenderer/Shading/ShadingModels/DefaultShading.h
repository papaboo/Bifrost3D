// OptiX renderer default shading model.
// ---------------------------------------------------------------------------
// Copyright (C) 2015-2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _OPTIXRENDERER_SHADING_MODEL_DEFAULT_SHADING_H_
#define _OPTIXRENDERER_SHADING_MODEL_DEFAULT_SHADING_H_

#include <OptiXRenderer/Shading/BSDFs/OrenNayar.h>
#include <OptiXRenderer/Shading/BSDFs/GGX.h>
#include <OptiXRenderer/Utils.h>

#if GPU_DEVICE
rtDeclareVariable(unsigned int, default_shading_rho_texture_ID, , );
#endif

namespace OptiXRenderer {
namespace Shading {
namespace ShadingModels {

// ---------------------------------------------------------------------------
// The default shading material.
// Default shading consist of a diffuse base with a specular layer on top.
// The diffuse and specular contribution is weighted by a fresnel term.
// NOTE:
//  The fresnel term will cause the material to emit more than 100% energy 
//  at grazing angles when the material is very smooth. This is due to 
//  the specular component contributing almost 100%, because Fresnel will 
//  do little to reduce the contribution which is centered on the perfectly 
//  reflected directions. Meanwhile, across the rest ofe the hemisphere, 
//  the diffuse component will contribute by 1 - Fresnel, bringing the total 
//  contribution above 100%. Worst case seems to be 197%.
//  Possible solutions 
//  * Investigating the Disney BRDF and check if they are energy conserving.
//  * Subtract the specular rho from the diffuse contribution 
//    such that the summed contribution equals one or less.
//    The specular rho is currently a tecture though, 
//    which sorts of hampers this solution.
// ---------------------------------------------------------------------------
class DefaultShading {
private:
    const Material& m_material;
    optix::float3 m_tint;

    __inline_all__ static float schlick_fresnel(float incident_specular, float abs_cos_theta) {
        return incident_specular + (1.0f - incident_specular) * pow5(optix::fmaxf(0.0f, 1.0f - abs_cos_theta));
    }

    // Compute BSDF sampling probabilities based on their tinted weight.
    __inline_all__ static float compute_specular_probability(optix::float3 tint, optix::float3 specular_tint, float roughness, float specularity, float abs_cos_theta) {
        float diffuse_weight = sum(tint);
        float specular_weight = sum(specular_tint);

#if GPU_DEVICE
        // On the GPU we have access to a texture with preintegrated rho values.
        optix::float2 specular_diffuse_rho = optix::rtTex2D<optix::float2>(default_shading_rho_texture_ID, abs_cos_theta, roughness);
        float diffuse_rho = specular_diffuse_rho.x * (1.0f - specularity);
        diffuse_weight *= diffuse_rho;
        float specular_rho = optix::lerp(specular_diffuse_rho.y, 1.0f, specularity);
        specular_weight *= specular_rho;
#endif
        return specular_weight / (diffuse_weight + specular_weight);
    }

public:

    __inline_all__ DefaultShading(const Material& material)
        : m_material(material)
        , m_tint(material.tint) { }

#if GPU_DEVICE
    __inline_all__ DefaultShading(const Material& material, optix::float2 texcoord)
        : m_material(material)
    {
        m_tint = material.tint;
        if (material.tint_texture_ID)
            m_tint *= make_float3(optix::rtTex2D<optix::float4>(material.tint_texture_ID, texcoord.x, texcoord.y));
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
        if (wi.z < 0.0f) {
            wi.z = -wi.z;
            wo.z = -wo.z;
        }

        float3 halfway = normalize(wo + wi);
        float specularity = lerp(m_material.specularity, 1.0f, m_material.metallic);
        float fresnel = schlick_fresnel(specularity, dot(wo, halfway));
        float3 specular_tint = lerp(make_float3(1.0f), m_tint, m_material.metallic);
        float ggx_alpha = BSDFs::GGX::alpha_from_roughness(m_material.roughness);
        float3 specular = specular_tint * BSDFs::GGX::evaluate(ggx_alpha, wo, wi, halfway);
        float3 diffuse = BSDFs::OrenNayar::evaluate(m_tint, m_material.roughness, wo, wi);
        return lerp(diffuse, specular, fresnel);
    }

    __inline_all__ float PDF(const optix::float3& wo, const optix::float3& wi) const {
        using namespace optix;

        const float ggx_alpha = BSDFs::GGX::alpha_from_roughness(m_material.roughness);
        const float3 specular_tint = lerp(make_float3(1.0f), m_tint, m_material.metallic);
        float specularity = lerp(m_material.specularity, 1.0f, m_material.metallic);
        float abs_cos_theta = abs(wo.z);

        // Sample BSDFs based on the contribution of each BRDF.
        float specular_probability = compute_specular_probability(m_tint, specular_tint, m_material.roughness, specularity, abs_cos_theta);

        float diffuse_PDF = BSDFs::OrenNayar::PDF(m_material.roughness, wo, wi);
        float specular_PDF = BSDFs::GGX::PDF(ggx_alpha, wo, wi, normalize(wo + wi));
        return lerp(diffuse_PDF, specular_PDF, specular_probability);
    }

    __inline_all__ BSDFResponse evaluate_with_PDF(optix::float3 wo, optix::float3 wi) const {
        using namespace optix;

        bool is_same_hemisphere = wi.z * wo.z >= 0.00000001f;
        if (!is_same_hemisphere)
            return BSDFResponse::none();

        // Flip directions if on the backside of the material.
        if (wi.z < 0.0f) {
            wi.z = -wi.z;
            wo.z = -wo.z;
        }

        const float ggx_alpha = BSDFs::GGX::alpha_from_roughness(m_material.roughness);
        const float3 specular_tint = lerp(make_float3(1.0f), m_tint, m_material.metallic);
        const float specularity = lerp(m_material.specularity, 1.0f, m_material.metallic);
        const float abs_cos_theta = abs(wo.z);

        BSDFResponse res;

        const float3 halfway = normalize(wo + wi);
        BSDFResponse specular_eval = BSDFs::GGX::evaluate_with_PDF(specular_tint, ggx_alpha, wo, wi, halfway);
        BSDFResponse diffuse_eval = BSDFs::OrenNayar::evaluate_with_PDF(m_tint, m_material.roughness, wo, wi);

        const float fresnel = schlick_fresnel(specularity, dot(wo, halfway));
        res.weight = lerp(diffuse_eval.weight, specular_eval.weight, fresnel);

        const float specular_probability = compute_specular_probability(m_tint, specular_tint, m_material.roughness, specularity, abs_cos_theta);
        res.PDF = lerp(diffuse_eval.PDF, specular_eval.PDF, specular_probability);

        return res;
    }

    __inline_all__ BSDFSample sample_one(const optix::float3& wo, const optix::float3& random_sample) const {
        using namespace optix;

        const float3 specular_tint = lerp(make_float3(1.0f), m_tint, m_material.metallic);
        float specularity = lerp(m_material.specularity, 1.0f, m_material.metallic);
        float abs_cos_theta = abs(wo.z);

        // Sample BSDFs based on the contribution of each BRDF.
        float specular_probability = compute_specular_probability(m_tint, specular_tint, m_material.roughness, specularity, abs_cos_theta);
        bool sample_specular = random_sample.z < specular_probability;

        // Sample selected BRDF.
        BSDFSample bsdf_sample;
        if (sample_specular) {
            float alpha = BSDFs::GGX::alpha_from_roughness(m_material.roughness);
            bsdf_sample = BSDFs::GGX::sample(specular_tint, alpha, wo, make_float2(random_sample));
            bsdf_sample.PDF *= specular_probability;
        } else {
            bsdf_sample = BSDFs::OrenNayar::sample(m_tint, m_material.roughness, wo, make_float2(random_sample));
            bsdf_sample.PDF *= (1.0f - specular_probability);
        }

        // Apply Fresnel.
        optix::float3 halfway = normalize(wo + bsdf_sample.direction);
        float fresnel = schlick_fresnel(specularity, dot(wo, halfway));

        if (sample_specular)
            bsdf_sample.weight *= fresnel;
        else
            bsdf_sample.weight *= (1.0f - fresnel);

        return bsdf_sample;
    }

    __inline_all__ BSDFSample sample_all(const optix::float3& wo, const optix::float3& random_sample) const {
        using namespace optix;

        const float ggx_alpha = BSDFs::GGX::alpha_from_roughness(m_material.roughness);
        const float3 specular_tint = lerp(make_float3(1.0f), m_tint, m_material.metallic);
        float specularity = lerp(m_material.specularity, 1.0f, m_material.metallic);
        float abs_cos_theta = abs(wo.z);

        // Sample BSDFs based on the contribution of each BRDF.
        float specular_probability = compute_specular_probability(m_tint, specular_tint, m_material.roughness, specularity, abs_cos_theta);
        bool sample_specular = random_sample.z < specular_probability;

        // Sample selected BRDF.
        BSDFSample bsdf_sample;
        if (sample_specular) {
            bsdf_sample = BSDFs::GGX::sample(specular_tint, ggx_alpha, wo, make_float2(random_sample));
            bsdf_sample.PDF *= specular_probability;
        } else {
            bsdf_sample = BSDFs::OrenNayar::sample(m_tint, m_material.roughness, wo, make_float2(random_sample));
            bsdf_sample.PDF *= (1.0f - specular_probability);
        }

        // Compute Fresnel.
        optix::float3 halfway = normalize(wo + bsdf_sample.direction);
        float fresnel = schlick_fresnel(specularity, dot(wo, halfway));

        // Apply fresnel and compute contribution of the rest of the material.
        if (sample_specular) {
            bsdf_sample.weight *= fresnel;

            // Evaluate diffuse layer as well.
            BSDFResponse diffuse_response = BSDFs::OrenNayar::evaluate_with_PDF(m_tint, m_material.roughness, wo, bsdf_sample.direction);
            bsdf_sample.weight += (1.0f - fresnel) * diffuse_response.weight;
            bsdf_sample.PDF += (1.0f - specular_probability) * diffuse_response.PDF;
        } else {
            bsdf_sample.weight *= (1.0f - fresnel);

            // Evaluate specular layer as well.
            BSDFResponse glossy_response = BSDFs::GGX::evaluate_with_PDF(specular_tint, ggx_alpha, wo, bsdf_sample.direction, halfway);
            bsdf_sample.weight += fresnel * glossy_response.weight;
            bsdf_sample.PDF += specular_probability * glossy_response.PDF;
        }

        return bsdf_sample;
    }
};

} // NS ShadingModels
} // NS Shading
} // NS OptiXRenderer

#endif // _OPTIXRENDERER_SHADING_MODEL_DEFAULT_SHADING_H_