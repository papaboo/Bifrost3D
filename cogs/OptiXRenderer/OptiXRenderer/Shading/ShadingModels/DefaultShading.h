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

#define GGXUsed BSDFs::GGXWithVNDF

namespace OptiXRenderer {
namespace Shading {
namespace ShadingModels {

// ---------------------------------------------------------------------------
// The default shading material.
// Default shading consist of a diffuse base with a specular layer on top.
// The diffuse tint is weighted by contribution, or rho, of the specular term.
// Future work:
// * Perfectly specular metals' diffuse tint should be 0.
// * Reintroduce the Helmholtz reciprocity.
//   Basically the diffuse tint should depend on rho from both wo.z and wi.z.
// * All materials are white at grazing angles, even metals.
//   http://bitsquid.blogspot.dk/2017/07/validating-materials-and-lights-in.html
//   This will make it harder to precompute rho for metals.
// ---------------------------------------------------------------------------
class DefaultShading {
private:
    const Material& m_material;
    optix::float3 m_tint;

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

    __inline_all__ float coverage(optix::float2 texcoord) const { return coverage(m_material, texcoord); }

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

        float specularity = compute_specularity(m_material.specularity, m_material.metallic);
        float3 specular_tint = lerp(make_float3(1.0f), m_tint, m_material.metallic);
        float3 diffuse_tint = m_tint * (1.0f - compute_specular_rho(specularity, wo.z, m_material.roughness));

        float3 halfway = normalize(wo + wi);
        float ggx_alpha = GGXUsed::alpha_from_roughness(m_material.roughness);
        float fresnel = schlick_fresnel(specularity, dot(wo, halfway));
        float3 specular = specular_tint * GGXUsed::evaluate(ggx_alpha, wo, wi, halfway);
        float3 diffuse = BSDFs::Lambert::evaluate(diffuse_tint, wo, wi);
        return diffuse + specular * fresnel;
    }

    __inline_all__ float PDF(const optix::float3& wo, const optix::float3& wi) const {
        using namespace optix;

        float specularity = compute_specularity(m_material.specularity, m_material.metallic);
        float abs_cos_theta = abs(wo.z);
        float3 specular_tint = lerp(make_float3(1.0f), m_tint, m_material.metallic);
        float specular_rho = compute_specular_rho(specularity, abs_cos_theta, m_material.roughness);
        float3 diffuse_tint = m_tint * (1.0f - specular_rho);

        // Sample BSDFs based on the contribution of each BRDF.
        float specular_probability = compute_specular_probability(diffuse_tint, specular_tint, specular_rho);

        float diffuse_PDF = BSDFs::Lambert::PDF(wo, wi);
        float ggx_alpha = GGXUsed::alpha_from_roughness(m_material.roughness);
        float specular_PDF = GGXUsed::PDF(ggx_alpha, wo, normalize(wo + wi));
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

        float specularity = compute_specularity(m_material.specularity, m_material.metallic);
        float abs_cos_theta = abs(wo.z);
        float3 specular_tint = lerp(make_float3(1.0f), m_tint, m_material.metallic);
        float specular_rho = compute_specular_rho(specularity, abs_cos_theta, m_material.roughness);
        float3 diffuse_tint = m_tint * (1.0f - specular_rho);

        const float ggx_alpha = GGXUsed::alpha_from_roughness(m_material.roughness);
        const float3 halfway = normalize(wo + wi);
        BSDFResponse specular_eval = GGXUsed::evaluate_with_PDF(specular_tint, ggx_alpha, wo, wi, halfway);
        BSDFResponse diffuse_eval = BSDFs::Lambert::evaluate_with_PDF(diffuse_tint, wo, wi);

        BSDFResponse res;
        const float fresnel = schlick_fresnel(specularity, dot(wo, halfway));
        res.weight = diffuse_eval.weight + specular_eval.weight * fresnel;

        const float specular_probability = compute_specular_probability(diffuse_tint, specular_tint, specular_rho);
        res.PDF = lerp(diffuse_eval.PDF, specular_eval.PDF, specular_probability);

        return res;
    }

    __inline_all__ BSDFSample sample_one(const optix::float3& wo, const optix::float3& random_sample) const {
        using namespace optix;

        float specularity = compute_specularity(m_material.specularity, m_material.metallic);
        float abs_cos_theta = abs(wo.z);
        float3 specular_tint = lerp(make_float3(1.0f), m_tint, m_material.metallic);
        float specular_rho = compute_specular_rho(specularity, abs_cos_theta, m_material.roughness);
        float3 diffuse_tint = m_tint * (1.0f - specular_rho);

        // Sample BSDFs based on the contribution of each BRDF.
        float specular_probability = compute_specular_probability(diffuse_tint, specular_tint, specular_rho);
        bool sample_specular = random_sample.z < specular_probability;

        // Sample selected BRDF.
        BSDFSample bsdf_sample;
        if (sample_specular) {
            float alpha = GGXUsed::alpha_from_roughness(m_material.roughness);
            bsdf_sample = GGXUsed::sample(specular_tint, alpha, wo, make_float2(random_sample));
            bsdf_sample.PDF *= specular_probability;
        } else {
            bsdf_sample = BSDFs::Lambert::sample(diffuse_tint, make_float2(random_sample));
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

        float specularity = compute_specularity(m_material.specularity, m_material.metallic);
        float abs_cos_theta = abs(wo.z);
        float3 specular_tint = lerp(make_float3(1.0f), m_tint, m_material.metallic);
        float specular_rho = compute_specular_rho(specularity, abs_cos_theta, m_material.roughness);
        float3 diffuse_tint = m_tint * (1.0f - specular_rho);

        // Sample BSDFs based on the contribution of each BRDF.
        float specular_probability = compute_specular_probability(diffuse_tint, specular_tint, specular_rho);
        bool sample_specular = random_sample.z < specular_probability;

        const float ggx_alpha = GGXUsed::alpha_from_roughness(m_material.roughness);

        // Sample selected BRDF.
        BSDFSample bsdf_sample;
        if (sample_specular) {
            bsdf_sample = GGXUsed::sample(specular_tint, ggx_alpha, wo, make_float2(random_sample));
            bsdf_sample.PDF *= specular_probability;
        } else {
            bsdf_sample = BSDFs::Lambert::sample(diffuse_tint, make_float2(random_sample));
            bsdf_sample.PDF *= (1.0f - specular_probability);
        }

        // Compute Fresnel.
        optix::float3 halfway = normalize(wo + bsdf_sample.direction);
        float fresnel = schlick_fresnel(specularity, dot(wo, halfway));

        // Apply fresnel and compute contribution of the rest of the material.
        if (sample_specular) {
            bsdf_sample.weight *= fresnel;

            // Evaluate diffuse layer as well.
            BSDFResponse diffuse_response = BSDFs::Lambert::evaluate_with_PDF(diffuse_tint, wo, bsdf_sample.direction);
            bsdf_sample.weight += (1.0f - fresnel) * diffuse_response.weight;
            bsdf_sample.PDF += (1.0f - specular_probability) * diffuse_response.PDF;
        } else {
            bsdf_sample.weight *= (1.0f - fresnel);

            // Evaluate specular layer as well.
            BSDFResponse glossy_response = GGXUsed::evaluate_with_PDF(specular_tint, ggx_alpha, wo, bsdf_sample.direction, halfway);
            bsdf_sample.weight += fresnel * glossy_response.weight;
            bsdf_sample.PDF += specular_probability * glossy_response.PDF;
        }

        return bsdf_sample;
    }

    // Apply the shading model to the IBL.
    __inline_dev__ optix::float3 IBL(optix::float3 wi, optix::float3 normal, int ibl_texture) const {
        using namespace optix;

        float specularity = compute_specularity(m_material.specularity, m_material.metallic);
        float abs_cos_theta = abs(dot(wi, normal));
        float specular_rho = compute_specular_rho(specularity, abs_cos_theta, m_material.roughness);
        float3 specular_tint = lerp(make_float3(1.0f, 1.0f, 1.0f), m_tint, m_material.metallic) * specular_rho;
        float3 diffuse_tint = m_tint * (1.0f - specular_rho);

        // float width, height, mip_count;
        // environment_tex.GetDimensions(0, width, height, mip_count);

        float2 diffuse_tc = direction_to_latlong_texcoord(normal);
        float3 diffuse = diffuse_tint; // *environment_tex.SampleLevel(environment_sampler, diffuse_tc, mip_count).rgb;

        float2 specular_tc = direction_to_latlong_texcoord(wi);
        float3 specular = specular_tint; // *environment_tex.SampleLevel(environment_sampler, specular_tc, mip_count * m_roughness).rgb;

        return diffuse + specular;
    }

};

} // NS ShadingModels
} // NS Shading
} // NS OptiXRenderer

#endif // _OPTIXRENDERER_SHADING_MODEL_DEFAULT_SHADING_H_