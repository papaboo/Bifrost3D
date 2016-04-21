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

namespace OptiXRenderer {
namespace Shading {
namespace ShadingModels {

class DefaultShading {
private:
    const Material& m_material;

public:
    // TODO Implement Unreal 4 spherical gaussian Fresnel as well.
    __inline_all__ static float schlick_fresnel(float incident_specular, float abs_cos_theta) {
        return incident_specular + (1.0f - incident_specular) * pow(optix::fmaxf(0.0f, 1.0f - abs_cos_theta), 5.0f);
    }

    __inline_all__ DefaultShading(const Material& material)
        : m_material(material) {}

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
        float3 specular_tint = lerp(make_float3(1.0f), m_material.base_tint, m_material.metallic);
        float alpha = BSDFs::GGX::alpha_from_roughness(m_material.base_roughness);
        float3 specular = specular_tint * (fresnel * BSDFs::GGX::evaluate(alpha, wo, wi, halfway));
        float3 diffuse = (1.0f - fresnel) * BSDFs::Lambert::evaluate(m_material.base_tint);
        return diffuse + specular;
    }

    __inline_all__ BSDFSample sample_one(const optix::float3& wo, const optix::float3& random_sample) const {
        using namespace optix;

        const float3 specular_tint = lerp(make_float3(1.0f), m_material.base_tint, m_material.metallic);

        // Sample BSDFs based on their tinted weight.
        float diffuse_weight = average(m_material.base_tint) * (1.0f - m_material.metallic);
        float specular_weight = average(specular_tint);
        float specular_probability = specular_weight / (diffuse_weight + specular_weight);
        bool sample_specular = random_sample.z < specular_probability;

        // Sample selected BRDF.
        BSDFSample bsdf_sample;
        if (sample_specular) {
            float alpha = BSDFs::GGX::alpha_from_roughness(m_material.base_roughness);
            bsdf_sample = BSDFs::GGX::sample(specular_tint, alpha, wo, make_float2(random_sample));
            bsdf_sample.PDF *= specular_probability;
        } else {
            bsdf_sample = BSDFs::Lambert::sample(m_material.base_tint, make_float2(random_sample));
            bsdf_sample.PDF *= (1.0f - specular_probability);
        }

        // Apply Fresnel.
        optix::float3 halfway = normalize(wo + bsdf_sample.direction);
        float specularity = lerp(m_material.specularity, 1.0f, m_material.metallic);
        float fresnel = schlick_fresnel(specularity, dot(wo, halfway));

        if (sample_specular)
            bsdf_sample.weight *= fresnel;
        else
            bsdf_sample.weight *= (1.0f - fresnel);

        return bsdf_sample;
    }

    __inline_all__ BSDFSample sample_all(const optix::float3& wo, const optix::float3& random_sample) const {
        using namespace optix;

        const float ggx_alpha = BSDFs::GGX::alpha_from_roughness(m_material.base_roughness);
        const float3 specular_tint = lerp(make_float3(1.0f), m_material.base_tint, m_material.metallic);

        // Sample BSDFs based on their tinted weight.
        float diffuse_weight = average(m_material.base_tint) * (1.0f - m_material.metallic);
        float specular_weight = average(specular_tint);
        float specular_probability = specular_weight / (diffuse_weight + specular_weight);
        bool sample_specular = random_sample.z < specular_probability;

        // Sample selected BRDF.
        BSDFSample bsdf_sample;
        if (sample_specular) {
            bsdf_sample = BSDFs::GGX::sample(specular_tint, ggx_alpha, wo, make_float2(random_sample));
            bsdf_sample.PDF *= specular_probability;
        } else {
            bsdf_sample = BSDFs::Lambert::sample(m_material.base_tint, make_float2(random_sample));
            bsdf_sample.PDF *= (1.0f - specular_probability);
        }

        // Compute Fresnel.
        optix::float3 halfway = normalize(wo + bsdf_sample.direction);
        float specularity = lerp(m_material.specularity, 1.0f, m_material.metallic);
        float fresnel = schlick_fresnel(specularity, dot(wo, halfway));

        // Apply fresnel and compute contribution of the rest of the material.
        if (sample_specular) {
            bsdf_sample.weight *= fresnel;

            // Evaluate diffuse layer as well.
            bsdf_sample.weight += (1.0f - fresnel) * BSDFs::Lambert::evaluate(m_material.base_tint, wo, bsdf_sample.direction);
            bsdf_sample.PDF += (1.0f - specular_probability) * BSDFs::Lambert::PDF(wo, bsdf_sample.direction);
        } else {
            bsdf_sample.weight *= (1.0f - fresnel);

            // Evaluate specular layer as well.
            bsdf_sample.weight += fresnel * BSDFs::GGX::evaluate(specular_tint, ggx_alpha, wo, bsdf_sample.direction);
            bsdf_sample.PDF += specular_probability * BSDFs::GGX::PDF(ggx_alpha, wo, bsdf_sample.direction);
        }

        return bsdf_sample;
    }

    __inline_all__ float PDF(const optix::float3& wo, const optix::float3& wi) const {
        using namespace optix;

        const float ggx_alpha = BSDFs::GGX::alpha_from_roughness(m_material.base_roughness);
        const float3 specular_tint = lerp(make_float3(1.0f), m_material.base_tint, m_material.metallic);

        // Sample BSDFs based on their tinted weight.
        float diffuse_weight = average(m_material.base_tint) * (1.0f - m_material.metallic);
        float specular_weight = average(specular_tint);
        float specular_probability = specular_weight / (diffuse_weight + specular_weight);
        float base_probability = (1.0f - specular_probability);

        return base_probability * BSDFs::Lambert::PDF(wo, wi) +
            specular_probability * BSDFs::GGX::PDF(ggx_alpha, wo, wi);
    }

}; // NS DefaultShading

} // NS ShadingModels
} // NS Shading
} // NS OptiXRenderer

#endif // _OPTIXRENDERER_SHADING_MODEL_DEFAULT_SHADING_H_