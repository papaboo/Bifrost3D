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

    __inline_all__ optix::float3 evaluate(optix::float3 wo, optix::float3 wi) {
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
        float fresnel = schlick_fresnel(m_material.specularity, dot(wo, halfway));
        fresnel = lerp(fresnel, 1.0f, m_material.metallic);
        float3 specular_tint = lerp(make_float3(1.0f), m_material.base_color, m_material.metallic);
        float3 specular = specular_tint * (fresnel * BSDFs::GGX::evaluate(m_material.base_roughness, wo, wi, halfway));
        float3 diffuse = (1.0f - fresnel) * BSDFs::Lambert::evaluate(m_material.base_color);
        return diffuse + specular;
    }

    __inline_all__ BSDFSample naive_sample(const optix::float3& wo, const optix::float3& random_sample) {
        using namespace optix;

        BSDFSample bsdf_sample;
        
        // 50/50 split between the two BSDFs. TODO Base split on BSDF tints.
        bool sample_specular = random_sample.z < 0.5f;
        
        if (sample_specular) {
            float3 specular_tint = lerp(make_float3(1.0f), m_material.base_color, m_material.metallic);
            bsdf_sample = BSDFs::GGX::sample(specular_tint, m_material.base_roughness, wo, make_float2(random_sample));
        } else {
            bsdf_sample = BSDFs::Lambert::sample(m_material.base_color, make_float2(random_sample));
        }
        bsdf_sample.PDF *= 0.5f;

        // Apply Fresnel.
        optix::float3 halfway = normalize(wo + bsdf_sample.direction);
        float fresnel = schlick_fresnel(m_material.specularity, dot(wo, halfway));
        fresnel = lerp(fresnel, 1.0f, m_material.metallic);

        if (sample_specular)
            bsdf_sample.weight *= fresnel;
        else
            bsdf_sample.weight *= (1.0f - fresnel);

        return bsdf_sample;
    }

}; // NS DefaultShading

} // NS ShadingModels
} // NS Shading
} // NS OptiXRenderer

#endif // _OPTIXRENDERER_SHADING_MODEL_DEFAULT_SHADING_H_