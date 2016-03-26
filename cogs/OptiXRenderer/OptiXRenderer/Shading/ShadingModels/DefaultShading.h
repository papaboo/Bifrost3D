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
        return incident_specular + (1.0f - incident_specular) * pow(1.0f - abs_cos_theta, 5.0f);
    }

    __inline_all__ DefaultShading(const Material& material)
        : m_material(material) {}

    __inline_all__ optix::float3 evaluate(optix::float3 wo, optix::float3 wi) {
        bool is_same_hemisphere = wi.z * wo.z >= 0.0f;
        if (!is_same_hemisphere)
            return optix::make_float3(0.0f);

        // Flip directions if on the backside of the material.
        if (wi.z < 0.0f) {
            wi.z = -wi.z;
            wo.z = -wo.z;
        }

        optix::float3 halfway = optix::normalize(wo + wi);
        float fresnel = schlick_fresnel(0.04f, halfway.z);
        float specular = fresnel * BSDFs::GGX::evaluate(m_material.base_roughness, wo, wi, halfway);
        optix::float3 diffuse = (1.0f - fresnel) * BSDFs::Lambert::evaluate(m_material.base_color);
        return diffuse + specular;
    }

    __inline_all__ BSDFSample naive_sample(const optix::float3& wo, const optix::float3& random_sample) {
        
        using namespace optix;

        BSDFSample bsdf_sample;
        
        // 50/50 split between the two BSDFs. TODO Base split on BSDF tints.
        bool sample_glossy = random_sample.z < 0.5f;
        
        if (sample_glossy) {
            float3 glossy_tint = make_float3(1.0f);
            bsdf_sample = BSDFs::GGX::sample(glossy_tint, m_material.base_roughness, wo, make_float2(random_sample));
        } else {
            bsdf_sample = BSDFs::Lambert::sample(m_material.base_color, make_float2(random_sample));
        }
        bsdf_sample.PDF *= 0.5f;

        // Apply Fresnel.
        optix::float3 halfway = normalize(wo + bsdf_sample.direction);
        float fresnel = schlick_fresnel(0.04f, halfway.z);

        if (sample_glossy)
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