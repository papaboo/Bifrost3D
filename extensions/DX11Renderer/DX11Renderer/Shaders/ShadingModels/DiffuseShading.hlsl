// Diffuse shading.
// ------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _DX11_RENDERER_SHADERS_SHADING_MODELS_DIFFUSE_SHADING_H_
#define _DX11_RENDERER_SHADERS_SHADING_MODELS_DIFFUSE_SHADING_H_

#include <BSDFs/Diffuse.hlsl>
#include <ShadingModels/IShadingModel.hlsl>
#include <ShadingModels/Utils.hlsl>

namespace ShadingModels {

// ------------------------------------------------------------------------------------------------
// DiffuseShading shading.
// ------------------------------------------------------------------------------------------------
struct DiffuseShading : IShadingModel {
    float3 m_tint;
    float m_roughness;

    // Set up the shading model.
    static DiffuseShading create(float3 tint, float roughness) {
        DiffuseShading shading;
        shading.m_tint = tint;
        shading.m_roughness = roughness;
        return shading;
    }

    // --------------------------------------------------------------------------------------------
    // Evaluations.
    // --------------------------------------------------------------------------------------------
    float3 evaluate(float3 wo, float3 wi) {
        // Return no contribution if seen or lit from the backside.
        if (wo.z <= 0.0f || wi.z <= 0.0f)
            return float3(0, 0, 0);

        return m_tint * BSDFs::OrenNayar::evaluate(m_roughness, wo, wi);
    }

    // Evaluate the material lit by a sphere light.
    // Not fitted to OrenNayar, so we use a lambertian (roughness = 0) approximation
    float3 evaluate_sphere_light(float3 wo, SphereLight light, float ambient_visibility) {
        // Return no contribution if seen or lit from the backside.
        bool light_below_surface = light.position.z < -light.radius;
        if (wo.z <= 0.0f || light_below_surface)
            return float3(0, 0, 0);

        float3 light_radiance = light.power * rcp(4.0f * PI * dot(light.position, light.position));
        return evaluate_sphere_light_lambert(light, light_radiance, wo, m_tint, ambient_visibility);
    }

    // Evaluate the material lit by an IBL.
    float3 evaluate_IBL(float3 wo, float3 normal, float ambient_visibility) {
        return evaluate_IBL_lambert(wo, normal, m_tint, ambient_visibility);
    }
};

} // NS ShadingModels

#endif // _DX11_RENDERER_SHADERS_SHADING_MODELS_DIFFUSE_SHADING_H_