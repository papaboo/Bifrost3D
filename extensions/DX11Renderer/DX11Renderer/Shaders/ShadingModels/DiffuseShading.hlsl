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
    
    // Sets up the diffuse and specular component on the base layer.
    static DiffuseShading create(float3 tint) {
        DiffuseShading shading;
        shading.m_tint = tint;
        return shading;
    }

    // --------------------------------------------------------------------------------------------
    // Evaluations.
    // --------------------------------------------------------------------------------------------
    float3 evaluate(float3 wo, float3 wi) {
        bool is_same_hemisphere = wi.z * wo.z >= 0.0;
        if (!is_same_hemisphere)
            return float3(0.0, 0.0, 0.0);

        // Flip directions if on the backside of the material.
        if (wo.z < 0.0) {
            wi.z = -wi.z;
            wo.z = -wo.z;
        }

        return m_tint * BSDFs::Lambert::evaluate();
    }

    // Evaluate the material lit by a sphere light.
    // Uses evaluation by most representative point internally.
    float3 evaluate_sphere_light(float3 wo, SphereLight light, float ambient_visibility) {
        float3 light_radiance = light.power * rcp(4.0f * PI * dot(light.position, light.position));
        return evaluate_sphere_light_lambert(light, light_radiance, wo, m_tint, ambient_visibility);
    }

    // Apply the shading model to the IBL.
    // TODO Take the current LOD and pixel density into account before choosing sample LOD.
    //      See http://casual-effects.blogspot.dk/2011/08/plausible-environment-lighting-in-two.html 
    //      for how to derive the LOD level for cubemaps.
    float3 evaluate_IBL(float3 wo, float3 normal) {
        float width, height, mip_count;
        environment_tex.GetDimensions(0, width, height, mip_count);

        float2 diffuse_tc = direction_to_latlong_texcoord(normal);
        float3 radiance = m_tint * environment_tex.SampleLevel(environment_sampler, diffuse_tc, mip_count).rgb;

        return radiance;
    }
};

} // NS ShadingModels

#endif // _DX11_RENDERER_SHADERS_SHADING_MODELS_DIFFUSE_SHADING_H_