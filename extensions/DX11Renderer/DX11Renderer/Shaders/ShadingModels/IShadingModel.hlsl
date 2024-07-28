// Shading model interface.
// ------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _DX11_RENDERER_SHADERS_SHADING_MODELS_SHADING_MODEL_INTERFACE_H_
#define _DX11_RENDERER_SHADERS_SHADING_MODELS_SHADING_MODEL_INTERFACE_H_

#include <LightSources.hlsl>

namespace ShadingModels {

// ------------------------------------------------------------------------------------------------
// Shading model interface.
// ------------------------------------------------------------------------------------------------
interface IShadingModel {
    float3 evaluate(float3 wo, float3 wi);

    // Evaluate the material lit by a sphere light.
    float3 evaluate_sphere_light(float3 wo, SphereLight light, float ambient_visibility);

    // Evaluate the material lit by an IBL.
    float3 evaluate_IBL(float3 wo, float3 normal, float ambient_visibility);
};

} // NS ShadingModels

#endif // _DX11_RENDERER_SHADERS_SHADING_MODELS_SHADING_MODEL_INTERFACE_H_