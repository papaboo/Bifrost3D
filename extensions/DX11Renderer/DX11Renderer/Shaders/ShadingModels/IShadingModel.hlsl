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

    // Evaluate the material lit by an area light.
    float3 evaluate_area_light(LightData light, float3 world_position, float3 wo, float3x3 world_to_shading_TBN, float ambient_visibility);

    // Apply the shading model to the IBL.
    float3 evaluate_IBL(float3 wo, float3 normal);
};

} // NS ShadingModels

#endif // _DX11_RENDERER_SHADERS_SHADING_MODELS_SHADING_MODEL_INTERFACE_H_