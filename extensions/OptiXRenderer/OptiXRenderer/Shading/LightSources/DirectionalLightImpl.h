// OptiX renderer functions for directional lights.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _OPTIXRENDERER_DIRECTIONAL_LIGHT_IMPLEMENTATION_H_
#define _OPTIXRENDERER_DIRECTIONAL_LIGHT_IMPLEMENTATION_H_

#include <OptiXRenderer/Types.h>

namespace OptiXRenderer {
namespace LightSources {

__inline_all__ bool is_delta_light(const DirectionalLight& light) {
    return true;
}

__inline_all__ LightSample sample_radiance(const DirectionalLight& light, optix::float2 random_sample) {
    LightSample sample;
    sample.radiance = light.radiance;
    sample.PDF = 1.0f;
    sample.direction_to_light = -light.direction;
    sample.distance = 1e30f;
    return sample;
}

__inline_all__ float PDF(const DirectionalLight& light, optix::float3 direction_to_light) {
    return 0.0f;
}

__inline_all__ optix::float3 evaluate(const DirectionalLight& light, optix::float3 direction_to_light) {
    return optix::make_float3(0.0f, 0.0f, 0.0f);
}

// ------------------------------------------------------------------------------------------------
// Functions with generalized parameters.
// ------------------------------------------------------------------------------------------------

__inline_dev__ LightSample sample_radiance(const DirectionalLight& light, optix::float3 lit_position, optix::float2 random_sample) {
    return sample_radiance(light, random_sample);
}

__inline_dev__ float PDF(const DirectionalLight& light, optix::float3 lit_position, optix::float3 direction_to_light) {
    return PDF(light, direction_to_light);
}

__inline_dev__ optix::float3 evaluate(const DirectionalLight& light, optix::float3 lit_position, optix::float3 direction_to_light) {
    return evaluate(light, direction_to_light);
}

} // NS LightSources
} // NS OptiXRenderer

#endif // _OPTIXRENDERER_DIRECTIONAL_LIGHT_IMPLEMENTATION_H_