// OptiX renderer functions for presampled environment lights.
// ------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License. 
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _OPTIXRENDERER_PRESAMPLED_ENVIRONMENT_LIGHT_IMPLEMENTATION_H_
#define _OPTIXRENDERER_PRESAMPLED_ENVIRONMENT_LIGHT_IMPLEMENTATION_H_

#include <OptiXRenderer/Types.h>
#include <OptiXRenderer/Utils.h>

namespace OptiXRenderer {
namespace LightSources {

__inline_dev__ bool is_delta_light(const PresampledEnvironmentLight& light) {
    return false;
}

__inline_dev__ LightSample sample_radiance(const PresampledEnvironmentLight& light, optix::float2 random_sample) {
    int index = random_sample.x * light.sample_count;
    return rtBufferId<LightSample, 1>(light.samples_ID)[index];
}

__inline_dev__ float PDF(const PresampledEnvironmentLight& light, optix::float3 direction_to_light) {
    optix::float2 uv = direction_to_latlong_texcoord(direction_to_light);
    float sin_theta = sqrtf(1.0f - direction_to_light.y * direction_to_light.y);
    float PDF = optix::rtTex2D<float>(light.per_pixel_PDF_ID, uv.x, uv.y) / sin_theta;
    return sin_theta == 0.0f ? 0.0f : PDF;
}

__inline_dev__ optix::float3 evaluate(const PresampledEnvironmentLight& light, optix::float3 direction_to_light) {
    optix::float2 uv = direction_to_latlong_texcoord(direction_to_light);
    return optix::make_float3(optix::rtTex2D<optix::float4>(light.environment_map_ID, uv.x, uv.y));
}

// ------------------------------------------------------------------------------------------------
// Functions with generalized parameters.
// ------------------------------------------------------------------------------------------------

__inline_dev__ LightSample sample_radiance(const PresampledEnvironmentLight& light, optix::float3 lit_position, optix::float2 random_sample) {
    return sample_radiance(light, random_sample);
}

__inline_dev__ float PDF(const PresampledEnvironmentLight& light, optix::float3 lit_position, optix::float3 direction_to_light) {
    return PDF(light, direction_to_light);
}

__inline_dev__ optix::float3 evaluate(const PresampledEnvironmentLight& light, optix::float3 lit_position, optix::float3 direction_to_light) {
    return evaluate(light, direction_to_light);
}

} // NS LightSources
} // NS OptiXRenderer

#endif // _OPTIXRENDERER_PRESAMPLED_ENVIRONMENT_LIGHT_IMPLEMENTATION_H_