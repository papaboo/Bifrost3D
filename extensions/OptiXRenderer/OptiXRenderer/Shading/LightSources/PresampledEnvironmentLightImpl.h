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

namespace OptiXRenderer::LightSources {

__inline_dev__ bool is_delta_light(const PresampledEnvironmentLight& light) {
    return false;
}

__inline_dev__ LightSample sample_radiance(const PresampledEnvironmentLight& light, optix::float2 random_sample) {
    int index = random_sample.x * light.sample_count;
    LightSample sample = rtBufferId<LightSample, 1>(light.samples_ID)[index];
    sample.radiance *= light.tint;
    return sample;
}

__inline_dev__ LightResponse evaluate_with_PDF(const PresampledEnvironmentLight& light, optix::float3 direction_to_light) {
    optix::float2 uv = direction_to_latlong_texcoord(direction_to_light);
    float sin_theta = sqrtf(1.0f - direction_to_light.y * direction_to_light.y);
    float pdf = optix::rtTex2D<float>(light.per_pixel_PDF_ID, uv.x, uv.y) / sin_theta;
    PDF checked_PDF = sin_theta == 0.0f ? PDF::delta_dirac(0) : pdf;

    RGB radiance = light.tint * to_rgb(optix::make_float3(optix::rtTex2D<optix::float4>(light.environment_map_ID, uv.x, uv.y)));

    return { radiance, checked_PDF };
}

} // NS OptiXRenderer::LightSources

#endif // _OPTIXRENDERER_PRESAMPLED_ENVIRONMENT_LIGHT_IMPLEMENTATION_H_