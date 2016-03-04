// OptiX renderer functions for point lights.
// ---------------------------------------------------------------------------
// Copyright (C) 2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _OPTIXRENDERER_POINT_LIGHT_IMPLEMENTATION_H_
#define _OPTIXRENDERER_POINT_LIGHT_IMPLEMENTATION_H_

#include <OptiXRenderer/Types.h>

namespace OptiXRenderer {
namespace LightSources {

__inline_all__ LightSample sample_radiance(const PointLight& light, const optix::float3& position, optix::float2 random_sample) {
    LightSample light_sample;
    light_sample.direction = light.position - position;
    light_sample.distance = optix::length(light_sample.direction);
    light_sample.direction /= light_sample.distance;
    light_sample.radiance = light.power / (4.0f * M_PIf * light_sample.distance * light_sample.distance);
    light_sample.PDF = 1.0f;
    return light_sample;
}

__inline_all__ optix::float3 evaluate(const PointLight& light, const optix::float3& position, const optix::float3& direction) {
    float inv_divisor = 1.0f / (M_PIf * 4.0f);
    return light.power * inv_divisor;
}

} // NS LightSources
} // NS OptiXRenderer

#endif // _OPTIXRENDERER_POINT_LIGHT_IMPLEMENTATION_H_