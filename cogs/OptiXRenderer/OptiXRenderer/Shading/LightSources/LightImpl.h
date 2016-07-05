// OptiX renderer functions for directional lights.
// ---------------------------------------------------------------------------
// Copyright (C) 2015-2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _OPTIXRENDERER_LIGHT_IMPLEMENTATION_H_
#define _OPTIXRENDERER_LIGHT_IMPLEMENTATION_H_

#include <OptiXRenderer/Shading/LightSources/DirectionalLightImpl.h>
#include <OptiXRenderer/Shading/LightSources/EnvironmentLightImpl.h>
#include <OptiXRenderer/Shading/LightSources/SphereLightImpl.h>

namespace OptiXRenderer {
namespace LightSources {

__inline_dev__ bool is_delta_light(const Light& light, const optix::float3& position) {
    switch (light.type) {
    case LightTypes::Sphere:
        return is_delta_light(light.sphere, position);
    case LightTypes::Directional:
        return is_delta_light(light.directional, position);
    case LightTypes::Environment:
        return is_delta_light(light.environment, position);
    }
    return false;
}

__inline_dev__ LightSample sample_radiance(const Light& light, const optix::float3& position, optix::float2 random_sample) {
    switch (light.type) {
    case LightTypes::Sphere:
        return sample_radiance(light.sphere, position, random_sample);
    case LightTypes::Directional:
        return sample_radiance(light.directional, position, random_sample);
    case LightTypes::Environment:
        return sample_radiance(light.environment, position, random_sample);
    }
    return LightSample::none();
}

__inline_dev__ float PDF(const Light& light, const optix::float3& lit_position, const optix::float3& direction_to_light) {
    switch (light.type) {
    case LightTypes::Sphere:
        return PDF(light.sphere, lit_position, direction_to_light);
    case LightTypes::Directional:
        return PDF(light.directional, lit_position, direction_to_light);
    case LightTypes::Environment:
        return PDF(light.environment, lit_position, direction_to_light);
    }
    return 0.0f;
}

__inline_dev__ optix::float3 evaluate(const Light& light, const optix::float3& position, const optix::float3& direction_to_light) {
    switch (light.type) {
    case LightTypes::Sphere:
        return evaluate(light.sphere, position, direction_to_light);
    case LightTypes::Directional:
        return evaluate(light.directional, position, direction_to_light);
    case LightTypes::Environment:
        return evaluate(light.environment, position, direction_to_light);
    }
    return optix::make_float3(0.0f);
}

} // NS LightSources
} // NS OptiXRenderer

#endif // _OPTIXRENDERER_LIGHT_IMPLEMENTATION_H_