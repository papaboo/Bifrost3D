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
#include <OptiXRenderer/Shading/LightSources/SphereLightImpl.h>

namespace OptiXRenderer {
namespace LightSources {

__inline_all__ bool is_delta_light(const Light& light, const optix::float3& position) {
    switch (light.flags) {
    case LightFlags::SphereLight:
        return is_delta_light(light.sphere, position);
    case LightFlags::DirectionalLight:
        return is_delta_light(light.directional, position);
    }
    return false;
}

__inline_all__ LightSample sample_radiance(const Light& light, const optix::float3& position, optix::float2 random_sample) {
    switch (light.flags) {
    case LightFlags::SphereLight:
        return sample_radiance(light.sphere, position, random_sample);
    case LightFlags::DirectionalLight:
        return sample_radiance(light.directional, position, random_sample);
    }
    return LightSample::none();
}

__inline_all__ float PDF(const Light& light, const optix::float3& lit_position, const optix::float3& direction_to_light) {
    switch (light.flags) {
    case LightFlags::SphereLight:
        return PDF(light.sphere, lit_position, direction_to_light);
    case LightFlags::DirectionalLight:
        return PDF(light.directional, lit_position, direction_to_light);
    }
    return 0.0f;
}

__inline_all__ optix::float3 evaluate(const Light& light, const optix::float3& position, const optix::float3& direction_to_light) {
    switch (light.flags) {
    case LightFlags::SphereLight:
        return evaluate(light.sphere, position, direction_to_light);
    case LightFlags::DirectionalLight:
        return evaluate(light.directional, position, direction_to_light);
    }
    return optix::make_float3(0.0f);
}

} // NS LightSources
} // NS OptiXRenderer

#endif // _OPTIXRENDERER_LIGHT_IMPLEMENTATION_H_