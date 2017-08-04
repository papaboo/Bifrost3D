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
#include <OptiXRenderer/Shading/LightSources/PresampledEnvironmentLightImpl.h>
#include <OptiXRenderer/Shading/LightSources/SphereLightImpl.h>

namespace OptiXRenderer {
namespace LightSources {

__inline_dev__ bool is_delta_light(const Light& light, const optix::float3& position) {
    switch (light.get_type()) {
    case Light::Sphere:
        return is_delta_light(light.sphere, position);
    case Light::Directional:
        return is_delta_light(light.directional);
    case Light::Environment:
        return is_delta_light(light.environment);
    case Light::PresampledEnvironment:
        return is_delta_light(light.presampled_environment);
    }
    return false;
}

__inline_dev__ LightSample sample_radiance(const Light& light, const optix::float3& position, optix::float2 random_sample) {
    switch (light.get_type()) {
    case Light::Sphere:
        return sample_radiance(light.sphere, position, random_sample);
    case Light::Directional:
        return sample_radiance(light.directional, random_sample);
    case Light::Environment:
        return sample_radiance(light.environment, random_sample);
    case Light::PresampledEnvironment:
        return sample_radiance(light.presampled_environment, random_sample);
    }
    return LightSample::none();
}

__inline_dev__ float PDF(const Light& light, const optix::float3& lit_position, const optix::float3& direction_to_light) {
    switch (light.get_type()) {
    case Light::Sphere:
        return PDF(light.sphere, lit_position, direction_to_light);
    case Light::Directional:
        return PDF(light.directional, direction_to_light);
    case Light::Environment:
        return PDF(light.environment, direction_to_light);
    case Light::PresampledEnvironment:
        return PDF(light.presampled_environment, direction_to_light);
    }
    return 0.0f;
}

__inline_dev__ optix::float3 evaluate(const Light& light, const optix::float3& position, const optix::float3& direction_to_light) {
    switch (light.get_type()) {
    case Light::Sphere:
        return evaluate(light.sphere, position, direction_to_light);
    case Light::Directional:
        return evaluate(light.directional, direction_to_light);
    case Light::Environment:
        return evaluate(light.environment, direction_to_light);
    case Light::PresampledEnvironment:
        return evaluate(light.presampled_environment, direction_to_light);
    }
    return optix::make_float3(0.0f);
}

template <typename LightType>
__inline_dev__ optix::float3 evaluate_intersection(const LightType& light, const optix::float3& position, const optix::float3& direction_to_light, 
                                                  float bsdf_PDF, bool next_event_estimated) {
    optix::float3 radiance = evaluate(light, position, ray.direction);

    // bool next_event_estimated = monte_carlo_payload.bounces != 0; // Was next event estimated at previous intersection.
    bool apply_MIS = monte_carlo_payload.bsdf_MIS_PDF > 0.0f;
    if (apply_MIS) {
        // Calculate MIS weight and scale the radiance by it.
        float light_PDF = PDF(light, position, ray.direction);
        float mis_weight = RNG::power_heuristic(bsdf_PDF, light_PDF);
        radiance *= mis_weight;
    } else if (next_event_estimated) // TODO This should superseed the MIS condition.
        // Previous bounce used next event estimation, but did not calculate MIS, so don't apply light contribution.
        radiance = make_float3(0.0f);

    return radiance;
}

} // NS LightSources
} // NS OptiXRenderer

#endif // _OPTIXRENDERER_LIGHT_IMPLEMENTATION_H_