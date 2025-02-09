// OptiX renderer functions for directional lights.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _OPTIXRENDERER_LIGHT_IMPLEMENTATION_H_
#define _OPTIXRENDERER_LIGHT_IMPLEMENTATION_H_

#include <OptiXRenderer/MonteCarlo.h>
#include <OptiXRenderer/Shading/LightSources/DirectionalLightImpl.h>
#include <OptiXRenderer/Shading/LightSources/EnvironmentLightImpl.h>
#include <OptiXRenderer/Shading/LightSources/PresampledEnvironmentLightImpl.h>
#include <OptiXRenderer/Shading/LightSources/SphereLightImpl.h>
#include <OptiXRenderer/Shading/LightSources/SpotLightImpl.h>

namespace OptiXRenderer {
namespace LightSources {

__inline_dev__ bool is_delta_light(const Light& light, optix::float3 position) {
    switch (light.get_type()) {
    case Light::Sphere:
        return is_delta_light(light.sphere, position);
    case Light::Directional:
        return is_delta_light(light.directional);
    case Light::Environment:
        return is_delta_light(light.environment);
    case Light::PresampledEnvironment:
        return is_delta_light(light.presampled_environment);
    case Light::Spot:
        return is_delta_light(light.spot);
    }
    return false;
}

__inline_dev__ LightSample sample_radiance(const Light& light, optix::float3 position, optix::float2 random_sample) {
    switch (light.get_type()) {
    case Light::Sphere:
        return sample_radiance(light.sphere, position, random_sample);
    case Light::Directional:
        return sample_radiance(light.directional, random_sample);
    case Light::Environment:
        return sample_radiance(light.environment, random_sample);
    case Light::PresampledEnvironment:
        return sample_radiance(light.presampled_environment, random_sample);
    case Light::Spot:
        return sample_radiance(light.spot, position, random_sample);
    }
    return LightSample::none();
}

__inline_dev__ PDF pdf(const Light& light, optix::float3 lit_position, optix::float3 direction_to_light) {
    switch (light.get_type()) {
    case Light::Sphere:
        return pdf(light.sphere, lit_position, direction_to_light);
    case Light::Directional:
        return pdf(light.directional, direction_to_light);
    case Light::Environment:
        return pdf(light.environment, direction_to_light);
    case Light::PresampledEnvironment:
        return pdf(light.presampled_environment, direction_to_light);
    case Light::Spot:
        return pdf(light.spot, lit_position, direction_to_light);
    }
    return PDF::invalid();
}

__inline_dev__ optix::float3 evaluate(const Light& light, optix::float3 position, optix::float3 direction_to_light) {
    switch (light.get_type()) {
    case Light::Sphere:
        return evaluate(light.sphere, position, direction_to_light);
    case Light::Directional:
        return evaluate(light.directional, direction_to_light);
    case Light::Environment:
        return evaluate(light.environment, direction_to_light);
    case Light::PresampledEnvironment:
        return evaluate(light.presampled_environment, direction_to_light);
    case Light::Spot:
        return evaluate(light.spot, position, direction_to_light);
    }
    return optix::make_float3(0.0f);
}

template <typename LightType>
__inline_dev__ optix::float3 evaluate_intersection(const LightType& light, optix::float3 position, optix::float3 direction_to_light, PDF bsdf_PDF) {
    optix::float3 radiance = evaluate(light, position, ray.direction);

    if (bsdf_PDF.use_for_MIS()) {
        // Calculate MIS weight and scale the radiance by it.
        PDF light_PDF = pdf(light, position, ray.direction);
        radiance *= MonteCarlo::MIS_weight(bsdf_PDF, light_PDF);
    }

    return radiance;
}

__inline_dev__ optix::float3 evaluate_intersection(const Light& light, optix::float3 position, optix::float3 direction_to_light, PDF bsdf_PDF) {
    switch (light.get_type()) {
    case Light::Sphere:
        return evaluate_intersection(light.sphere, position, direction_to_light, bsdf_PDF);
    case Light::Spot:
        return evaluate_intersection(light.spot, position, direction_to_light, bsdf_PDF);
    default:
        return optix::make_float3(1000.0f, 0, 1000);
    }
}

} // NS LightSources
} // NS OptiXRenderer

#endif // _OPTIXRENDERER_LIGHT_IMPLEMENTATION_H_