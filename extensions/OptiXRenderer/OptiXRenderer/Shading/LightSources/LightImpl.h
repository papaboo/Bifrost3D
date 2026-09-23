// OptiX renderer functions for directional lights.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _OPTIXRENDERER_LIGHT_IMPLEMENTATION_H_
#define _OPTIXRENDERER_LIGHT_IMPLEMENTATION_H_

#include <OptiXRenderer/Shading/LightSources/DirectionalLightImpl.h>
#include <OptiXRenderer/Shading/LightSources/EnvironmentLightImpl.h>
#include <OptiXRenderer/Shading/LightSources/PresampledEnvironmentLightImpl.h>
#include <OptiXRenderer/Shading/LightSources/SphereLightImpl.h>
#include <OptiXRenderer/Shading/LightSources/SpotLightImpl.h>

namespace OptiXRenderer::LightSources {

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

__inline_dev__ LightResponse evaluate_with_PDF(const Light& light, optix::float3 lit_position, optix::float3 direction_to_light) {
    switch (light.get_type()) {
    case Light::Sphere:
        return { to_rgb(evaluate(light.sphere, lit_position, direction_to_light)), pdf(light.sphere, lit_position, direction_to_light) };
    case Light::Directional:
        return { to_rgb(evaluate(light.directional, direction_to_light)), pdf(light.directional, direction_to_light) };
    case Light::Environment:
        return { to_rgb(evaluate(light.environment, direction_to_light)), pdf(light.environment, direction_to_light) };
    case Light::PresampledEnvironment:
        return { to_rgb(evaluate(light.presampled_environment, direction_to_light)), pdf(light.presampled_environment, direction_to_light) };
    case Light::Spot:
        return { to_rgb(evaluate(light.spot, lit_position, direction_to_light)), pdf(light.spot, lit_position, direction_to_light) };
    }
    return LightResponse::none();
}

__inline_dev__ optix::float3 evaluate_intersection(const Light& light, optix::float3 lit_position, optix::float3 direction_to_light, PDF bsdf_PDF) {
    LightResponse response = evaluate_with_PDF(light, lit_position, direction_to_light);

    if (bsdf_PDF.use_for_MIS())
        // Calculate MIS weight and scale the radiance by it.
        response.radiance *= MIS_weight(bsdf_PDF, response.PDF);

    return to_float3(response.radiance);
}

} // NS OptiXRenderer::LightSources

#endif // _OPTIXRENDERER_LIGHT_IMPLEMENTATION_H_