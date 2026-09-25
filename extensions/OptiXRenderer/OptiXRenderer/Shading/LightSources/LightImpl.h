// OptiX renderer functions for directional lights.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _OPTIXRENDERER_LIGHT_IMPLEMENTATION_H_
#define _OPTIXRENDERER_LIGHT_IMPLEMENTATION_H_

#include <OptiXRenderer/Shading/LightSources/EnvironmentLightImpl.h>
#include <OptiXRenderer/Shading/LightSources/PresampledEnvironmentLightImpl.h>
#include <OptiXRenderer/Shading/LightSources/SpotLightImpl.h>

namespace OptiXRenderer::LightSources {

__inline_dev__ bool is_delta_light(const Light& light, optix::float3 lit_position) {
    switch (light.get_type()) {
    case Light::Sphere:
        return light.sphere.is_delta_light(to_vector3f(lit_position));
    case Light::Directional:
        return light.directional.is_delta_light();
    case Light::Environment:
        return is_delta_light(light.environment);
    case Light::PresampledEnvironment:
        return is_delta_light(light.presampled_environment);
    case Light::Spot:
        return is_delta_light(light.spot);
    }
    return false;
}

__inline_dev__ LightSample sample_radiance(const Light& light, optix::float3 lit_position, optix::float2 random_sample) {
    switch (light.get_type()) {
    case Light::Sphere:
        return light.sphere.sample_radiance(to_vector3f(lit_position), { random_sample.x, random_sample.y });
    case Light::Directional:
        return light.directional.sample_radiance();
    case Light::Environment:
        return sample_radiance(light.environment, random_sample);
    case Light::PresampledEnvironment:
        return sample_radiance(light.presampled_environment, random_sample);
    case Light::Spot:
        return sample_radiance(light.spot, to_vector3f(lit_position), { random_sample.x, random_sample.y });
    }
    return LightSample::none();
}

__inline_dev__ LightResponse evaluate_with_PDF(const Light& light, optix::float3 lit_position, optix::float3 direction_to_light) {
    switch (light.get_type()) {
    case Light::Sphere:
        return light.sphere.evaluate_with_PDF(to_vector3f(lit_position), to_vector3f(direction_to_light));
    case Light::Directional:
        return light.directional.evaluate_with_PDF(to_vector3f(lit_position), to_vector3f(direction_to_light));
    case Light::Environment:
        return evaluate_with_PDF(light.environment, direction_to_light);
    case Light::PresampledEnvironment:
        return evaluate_with_PDF(light.presampled_environment, direction_to_light);
    case Light::Spot:
        return evaluate_with_PDF(light.spot, to_vector3f(lit_position), to_vector3f(direction_to_light));
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