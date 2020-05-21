// OptiX renderer functions for spot lights.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _OPTIXRENDERER_SPOT_LIGHT_IMPLEMENTATION_H_
#define _OPTIXRENDERER_SPOT_LIGHT_IMPLEMENTATION_H_

#include <OptiXRenderer/Distributions.h>
#include <OptiXRenderer/Intersect.h>
#include <OptiXRenderer/TBN.h>
#include <OptiXRenderer/Types.h>

namespace OptiXRenderer {
namespace LightSources {

__inline_all__ float surface_area(const SpotLight& light) {
    return PIf * pow2(light.radius);
}

__inline_all__ float is_delta_light(const SpotLight& light) {
    return light.radius == 0.0f;
}

__inline_all__ float PDF(const SpotLight& light, optix::float3 lit_position, optix::float3 direction_to_light) {
    if (!is_delta_light(light) && Intersect::ray_disk(lit_position, direction_to_light, light.position, light.direction, light.radius) >= 0.0f)
        return Distributions::Disk::PDF(light.radius);
    else
        return 0;
}

__inline_all__ optix::float3 evaluate(const SpotLight& light, optix::float3 lit_position, optix::float3 direction_to_light) {
    using namespace optix;

    float cos_theta = fmaxf(0, -dot(light.direction, normalize(direction_to_light)));
    float normalization = TWO_PIf * (1 - light.cos_angle);
    float3 radiance = light.power / (normalization * length_squared(direction_to_light));
    radiance *= (cos_theta > light.cos_angle) ? 1.0f : 0.0f;
    if (!is_delta_light(light))
        radiance /= surface_area(light);
    return radiance;
}

__inline_all__ LightSample sample_radiance(const SpotLight& light, optix::float3 lit_position, optix::float2 random_sample) {
    using namespace optix;

    if (is_delta_light(light)) {
        optix::float3 vector_to_light = light.position - lit_position;

        LightSample light_sample;
        light_sample.direction_to_light = vector_to_light;
        light_sample.distance = optix::length(light_sample.direction_to_light);
        light_sample.direction_to_light /= light_sample.distance;
        light_sample.radiance = evaluate(light, lit_position, vector_to_light);
        light_sample.PDF = 1.0f;
        return light_sample;
    } else {
        // Sample disk and transform to world space.
        const TBN light_to_world = TBN(light.direction);
        auto disk_sample = Distributions::Disk::sample(light.radius, random_sample);
        float3 sampled_position = light.position + make_float3(disk_sample.position, 0.0f) * light_to_world;
        optix::float3 vector_to_light = sampled_position - lit_position;

        // Create sample
        LightSample light_sample;
        light_sample.direction_to_light = vector_to_light;
        light_sample.radiance = evaluate(light, lit_position, vector_to_light);
        light_sample.PDF = Distributions::Disk::PDF(light.radius);

        light_sample.distance = optix::length(light_sample.direction_to_light);
        light_sample.direction_to_light /= light_sample.distance;

        return light_sample;
    }
}

} // NS LightSources
} // NS OptiXRenderer

#endif // _OPTIXRENDERER_SPOT_LIGHT_IMPLEMENTATION_H_