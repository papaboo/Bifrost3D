// OptiX renderer functions for sphere lights.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _OPTIXRENDERER_SPHERE_LIGHT_IMPLEMENTATION_H_
#define _OPTIXRENDERER_SPHERE_LIGHT_IMPLEMENTATION_H_

#include <OptiXRenderer/Distributions.h>
#include <OptiXRenderer/Intersect.h>
#include <OptiXRenderer/TBN.h>
#include <OptiXRenderer/Types.h>

namespace OptiXRenderer {
namespace LightSources {

__device__ static const float sphere_light_small_sin_theta_squared = 0.0f; // 1e-5f;

__inline_all__ float surface_area(const SphereLight& light) {
    return 4.0f * PIf * light.radius * light.radius;
}

// Returns true if the sphere light should be interpreted as a delta light / point light.
// Ideally this should only happen if the radius is zero, but due to floating point 
// imprecission when sampling cones, we draw the line at very tiny subtended angles.
// TODO Rethinke this fix next time it's reproduced, as the fix caused the small light 
//      in the veach scene to be too dim and not produce bloom.
//      There are at least 3 'unguarded' sqrt functions when sampling cones, 
//      where the input has a risc of becoming negative.
__inline_all__ bool is_delta_light(const SphereLight& light, optix::float3 position) {
    optix::float3 vector_to_light = light.position - position;
    float sin_theta_squared = light.radius * light.radius / optix::dot(vector_to_light, vector_to_light);
    return sin_theta_squared <= sphere_light_small_sin_theta_squared;
}

__inline_all__ LightSample sample_radiance(const SphereLight& light, optix::float3 position, optix::float2 random_sample) {

    // Sample Sphere light by sampling a cone with the angle subtended by the sphere.
    optix::float3 vector_to_light = light.position - position;

    float sin_theta_squared = light.radius * light.radius / optix::dot(vector_to_light, vector_to_light);

    LightSample light_sample;
    if (sin_theta_squared <= sphere_light_small_sin_theta_squared) {
        // If the subtended angle is too small, then sampling produces NaN's, so just fall back to a point light.
        light_sample.direction_to_light = vector_to_light;
        light_sample.distance = optix::length(light_sample.direction_to_light);
        light_sample.direction_to_light /= light_sample.distance;
        light_sample.radiance = light.power / (4.0f * PIf * light_sample.distance * light_sample.distance);
        light_sample.distance -= light.radius;
        light_sample.PDF = 1.0f;
    } else {
        // Sample the cone and project the sample onto the sphere.
        float cos_theta = sqrtf(1.0f - sin_theta_squared);
        
        auto cone_sample = Distributions::Cone::sample(cos_theta, random_sample);

        const TBN tbn = TBN(optix::normalize(vector_to_light));
        light_sample.direction_to_light = cone_sample.direction * tbn;
        light_sample.PDF = cone_sample.PDF;
        light_sample.distance = Intersect::ray_sphere(position, light_sample.direction_to_light, light.position, light.radius);
        if (light_sample.distance <= 0.0f)
            // The ray missed the sphere, but since it was sampled to be inside the sphere, just assume that it hit at a grazing angle.
            light_sample.distance = optix::dot(vector_to_light, light_sample.direction_to_light);
        
        // Compute radiance.
        float inv_divisor = 1.0f / (PIf * surface_area(light));
        light_sample.radiance = light.power * inv_divisor;
    }

    // Decrement the shadow ray distance by one ULP. This should avoid self intersections,
    // as it's the same intersection test that is used in the intersection program.
    light_sample.distance = nextafterf(light_sample.distance, 0.0f);

    return light_sample;
}

__inline_all__ float PDF(const SphereLight& light, optix::float3 lit_position, optix::float3 direction_to_light) {
    optix::float3 vector_to_light_center = light.position - lit_position;

    float sin_theta_squared = light.radius * light.radius / optix::dot(vector_to_light_center, vector_to_light_center);
    if (sin_theta_squared < sphere_light_small_sin_theta_squared)
        return 0.0f;
    else {
        float cos_theta_max = sqrtf(1.0f - sin_theta_squared);
        float cos_theta = optix::dot(direction_to_light, optix::normalize(vector_to_light_center));

        float valid_direction = cos_theta >= cos_theta_max ? 1.0f : 0.0f;
        return Distributions::Cone::PDF(cos_theta_max) * valid_direction;
    }
}

__inline_all__ optix::float3 evaluate(const SphereLight& light, optix::float3 position) {
    float inv_divisor = 1.0f / (is_delta_light(light, position) ? (4.0f * PIf) : (PIf * surface_area(light)));
    return light.power * inv_divisor;
}

__inline_all__ optix::float3 evaluate(const SphereLight& light, optix::float3 position, optix::float3 direction) {
    return evaluate(light, position);
}

} // NS LightSources
} // NS OptiXRenderer

#endif // _OPTIXRENDERER_SPHERE_LIGHT_IMPLEMENTATION_H_