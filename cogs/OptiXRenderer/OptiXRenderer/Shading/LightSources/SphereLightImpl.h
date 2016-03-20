// OptiX renderer functions for sphere lights.
// ---------------------------------------------------------------------------
// Copyright (C) 2015-2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _OPTIXRENDERER_SPHERE_LIGHT_IMPLEMENTATION_H_
#define _OPTIXRENDERER_SPHERE_LIGHT_IMPLEMENTATION_H_

#include <OptiXRenderer/Distributions.h>
#include <OptiXRenderer/Intersect.h>
#include <OptiXRenderer/TBN.h>
#include <OptiXRenderer/Types.h>

namespace OptiXRenderer {
namespace LightSources {

__inline_all__ float surface_area(const SphereLight& light) {
    return 4.0f * M_PIf * light.radius * light.radius;
}

// Returns true if the sphere light should be interpreted as a delta light / point light.
// Ideally this should only happen if the radius is zero, but due to floating point 
// imprecission when sampling cones, we draw the line at very tiny subtended angles.
__inline_all__ bool is_delta_light(const SphereLight& light, const optix::float3& position) {
    optix::float3 vector_to_light = light.position - position;
    float sin_theta_squared = light.radius * light.radius / optix::dot(vector_to_light, vector_to_light);
    return sin_theta_squared < 1e-5f;
}

__inline_all__ LightSample sample_radiance(const SphereLight& light, const optix::float3& position, optix::float2 random_sample) {

    // Sample Sphere light by sampling a cone with the angle subtended by the sphere.
    optix::float3 vector_to_light = light.position - position;

    float sin_theta_squared = light.radius * light.radius / optix::dot(vector_to_light, vector_to_light);

    LightSample light_sample;
    if (sin_theta_squared < 1e-5f) {
        // If the subtended angle is too small, then sampling produces NaN's, so just fall back to a point light.
        light_sample.direction = vector_to_light;
        light_sample.distance = optix::length(light_sample.direction);
        light_sample.direction /= light_sample.distance;
        light_sample.radiance = light.power / (4.0f * PIf * light_sample.distance * light_sample.distance);
        light_sample.distance -= 1.1f * light.radius; // Reduce distance by slightly more than the radius to avoid self intersections.
        light_sample.PDF = 1.0f;
    } else {
        // Sample the cone and project the sample onto the sphere.
        using namespace Distributions;

        float cos_theta = sqrtf(1.0f - sin_theta_squared);
        
        DirectionalSample cone_sample = Cone::sample(cos_theta, random_sample);

        const TBN tbn = TBN(vector_to_light / optix::length(vector_to_light));
        light_sample.direction = cone_sample.direction * tbn;
        light_sample.PDF = cone_sample.PDF;
        light_sample.distance = Intersect::ray_sphere(position, light_sample.direction, light.position, light.radius);
        if (light_sample.distance <= 0.0f)
            // The ray missed the sphere, but since it was sampled to be inside the sphere, just assume that it hit at a grazing angle.
            light_sample.distance = optix::dot(vector_to_light, light_sample.direction);
        
        // Compute radiance.
        float inv_divisor = 1.0f / (PIf * surface_area(light));
        light_sample.radiance = light.power * inv_divisor;
    }

    return light_sample;
}

__inline_all__ optix::float3 evaluate(const SphereLight& light, const optix::float3& position, const optix::float3& direction) {
    float inv_divisor = 1.0f / (is_delta_light(light, position) ? (4.0f * PIf) : (PIf * surface_area(light)));
    return light.power * inv_divisor;
}

} // NS LightSources
} // NS OptiXRenderer

#endif // _OPTIXRENDERER_SPHERE_LIGHT_IMPLEMENTATION_H_