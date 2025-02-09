// OptiX renderer functions for spot lights.
// ------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _OPTIXRENDERER_SPOT_LIGHT_IMPLEMENTATION_H_
#define _OPTIXRENDERER_SPOT_LIGHT_IMPLEMENTATION_H_

#include <OptiXRenderer/Distributions.h>
#include <OptiXRenderer/Intersect.h>
#include <OptiXRenderer/TBN.h>
#include <OptiXRenderer/Types.h>

namespace OptiXRenderer {
namespace LightSources {

__device__ static const float spot_light_min_cone_angle_to_sample = 1e-5f;

// ------------------------------------------------------------------------------------------------
// Disk spot light.
// Future work:
// * Improved sampling of the disk wrt solid angle.
//   https://www.arnoldrenderer.com/research/egsr2017_spherical_ellipse.pdf.
// ------------------------------------------------------------------------------------------------

__inline_all__ float surface_area(const SpotLight& light) {
    return PIf * pow2(light.radius);
}

__inline_all__ float is_delta_light(const SpotLight& light) {
    return light.radius == 0.0f;
}

__inline_all__ PDF pdf(const SpotLight& light, optix::float3 lit_position, optix::float3 direction_to_light) {
    using namespace optix;

    float cos_theta = -dot(light.direction, direction_to_light);

    if (cos_theta > 0.0f && !is_delta_light(light)) {
        // Sample the spot light by either sampling the surface of the spotlight or sampling the cone, whichever has the lowest radius.
        float t = Intersect::ray_plane(lit_position, -light.direction, light.position, light.direction);
        float cone_radius_at_intersection = t * sqrtf(1.0f - pow2(light.cos_angle)) / light.cos_angle;
        if (light.radius > cone_radius_at_intersection && light.cos_angle > spot_light_min_cone_angle_to_sample)
            return Distributions::Cone::PDF(light.cos_angle);
        else {
            float t = Intersect::ray_disk(lit_position, direction_to_light, light.position, light.direction, light.radius);
            if (t >= 0.0f) {
                float area_PDF_to_solid_angle_PDF = (t * t) / cos_theta;
                return Distributions::Disk::PDF(light.radius) * area_PDF_to_solid_angle_PDF;
            }
        }
    }

    return PDF::delta_dirac(0);
}

__inline_all__ optix::float3 evaluate(const SpotLight& light, optix::float3 lit_position, optix::float3 direction_to_light) {
    using namespace optix;

    float cos_theta = -dot(light.direction, direction_to_light);

    float normalization = TWO_PIf * (1 - light.cos_angle);
    if (is_delta_light(light))
        normalization *= length_squared(light.position - lit_position);
    else
        normalization *= surface_area(light) * cos_theta;
    float3 radiance = light.power / normalization;

    return (cos_theta > light.cos_angle) ? radiance : make_float3(0.0f, 0.0f, 0.0f);
}

__inline_all__ LightSample sample_radiance(const SpotLight& light, optix::float3 lit_position, optix::float2 random_sample) {
    using namespace optix;

    if (is_delta_light(light)) {
        LightSample light_sample;
        light_sample.direction_to_light = light.position - lit_position;
        light_sample.distance = optix::length(light_sample.direction_to_light);
        light_sample.direction_to_light /= light_sample.distance;

        light_sample.PDF = 1.0f;
        light_sample.radiance = evaluate(light, lit_position, light_sample.direction_to_light);
        return light_sample;
    } else {
        const TBN light_to_world = TBN(light.direction);

        LightSample light_sample;

        // Sample the spot light by either sampling the surface of the spotlight or sampling the cone, whichever has the lowest radius.
        float t = Intersect::ray_plane(lit_position, -light.direction, light.position, light.direction);
        float cone_radius_at_intersection = t * sqrtf(1.0f - pow2(light.cos_angle)) / light.cos_angle;
        if (light.radius > cone_radius_at_intersection && light.cos_angle > spot_light_min_cone_angle_to_sample) {
            auto cone_sample = Distributions::Cone::sample(light.cos_angle, random_sample);
            light_sample.direction_to_light = -cone_sample.direction * light_to_world;
            light_sample.distance = Intersect::ray_plane(lit_position, light_sample.direction_to_light, light.position, light.direction);
            light_sample.PDF = cone_sample.PDF;

            // Evaluate radiance if sampled position is on the surface of the light, otherwise it is black.
            light_sample.radiance = { 0, 0, 0 };
            optix::float3 sample_position_on_light = lit_position + light_sample.direction_to_light * light_sample.distance;
            optix::float3 light_pos_to_sample_pos = sample_position_on_light - light.position;
            if (length_squared(light_pos_to_sample_pos) < pow2(light.radius))
                light_sample.radiance = evaluate(light, lit_position, light_sample.direction_to_light);
        } else {
            // Sample disk and transform to world space.
            auto disk_sample = Distributions::Disk::sample(light.radius, random_sample);
            float3 sampled_position = light.position + make_float3(disk_sample.position, 0.0f) * light_to_world;

            // Create sample
            light_sample.direction_to_light = sampled_position - lit_position;
            light_sample.distance = optix::length(light_sample.direction_to_light);
            light_sample.direction_to_light /= light_sample.distance;

            float cos_theta = -dot(light.direction, light_sample.direction_to_light);
            float area_PDF_to_solid_angle_PDF = pow2(light_sample.distance) / cos_theta;
            light_sample.PDF = Distributions::Disk::PDF(light.radius) * area_PDF_to_solid_angle_PDF;

            light_sample.radiance = evaluate(light, lit_position, light_sample.direction_to_light);
        }

        // Decrement the shadow ray distance by one ULP. This should avoid self intersections,
        // as it's the same intersection test that is used in the intersection program.
        light_sample.distance = nextafterf(light_sample.distance, 0.0f);

        return light_sample;
    }
}

} // NS LightSources
} // NS OptiXRenderer

#endif // _OPTIXRENDERER_SPOT_LIGHT_IMPLEMENTATION_H_