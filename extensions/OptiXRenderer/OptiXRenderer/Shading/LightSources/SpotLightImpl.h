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

// ------------------------------------------------------------------------------------------------
// Disk spot light.
// Future work:
// * Sample the disk or a cone from the lit position projected onto the spot lights plane,
//   whichever has the smallest radius.
// * Improved sampling of the disk wrt solid angle.
//   https://www.arnoldrenderer.com/research/egsr2017_spherical_ellipse.pdf.
// ------------------------------------------------------------------------------------------------

__inline_all__ float surface_area(const SpotLight& light) {
    return PIf * pow2(light.radius);
}

__inline_all__ float is_delta_light(const SpotLight& light) {
    return light.radius == 0.0f;
}

__inline_all__ float PDF(const SpotLight& light, optix::float3 lit_position, optix::float3 direction_to_light) {
    using namespace optix;

    float cos_theta = fmaxf(0, -dot(light.direction, direction_to_light));

    if (cos_theta > 0.0f && !is_delta_light(light)) {
        float t = Intersect::ray_disk(lit_position, direction_to_light, light.position, light.direction, light.radius);
        if (t >= 0.0f) {
            float area_PDF_to_solid_angle_PDF = (t * t) / cos_theta;
            return Distributions::Disk::PDF(light.radius) * area_PDF_to_solid_angle_PDF;
        }
    }

    return 0;
}

__inline_all__ optix::float3 evaluate(const SpotLight& light, optix::float3 lit_position, optix::float3 direction_to_light) {
    using namespace optix;

    float cos_theta = fmaxf(0, -dot(light.direction, direction_to_light));

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

        light_sample.radiance = evaluate(light, lit_position, light_sample.direction_to_light);
        light_sample.PDF = 1.0f;
        return light_sample;
    } else {
        // Sample disk and transform to world space.
        const TBN light_to_world = TBN(light.direction);
        auto disk_sample = Distributions::Disk::sample(light.radius, random_sample);
        float3 sampled_position = light.position + make_float3(disk_sample.position, 0.0f) * light_to_world;

        // Create sample
        LightSample light_sample;
        light_sample.direction_to_light = sampled_position - lit_position;
        light_sample.distance = optix::length(light_sample.direction_to_light);
        light_sample.direction_to_light /= light_sample.distance;

        light_sample.radiance = evaluate(light, lit_position, light_sample.direction_to_light);
        light_sample.PDF = PDF(light, lit_position, light_sample.direction_to_light); // TODO Inline evaluate and PDF to reuse shared computations.
        return light_sample;
    }
}

} // NS LightSources
} // NS OptiXRenderer

#endif // _OPTIXRENDERER_SPOT_LIGHT_IMPLEMENTATION_H_