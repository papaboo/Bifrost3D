// OptiX renderer functions for environment lights.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _OPTIXRENDERER_ENVIRONMENT_LIGHT_IMPLEMENTATION_H_
#define _OPTIXRENDERER_ENVIRONMENT_LIGHT_IMPLEMENTATION_H_

#include <OptiXRenderer/Types.h>
#include <OptiXRenderer/Utils.h>

namespace OptiXRenderer {
namespace LightSources {

__inline_dev__ bool is_delta_light(const EnvironmentLight& light) {
    return false;
}

__inline_dev__ optix::float2 sample_CDFs_for_uv(const EnvironmentLight& light, optix::float2 random_sample) {
    using namespace optix;

    float2 uv;
    int conditional_row = 0;
    { // Binary search the marginal CDF for the sampled row.
        int lowerbound = 0;
        int upperbound = light.height; // The CDFs are one unit larger pr dimension than the environment image, hence no -1 is needed.
        while (lowerbound + 1 != upperbound) {
            int middlebound = (lowerbound + upperbound) / 2;
            float cdf = rtTex1D<float>(light.marginal_CDF_ID, middlebound);
            if (random_sample.y < cdf)
                upperbound = middlebound;
            else
                lowerbound = middlebound;
        }
        conditional_row = lowerbound;

        // Compute ux.y.
        float cdf_at_lowerbound = rtTex1D<float>(light.marginal_CDF_ID, lowerbound);
        float dv = random_sample.y - cdf_at_lowerbound;
        dv /= rtTex1D<float>(light.marginal_CDF_ID, lowerbound + 1) - cdf_at_lowerbound;
        uv.y = (lowerbound + dv) / float(light.height);
    }

    { // Binary search the row of the conditional CDF for the sampled texel.
        int lowerbound = 0;
        int upperbound = light.width; // The CDFs are one unit larger pr dimension than the environment image, hence no -1 is needed.
        while (lowerbound + 1 != upperbound) {
            int middlebound = (lowerbound + upperbound) / 2;
            float cdf = rtTex2D<float>(light.conditional_CDF_ID, middlebound, conditional_row);
            if (random_sample.x < cdf)
                upperbound = middlebound;
            else
                lowerbound = middlebound;
        }

        // Compute ux.x.
        float cdf_at_lowerbound = rtTex2D<float>(light.conditional_CDF_ID, lowerbound, conditional_row);
        float du = random_sample.x - cdf_at_lowerbound;
        du /= rtTex2D<float>(light.conditional_CDF_ID, lowerbound + 1, conditional_row) - cdf_at_lowerbound;
        uv.x = (lowerbound + du) / float(light.width);
    }

    return uv;
}

// For environment map monte carlo sampling see PBRT v2 chapter 14.6.5. 
__inline_dev__ LightSample sample_radiance(const EnvironmentLight& light, optix::float2 random_sample) {
    if (light.per_pixel_PDF_ID == 0) return LightSample::none();

    optix::float2 uv = sample_CDFs_for_uv(light, random_sample);
    LightSample sample;
    sample.direction_to_light = latlong_texcoord_to_direction(uv);
    sample.distance = 1e30f;
    sample.radiance = optix::make_float3(optix::rtTex2D<optix::float4>(light.environment_map_ID, uv.x, uv.y));
    float sin_theta = sqrtf(1.0f - sample.direction_to_light.y * sample.direction_to_light.y);
    float PDF = optix::rtTex2D<float>(light.per_pixel_PDF_ID, uv.x, uv.y) / sin_theta;
    sample.PDF = sin_theta == 0.0f ? 0.0f : PDF;
    return sample;
}

__inline_dev__ float PDF(const EnvironmentLight& light, optix::float3 direction_to_light) {
    if (light.per_pixel_PDF_ID == 0) return 0.0f;

    optix::float2 uv = direction_to_latlong_texcoord(direction_to_light);
    float sin_theta = sqrtf(1.0f - direction_to_light.y * direction_to_light.y);
    float PDF = optix::rtTex2D<float>(light.per_pixel_PDF_ID, uv.x, uv.y) / sin_theta;
    return sin_theta == 0.0f ? 0.0f : PDF;
}

__inline_dev__ optix::float3 evaluate(const EnvironmentLight& light, optix::float3 direction_to_light) {
    optix::float2 uv = direction_to_latlong_texcoord(direction_to_light);
    return optix::make_float3(optix::rtTex2D<optix::float4>(light.environment_map_ID, uv.x, uv.y));
}

// ------------------------------------------------------------------------------------------------
// Functions with generalized parameters.
// ------------------------------------------------------------------------------------------------

__inline_dev__ LightSample sample_radiance(const EnvironmentLight& light, optix::float3 lit_position, optix::float2 random_sample) {
    return sample_radiance(light, random_sample);
}

__inline_dev__ float PDF(const EnvironmentLight& light, optix::float3 lit_position, optix::float3 direction_to_light) {
    return PDF(light, direction_to_light);
}

__inline_dev__ optix::float3 evaluate(const EnvironmentLight& light, optix::float3 lit_position, optix::float3 direction_to_light) {
    return evaluate(light, direction_to_light);
}

} // NS LightSources
} // NS OptiXRenderer

#endif // _OPTIXRENDERER_ENVIRONMENT_LIGHT_IMPLEMENTATION_H_