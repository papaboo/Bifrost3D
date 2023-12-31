// Test utils for OptiXRenderer's BSDFs.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _OPTIXRENDERER_BSDF_TEST_UTILS_H_
#define _OPTIXRENDERER_BSDF_TEST_UTILS_H_

#include <Utils.h>

#include <OptiXRenderer/RNG.h>
#include <OptiXRenderer/Utils.h>

#include <gtest/gtest.h>

namespace OptiXRenderer {
namespace BSDFTestUtils {

using namespace optix;

struct RhoResult {
    float reflectance;
    float std_dev;
    optix::float3 mean_direction;
};

template <typename BSDFModel>
RhoResult directional_hemispherical_reflectance_function(BSDFModel bsdf_model, float3 wo, unsigned int sample_count) {
    double summed_reflectance = 0.0;
    double summed_reflectance_squared = 0.0;
    optix::double3 summed_directions = { 0.0, 0.0, 0.0 };
    for (unsigned int i = 0u; i < sample_count; ++i) {
        float3 rng_sample = make_float3(RNG::sample02(i), (i + 0.5f) / sample_count);
        BSDFSample sample = bsdf_model.sample(wo, rng_sample);
        if (is_PDF_valid(sample.PDF))
        {
            float reflectance = sample.reflectance.x * abs(sample.direction.z) / sample.PDF; // f * ||cos_theta|| / pdf
            summed_reflectance += reflectance;
            summed_reflectance_squared += reflectance * reflectance;

            float direction_weight = sample.reflectance.x / sample.PDF;
            summed_directions = optix::make_double3(summed_directions.x + direction_weight * sample.direction.x,
                                                    summed_directions.y + direction_weight * sample.direction.y,
                                                    summed_directions.z + direction_weight * sample.direction.z);
        }
    }

    double mean_reflectance = summed_reflectance / sample_count;
    double mean_reflectance_squared = summed_reflectance_squared / sample_count;
    double reflectance_variance = abs(mean_reflectance_squared - mean_reflectance * mean_reflectance);

    optix::float3 direction = { float(summed_directions.x), float(summed_directions.y), float(summed_directions.z) };

    return { float(mean_reflectance), float(sqrt(reflectance_variance)), optix::normalize(direction) };
}

template <typename BSDFModel>
void helmholtz_reciprocity(BSDFModel bsdf_model, float3 wo, unsigned int sample_count) {
    for (unsigned int i = 0u; i < sample_count; ++i) {
        float3 rng_sample = make_float3(RNG::sample02(i), (i + 0.5f) / sample_count);
        BSDFSample sample = bsdf_model.sample(wo, rng_sample);

        if (is_PDF_valid(sample.PDF)) {
            float3 f = bsdf_model.evaluate(sample.direction, wo);
            EXPECT_COLOR_EQ_EPS(sample.reflectance, f, 0.0001f) << bsdf_model.to_string();
        }
    }
}

template <typename BSDFModel>
void BSDF_consistency_test(BSDFModel bsdf_model, float3 wo, unsigned int sample_count) {
    for (unsigned int i = 0u; i < sample_count; ++i) {
        float3 rng_sample = make_float3(RNG::sample02(i), (i + 0.5f) / sample_count);
        BSDFSample sample = bsdf_model.sample(wo, rng_sample);

        EXPECT_GE(sample.PDF, 0.0f) << bsdf_model.to_string();
        if (is_PDF_valid(sample.PDF)) {
            EXPECT_GE(sample.reflectance.x, 0.0f) << bsdf_model.to_string();

            EXPECT_FLOAT_EQ_PCT(sample.PDF, bsdf_model.PDF(wo, sample.direction), 0.00002f) << bsdf_model.to_string();
            EXPECT_COLOR_EQ_PCT(sample.reflectance, bsdf_model.evaluate(wo, sample.direction), 0.00002f) << bsdf_model.to_string();

            BSDFResponse response = bsdf_model.evaluate_with_PDF(wo, sample.direction);
            EXPECT_COLOR_EQ_PCT(sample.reflectance, response.reflectance, 0.00002f) << bsdf_model.to_string();
            EXPECT_FLOAT_EQ_PCT(sample.PDF, response.PDF, 0.00002f) << bsdf_model.to_string();
        }
    }
}

template <typename BSDFModel>
void BSDF_sampling_variance_test(BSDFModel bsdf_model, unsigned int sample_count, float expected_rho_std_dev, float epsilon = 0.01f) {
    float total_std_dev = 0.0f;
    for (float cos_theta : {0.1f, 0.3f, 0.5f, 0.7f, 0.9f, 1.0f}) {
        float3 wo = wo_from_cos_theta(cos_theta);
        auto rho = directional_hemispherical_reflectance_function(bsdf_model, wo, sample_count);
        float rho_std_dev = rho.std_dev / rho.reflectance; // Normalize error wrt reflectance, so dark BSDFs don't automatically have a smaller error
        total_std_dev += rho_std_dev;
    }
    float average_std_dev = total_std_dev / 6;
    EXPECT_FLOAT_EQ_EPS(average_std_dev, expected_rho_std_dev, epsilon) << bsdf_model.to_string();
}

float3 wo_from_cos_theta(float cos_theta) {
    return { sqrt(1 - pow2(cos_theta)), 0.0f, cos_theta };
}

} // NS BSDFTestUtils
} // NS OptiXRenderer

#endif // _OPTIXRENDERER_BSDF_TEST_UTILS_H_