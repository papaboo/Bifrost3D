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

#include <Bifrost/Math/Statistics.h>

#include <OptiXRenderer/Distributions.h>
#include <OptiXRenderer/RNG.h>
#include <OptiXRenderer/Utils.h>

#include <gtest/gtest.h>

namespace OptiXRenderer {
namespace BSDFTestUtils {

using namespace optix;

struct RhoResult {
    optix::float3 reflectance;
    optix::float3 std_dev;
    optix::float3 mean_direction;
};

template <typename BSDFModel>
RhoResult directional_hemispherical_reflectance_function(BSDFModel bsdf_model, float3 wo, unsigned int sample_count) {
    using namespace Bifrost::Math;
    using namespace optix;

    Statistics<double> reflectance_statistics[3] = { Statistics<double>(), Statistics<double>(), Statistics<double>() };
    double3 summed_directions = { 0.0, 0.0, 0.0 };
    for (unsigned int i = 0u; i < sample_count; ++i) {
        float3 rng_sample = make_float3(RNG::sample02(i), (i + 0.5f) / sample_count);
        BSDFSample sample = bsdf_model.sample(wo, rng_sample);

        float3 reflectance = { 0, 0, 0 };
        if (is_PDF_valid(sample.PDF)) {
            reflectance = sample.reflectance * abs(sample.direction.z) / sample.PDF; // f * ||cos_theta|| / pdf

            float direction_weight = sum(sample.reflectance) / sample.PDF;
            summed_directions = { summed_directions.x + direction_weight * sample.direction.x,
                                  summed_directions.y + direction_weight * sample.direction.y,
                                  summed_directions.z + direction_weight * sample.direction.z };
        }

        reflectance_statistics[0].add(reflectance.x);
        reflectance_statistics[1].add(reflectance.y);
        reflectance_statistics[2].add(reflectance.z);
    }

    float3 mean_reflectance = { (float)reflectance_statistics[0].mean(),
                                (float)reflectance_statistics[1].mean(),
                                (float)reflectance_statistics[2].mean() };
    float3 reflectance_std_dev = { (float)reflectance_statistics[0].standard_deviation(),
                                   (float)reflectance_statistics[1].standard_deviation(),
                                   (float)reflectance_statistics[2].standard_deviation() };

    float3 direction = { float(summed_directions.x), float(summed_directions.y), float(summed_directions.z) };

    return { mean_reflectance, reflectance_std_dev, normalize(direction) };
}

template <typename BSDFModel>
void BSDF_sampling_variance_test(BSDFModel bsdf_model, unsigned int sample_count, optix::float3 expected_rho_std_dev, float epsilon = 0.01f) {
    optix::float3 total_std_dev = { 0, 0, 0 };
    for (float cos_theta : {0.1f, 0.3f, 0.5f, 0.7f, 0.9f, 1.0f}) {
        float3 wo = w_from_cos_theta(cos_theta);
        auto rho = directional_hemispherical_reflectance_function(bsdf_model, wo, sample_count);
        optix::float3 rho_std_dev = rho.std_dev / rho.reflectance; // Normalize error wrt reflectance, so dark BSDFs don't automatically have a smaller error
        total_std_dev += rho_std_dev;
    }
    optix::float3 average_std_dev = total_std_dev / 6;
    EXPECT_FLOAT3_EQ_EPS(average_std_dev, expected_rho_std_dev, epsilon) << bsdf_model.to_string();
}

template <typename BSDFModel>
void BSDF_sampling_variance_test(BSDFModel bsdf_model, unsigned int sample_count, float expected_rho_std_dev, float epsilon = 0.01f) {
    BSDF_sampling_variance_test(bsdf_model, sample_count, optix::make_float3(expected_rho_std_dev), epsilon);
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

// Sample BRDF over a sphere and validate that if the BRDF reflects light, then the PDF must be positive.
template <typename BSDFModel>
void PDF_positivity_test(BSDFModel bsdf_model, optix::float3 wo, unsigned int sample_count) {
    using namespace optix;

    for (unsigned int i = 0u; i < sample_count; ++i) {
        float2 rng_sample = RNG::sample02(i);
        auto wi = Distributions::UniformSphere::sample(rng_sample).direction;

        BSDFResponse sample = bsdf_model.evaluate_with_PDF(wo, wi);

        // Test that reflectance is never negative.
        EXPECT_GE(sample.reflectance.x, 0.0f) << bsdf_model.to_string();
        EXPECT_GE(sample.reflectance.y, 0.0f) << bsdf_model.to_string();
        EXPECT_GE(sample.reflectance.z, 0.0f) << bsdf_model.to_string();

        // Test that if the bsdf reflects light, then the PDF is positive.
        if (!is_black(sample.reflectance))
            EXPECT_GE(sample.PDF, 0.0f) << bsdf_model.to_string();
    }
}

float3 w_from_cos_theta(float cos_theta) {
    return { sqrt(1 - pow2(cos_theta)), 0.0f, cos_theta };
}

} // NS BSDFTestUtils
} // NS OptiXRenderer

#endif // _OPTIXRENDERER_BSDF_TEST_UTILS_H_