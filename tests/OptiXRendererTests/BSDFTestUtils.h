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
};

template <typename BSDFModel>
RhoResult directional_hemispherical_reflectance_function(BSDFModel bsdf_model, float3 wo, unsigned int sample_count) {
    double summed_weight = 0.0;
    double summed_weight_squared = 0.0;
    for (unsigned int i = 0u; i < sample_count; ++i) {
        float3 rng_sample = make_float3(RNG::sample02(i), (i + 0.5f) / sample_count);
        BSDFSample sample = bsdf_model.sample(wo, rng_sample);
        if (is_PDF_valid(sample.PDF))
        {
            float weight = sample.reflectance.x * abs(sample.direction.z) / sample.PDF; // f * ||cos_theta|| / pdf
            summed_weight += weight;
            summed_weight_squared += weight * weight;
        }
    }

    double mean = summed_weight / double(sample_count);
    double mean_squared = summed_weight_squared / double(sample_count);
    double variance = abs(mean_squared - mean * mean);

    return { float(mean), float(sqrt(variance)) };
}

template <typename BSDFModel>
void helmholtz_reciprocity(BSDFModel bsdf_model, float3 wo, unsigned int sample_count) {
    for (unsigned int i = 0u; i < sample_count; ++i) {
        float3 rng_sample = make_float3(RNG::sample02(i), (i + 0.5f) / sample_count);
        BSDFSample sample = bsdf_model.sample(wo, rng_sample);

        if (is_PDF_valid(sample.PDF)) {
            float3 f = bsdf_model.evaluate(sample.direction, wo);
            EXPECT_COLOR_EQ_EPS(sample.reflectance, f, make_float3(0.0001f));
        }
    }
}

template <typename BSDFModel>
void BSDF_consistency_test(BSDFModel bsdf_model, float3 wo, unsigned int sample_count) {
    for (unsigned int i = 0u; i < sample_count; ++i) {
        float3 rng_sample = make_float3(RNG::sample02(i), (i + 0.5f) / sample_count);
        BSDFSample sample = bsdf_model.sample(wo, rng_sample);

        EXPECT_GE(sample.PDF, 0.0f);
        if (is_PDF_valid(sample.PDF)) {
            EXPECT_GE(sample.reflectance.x, 0.0f);

            EXPECT_FLOAT_EQ_PCT(sample.PDF, bsdf_model.PDF(wo, sample.direction), 0.00002f);
            EXPECT_COLOR_EQ_PCT(sample.reflectance, bsdf_model.evaluate(wo, sample.direction), make_float3(0.00002f));

            BSDFResponse response = bsdf_model.evaluate_with_PDF(wo, sample.direction);
            EXPECT_COLOR_EQ_PCT(sample.reflectance, response.reflectance, make_float3(0.00002f));
            EXPECT_FLOAT_EQ_PCT(sample.PDF, response.PDF, 0.00002f);
        }
    }
};

} // NS BSDFTestUtils
} // NS OptiXRenderer

#endif // _OPTIXRENDERER_BSDF_TEST_UTILS_H_