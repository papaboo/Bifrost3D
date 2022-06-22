// Test utils for OptiXRenderer's shading models.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _OPTIXRENDERER_SHADING_MODEL_TEST_UTILS_H_
#define _OPTIXRENDERER_SHADING_MODEL_TEST_UTILS_H_

#include <Utils.h>

#include <OptiXRenderer/RNG.h>
#include <OptiXRenderer/Utils.h>

#include <gtest/gtest.h>

namespace OptiXRenderer {
namespace ShadingModelTestUtils {

using namespace optix;
    
Material gold_parameters() {
    Material gold_params = {};
    gold_params.tint = make_float3(1.0f, 0.766f, 0.336f);
    gold_params.roughness = 0.02f;
    gold_params.metallic = 1.0f;
    gold_params.specularity = 1.0f;
    return gold_params;
}

Material plastic_parameters() {
    Material plastic_params = {};
    plastic_params.tint = make_float3(0.02f, 0.27f, 0.33f);
    plastic_params.roughness = 0.7f;
    plastic_params.metallic = 0.0f;
    plastic_params.specularity = 0.02f;
    return plastic_params;
}

Material coated_plastic_parameters() {
    Material plastic_params = plastic_parameters();
    plastic_params.coat = 1.0f;
    plastic_params.coat_roughness = plastic_params.roughness;
    return plastic_params;
}

struct RhoResult {
    float reflectance;
    float variance;
};

template <typename ShadingModel>
RhoResult directional_hemispherical_reflectance_function(ShadingModel shading_model, float3 wo) {
    const unsigned int MAX_SAMPLES = 4096u;

    double summed_weight = 0.0;
    double summed_weight_squared = 0.0;
    for (unsigned int s = 0u; s < MAX_SAMPLES; ++s) {
        float3 rng_sample = make_float3(RNG::sample02(s), (s + 0.5f) / MAX_SAMPLES);
        BSDFSample sample = shading_model.sample(wo, rng_sample);
        if (is_PDF_valid(sample.PDF))
        {
            float weight = sample.reflectance.x * abs(sample.direction.z) / sample.PDF; // f * ||cos_theta|| / pdf
            summed_weight += weight;
            summed_weight_squared += weight * weight;
        }
    }

    double mean = summed_weight / double(MAX_SAMPLES);
    double mean_squared = summed_weight_squared / double(MAX_SAMPLES);
    double variance = mean_squared - mean * mean;

    return { float(mean), float(variance) };
}

template <typename ShadingModel>
void PDF_consistency_test(ShadingModel shading_model, float3 wo, unsigned int sample_count) {
    for (unsigned int i = 0u; i < sample_count; ++i) {
        float3 rng_sample = make_float3(RNG::sample02(i), (i + 0.5f) / sample_count);
        BSDFSample sample = shading_model.sample(wo, rng_sample);
        if (is_PDF_valid(sample.PDF)) {
            float PDF = shading_model.PDF(wo, sample.direction);
            EXPECT_FLOAT_EQ_PCT(sample.PDF, PDF, 0.0001f);
        }
    }
}

template <typename ShadingModel>
void evaluate_with_PDF_consistency_test(ShadingModel shading_model, float3 wo, unsigned int sample_count) {
    for (unsigned int i = 0u; i < sample_count; ++i) {
        float3 rng_sample = make_float3(RNG::sample02(i), (i + 0.5f) / sample_count);
        BSDFSample sample = shading_model.sample(wo, rng_sample);

        if (is_PDF_valid(sample.PDF)) {
            BSDFResponse response = shading_model.evaluate_with_PDF(wo, sample.direction);
            EXPECT_COLOR_EQ_PCT(shading_model.evaluate(wo, sample.direction), response.reflectance, make_float3(0.00002f));
            EXPECT_FLOAT_EQ(shading_model.PDF(wo, sample.direction), response.PDF);
        }
    }
};

} // NS ShadingModelTestUtils
} // NS OptiXRenderer

#endif // _OPTIXRENDERER_SHADING_MODEL_TEST_UTILS_H_