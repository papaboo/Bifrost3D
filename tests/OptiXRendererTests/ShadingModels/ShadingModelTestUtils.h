// Test utils for OptiXRenderer's shading models.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _OPTIXRENDERER_SHADING_MODEL_TEST_UTILS_H_
#define _OPTIXRENDERER_SHADING_MODEL_TEST_UTILS_H_

#include <BSDFTestUtils.h>

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

template <typename ShadingModel>
BSDFTestUtils::RhoResult directional_hemispherical_reflectance_function(ShadingModel shading_model, float3 wo) {
    unsigned int sample_count = 8192u;
    return BSDFTestUtils::directional_hemispherical_reflectance_function(shading_model, wo, sample_count);
}

template <typename ShadingModel>
void consistency_test(ShadingModel shading_model, float3 wo, unsigned int sample_count) {
    for (unsigned int i = 0u; i < sample_count; ++i) {
        float3 rng_sample = make_float3(RNG::sample02(i), (i + 0.5f) / sample_count);
        BSDFSample sample = shading_model.sample(wo, rng_sample);

        EXPECT_GE(sample.PDF, 0.0f) << shading_model.to_string();
        if (is_PDF_valid(sample.PDF)) {
            EXPECT_GE(sample.reflectance.x, 0.0f) << shading_model.to_string();

            BSDFResponse response = shading_model.evaluate_with_PDF(wo, sample.direction);
            EXPECT_COLOR_EQ_PCT(sample.reflectance, response.reflectance, 0.00002f) << shading_model.to_string();
            EXPECT_FLOAT_EQ_PCT(sample.PDF, response.PDF, 0.00002f) << shading_model.to_string();
        }
    }
}

} // NS ShadingModelTestUtils
} // NS OptiXRenderer

#endif // _OPTIXRENDERER_SHADING_MODEL_TEST_UTILS_H_