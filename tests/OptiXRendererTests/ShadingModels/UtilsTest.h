// Test OptiXRenderer's rough glass shading model.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _OPTIXRENDERER_SHADING_MODELS_UTILS_TEST_H_
#define _OPTIXRENDERER_SHADING_MODELS_UTILS_TEST_H_

#include <Utils.h>

#include <Bifrost/Assets/Shading/Fittings.h>

#include <OptiXRenderer/Shading/ShadingModels/Utils.h>

#include <gtest/gtest.h>

namespace OptiXRenderer {

GTEST_TEST(ShadingModelUtils, GGX_rho_texture_size) {
    int expected_angle_sample_count = Bifrost::Assets::Shading::Rho::GGX_angle_sample_count;
    int actual_angle_sample_count = Shading::ShadingModels::SpecularRho::angle_sample_count;
    EXPECT_EQ(actual_angle_sample_count, expected_angle_sample_count);

    int expected_roughness_sample_count = Bifrost::Assets::Shading::Rho::GGX_roughness_sample_count;
    int actual_roughness_sample_count = Shading::ShadingModels::SpecularRho::roughness_sample_count;
    EXPECT_EQ(actual_roughness_sample_count, expected_roughness_sample_count);
}

GTEST_TEST(ShadingModelUtils, GGX_minimum_roughness_texture_size) {
    int expected_angle_sample_count = Bifrost::Assets::Shading::Estimate_GGX_bounded_VNDF_alpha::wo_dot_normal_sample_count;
    int actual_angle_sample_count = Shading::ShadingModels::GGXMinimumRoughness::angle_sample_count;
    EXPECT_EQ(actual_angle_sample_count, expected_angle_sample_count);

    int expected_max_PDF_sample_count = Bifrost::Assets::Shading::Estimate_GGX_bounded_VNDF_alpha::max_PDF_sample_count;
    int actual_max_PDF_sample_count = Shading::ShadingModels::GGXMinimumRoughness::max_PDF_sample_count;
    EXPECT_EQ(actual_max_PDF_sample_count, expected_max_PDF_sample_count);
}

GTEST_TEST(ShadingModelUtils, GGX_minimum_roughness_PDF_encoding_consistent) {
    for (float pdf : {0.1f, 1.0f, 10.0f, 1000.0f, 100000.0f}) {
        float expected_encoded_PDF = Bifrost::Assets::Shading::Estimate_GGX_bounded_VNDF_alpha::encode_PDF(pdf);
        float actual_encoded_PDF = Shading::ShadingModels::GGXMinimumRoughness::encode_PDF(pdf);
        EXPECT_FLOAT_EQ(actual_encoded_PDF, expected_encoded_PDF);
    }
}

} // NS OptiXRenderer

#endif // _OPTIXRENDERER_SHADING_MODELS_UTILS_TEST_H_