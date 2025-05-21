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

GTEST_TEST(ShadingModelUtils, dielectric_GGX_rho_texture_size) {
    int expected_angle_sample_count = Bifrost::Assets::Shading::Rho::dielectric_GGX_angle_sample_count;
    int actual_angle_sample_count = Shading::ShadingModels::DielectricRho::angle_sample_count;
    EXPECT_EQ(actual_angle_sample_count, expected_angle_sample_count);

    int expected_roughness_sample_count = Bifrost::Assets::Shading::Rho::dielectric_GGX_roughness_sample_count;
    int actual_roughness_sample_count = Shading::ShadingModels::DielectricRho::roughness_sample_count;
    EXPECT_EQ(actual_roughness_sample_count, expected_roughness_sample_count);

    int expected_specularity_sample_count = Bifrost::Assets::Shading::Rho::dielectric_GGX_specularity_sample_count;
    int actual_specularity_sample_count = Shading::ShadingModels::DielectricRho::specularity_sample_count;
    EXPECT_EQ(actual_specularity_sample_count, expected_specularity_sample_count);
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

GTEST_TEST(ShadingModelUtils, GGX_minimum_roughness_edge_case_handling) {
    { // Delta dirac functions are considered as perfect reflections and support a perfectly smooth surfaces on the next bounce.
        float cos_theta = 0.5f;
        PDF delta_pdf = PDF::delta_dirac(1);
        float delta_minimum_roughness = Shading::ShadingModels::GGXMinimumRoughness::from_PDF(cos_theta, delta_pdf);
        EXPECT_FLOAT_EQ(delta_minimum_roughness, 0.0f);
    }

    { // An infinitely high PDF is from a practically mirror-like surface and should support perfectly smooth surfaces on the next bounce.
        float cos_theta = 0.5f;
        PDF infinite_pdf = PDF(INFINITY);
        float infinite_minimum_roughness = Shading::ShadingModels::GGXMinimumRoughness::from_PDF(cos_theta, infinite_pdf);
        EXPECT_FLOAT_EQ(infinite_minimum_roughness, 0.0f);
    }

    { // An invalid PDF is as such not important, as it shouldn't produce further bounces.
        // But for the sake of completeness we define that the next bounce should support perfectly smooth surfaces,
        // as invalid/NaN PDFs can happen if the numerics become unstable on really smooth surfaces.
        float cos_theta = 0.5f;
        PDF invalid_pdf = PDF::invalid();
        float invalid_minimum_roughness = Shading::ShadingModels::GGXMinimumRoughness::from_PDF(cos_theta, invalid_pdf);
        EXPECT_FLOAT_EQ(invalid_minimum_roughness, 0.0f);
    }

    { // A low PDF should result in the next surface interaction having a high roughness
        float cos_theta = 0.5f;
        PDF low_pdf = PDF(0);
        float low_pdf_minimum_roughness = Shading::ShadingModels::GGXMinimumRoughness::from_PDF(cos_theta, low_pdf);
        EXPECT_FLOAT_EQ(low_pdf_minimum_roughness, 1.0f);
    }
}

} // NS OptiXRenderer

#endif // _OPTIXRENDERER_SHADING_MODELS_UTILS_TEST_H_