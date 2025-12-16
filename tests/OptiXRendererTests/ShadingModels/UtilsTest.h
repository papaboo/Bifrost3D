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

    int expected_specularity_sample_count = Bifrost::Assets::Shading::Rho::dielectric_GGX_ior_i_over_o_sample_count;
    int actual_specularity_sample_count = Shading::ShadingModels::DielectricRho::ior_i_over_o_sample_count;
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

GTEST_TEST(ShadingModelUtils, lambertian_thin_sheet_reflects_all_energy) {
    using namespace optix;

    float3 tint = { 1.0f, 0.5f, 0.25f };
    float3 black = { 0.0f, 0.0f, 0.0f };

    auto lambertian_sampler = [=](float3 wo, float3 random_sample) -> BSDFSample {
        return Shading::BSDFs::Lambert::sample(tint, optix::make_float2(random_sample));
    };

    float3 wo = { 0, 0, 1 };
    unsigned int path_count = 2048;
    auto throughput = BSDFTestUtils::integrate_over_thin_sheet(lambertian_sampler, wo, path_count);

    EXPECT_FLOAT3_EQ(tint, throughput.reflected);
    EXPECT_FLOAT3_EQ(black, throughput.transmitted);
}

GTEST_TEST(ShadingModelUtils, smooth_ggx_thin_sheet_reflects_according_to_expectation) {
    using namespace Bifrost::Assets::Shading;
    using namespace optix;

    float alpha = 0.0; // Smooth surface
    float3 transmission_tint = { 1.0f, 0.5f, 0.25f };
    optix::float3 transmission_tint_per_side = sqrt3(transmission_tint);

    for (float medium_IOR : { Rho::dielectric_GGX_minimum_IOR_into_dense_medium, COAT_IOR, Rho::dielectric_GGX_maximum_IOR_into_dense_medium }) {
        float specularity = dielectric_specularity(AIR_IOR, medium_IOR);

        auto ggx_sampler = [=](float3 wo, float3 random_sample) -> BSDFSample {
            bool entering = wo.z >= 0.0f;
            float ior_i_over_o = entering ? (medium_IOR / AIR_IOR) : (AIR_IOR / medium_IOR);

            return Shading::BSDFs::GGX::sample(transmission_tint_per_side, alpha, specularity, ior_i_over_o, wo, random_sample);
        };

        for (float cos_theta_o : { 0.3f, 0.5f, 1.0f }) {
            float3 wo = BSDFTestUtils::w_from_cos_theta(cos_theta_o);
            unsigned int path_count = 4096;
            unsigned int max_bounce_count = 32;
            auto throughput = BSDFTestUtils::integrate_over_thin_sheet(ggx_sampler, wo, path_count, max_bounce_count);
            auto expected_throughput = BSDFTestUtils::smooth_thin_sheet_reflectance(cos_theta_o, medium_IOR, transmission_tint);

            EXPECT_FLOAT3_EQ_EPS(expected_throughput.reflected, throughput.reflected, 0.01f);
            EXPECT_FLOAT3_EQ_EPS(expected_throughput.transmitted, throughput.transmitted, 0.01f);
        }
    }
}

GTEST_TEST(ShadingModelUtils, approx_smooth_ggx_thin_sheet_is_nearly_exact_for_smooth_surfaces) {
    using namespace Bifrost::Assets::Shading;
    using namespace optix;
    using namespace OptiXRenderer::Shading::ShadingModels;

    float roughness = 0.0; // Smooth surface
    float3 transmission_tint = { 1.0f, 0.5f, 0.25f };

    for (float medium_IOR : { Rho::dielectric_GGX_minimum_IOR_into_dense_medium, COAT_IOR, Rho::dielectric_GGX_maximum_IOR_into_dense_medium }) {
        for (float cos_theta_o : { 0.3f, 0.5f, 1.0f }) {
            auto expected_throughput = BSDFTestUtils::smooth_thin_sheet_reflectance(cos_theta_o, medium_IOR, transmission_tint);
            auto approximate_throughput = approx_thin_sheet_reflectance(cos_theta_o, roughness, medium_IOR, transmission_tint);

            // The test needs a small epsilon as the approximate implementation discretizes the output, which causes inaccuracies.
            float epsilon = 0.025f;
            EXPECT_FLOAT3_EQ_EPS(expected_throughput.reflected, approximate_throughput.reflected, epsilon);
            EXPECT_FLOAT3_EQ_EPS(expected_throughput.transmitted, approximate_throughput.transmitted, epsilon);
        }
    }
}

GTEST_TEST(ShadingModelUtils, approx_rough_ggx_thin_sheet_RMSE_regression_test) {
    using namespace Bifrost::Assets::Shading;
    using namespace optix;
    using namespace OptiXRenderer::Shading::ShadingModels;

    float3 transmission_tint = { 1.0f, 0.5f, 0.25f };
    optix::float3 transmission_tint_per_side = sqrt3(transmission_tint);

    float tested_IORs[6] = {
        Rho::dielectric_GGX_minimum_IOR_into_light_medium, 1.0f / COAT_IOR, Rho::dielectric_GGX_maximum_IOR_into_light_medium,
        Rho::dielectric_GGX_minimum_IOR_into_dense_medium, COAT_IOR, Rho::dielectric_GGX_maximum_IOR_into_dense_medium,
    };

    float3 summed_squared_reflection_error = { 0, 0, 0 };
    float3 summed_squared_transmission_error = { 0, 0, 0 };
    float sample_count = 0;
    for (float roughness : { 0.0f, 0.5f, 1.0f }) {
        float alpha = Shading::BSDFs::GGX::alpha_from_roughness(roughness);
        for (float medium_IOR : tested_IORs) {
            float specularity = dielectric_specularity(AIR_IOR, medium_IOR);
            float ior_air_over_medium = AIR_IOR / medium_IOR;
            float ior_medium_over_air = medium_IOR / AIR_IOR;

            auto ggx_sampler = [=](float3 wo, float3 random_sample) -> BSDFSample {
                bool entering = wo.z >= 0.0f;
                float ior_i_over_o = entering ? ior_medium_over_air : ior_air_over_medium;

                auto sample = Shading::BSDFs::GGX::sample(transmission_tint_per_side, alpha, specularity, ior_i_over_o, wo, random_sample);
                float total_rho = DielectricRho::fetch(abs(wo.z), roughness, ior_i_over_o).total_rho;
                sample.reflectance /= total_rho; // Normalize wrt rho
                return sample;
            };

            for (float cos_theta_o : { 0.3f, 0.5f, 1.0f}) {
                float3 wo = BSDFTestUtils::w_from_cos_theta(cos_theta_o);
                unsigned int path_count = 16384; // High path count as both surfaces can be rough and have high variance, so a large sample size is needed to make it converge.
                unsigned int max_bounce_count = 32;
                auto expected_throughput = BSDFTestUtils::integrate_over_thin_sheet(ggx_sampler, wo, path_count, max_bounce_count);

                // The red channel doesn't loose energy while reflecting or transmitting, so we expect the total throughput to be close to 1
                float no_energy_loss_throughput = expected_throughput.reflected.x + expected_throughput.transmitted.x;
                EXPECT_FLOAT_EQ_EPS(1, no_energy_loss_throughput, 0.025f);

                auto approximate_throughput = approx_thin_sheet_reflectance(cos_theta_o, roughness, ior_medium_over_air, transmission_tint);

                summed_squared_reflection_error += pow2(expected_throughput.reflected - approximate_throughput.reflected);
                summed_squared_transmission_error += pow2(expected_throughput.transmitted - approximate_throughput.transmitted);
                sample_count++;
            }
        }
    }

    float3 root_mean_squared_reflection_error = sqrt3(summed_squared_reflection_error / sample_count);
    float3 root_mean_squared_transmission_error = sqrt3(summed_squared_transmission_error / sample_count);

    float epsilon = 0.025f;
    float3 expected_reflection_error = { 0.213529f, 0.200066f, 0.197312f };
    float3 expected_transmission_error = { 0.200979f, 0.100499f, 0.0502933f };
    EXPECT_FLOAT3_EQ_EPS(expected_reflection_error, root_mean_squared_reflection_error, epsilon);
    EXPECT_FLOAT3_EQ_EPS(expected_transmission_error, root_mean_squared_transmission_error, epsilon);
}

} // NS OptiXRenderer

#endif // _OPTIXRENDERER_SHADING_MODELS_UTILS_TEST_H_