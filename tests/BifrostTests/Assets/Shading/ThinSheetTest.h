// Test Bifrost thin sheet utilities.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _BIFROST_ASSETS_SHADING_THIN_SHEET_TEST_H_
#define _BIFROST_ASSETS_SHADING_THIN_SHEET_TEST_H_

#include <Bifrost/Assets/Shading/BSDFs/GGX.h>
#include <Bifrost/Assets/Shading/BSDFs/Lambert.h>
#include <Bifrost/Assets/Shading/Utils.h>

#include <Assets/Shading/BSDFTestUtils.h>
#include <Expects.h>

namespace Bifrost::Assets::Shading {

GTEST_TEST(Assets_Shading_Fittings_ThinSheetTest, lambertian_thin_sheet_reflects_all_energy) {
    Math::RGB tint = { 1.0f, 0.5f, 0.25f };
    Math::RGB black = Math::RGB::black();

    auto lambertian_sampler = [=](Math::Vector3f wo, Math::Vector3f random_sample) -> BSDFSample {
        return Shading::BSDFs::Lambert::sample(tint, { random_sample.x, random_sample.y });
    };

    Math::Vector3f wo = { 0, 0, 1 };
    unsigned int path_count = 2048;
    auto throughput = BSDFTestUtils::integrate_over_thin_sheet(lambertian_sampler, wo, path_count);

    EXPECT_RGB_EQ(tint, throughput.reflected);
    EXPECT_RGB_EQ(black, throughput.transmitted);
}

GTEST_TEST(Assets_Shading_Fittings_ThinSheetTest, smooth_ggx_thin_sheet_reflects_according_to_expectation) {
    using namespace Bifrost::Math;

    float alpha = 0.0; // Smooth surface
    RGB transmission_tint = { 1.0f, 0.5f, 0.25f };
    RGB transmission_tint_per_side = { sqrt(transmission_tint.r), sqrt(transmission_tint.g), sqrt(transmission_tint.b) };

    for (float medium_IOR : { ice_ior, coat_ior, diamond_ior }) {
        float specularity = dielectric_specularity(air_ior, medium_IOR);

        auto ggx_sampler = [=](Vector3f wo, Vector3f random_sample) -> BSDFSample {
            bool entering = wo.z >= 0.0f;
            float ior_i_over_o = entering ? (medium_IOR / air_ior) : (air_ior / medium_IOR);

            return Shading::BSDFs::GGX::sample(transmission_tint_per_side, alpha, specularity, ior_i_over_o, wo, random_sample);
        };

        for (float cos_theta_o : { 0.3f, 0.5f, 1.0f }) {
            Vector3f wo = BSDFTestUtils::w_from_cos_theta(cos_theta_o);
            unsigned int path_count = 4096;
            unsigned int max_bounce_count = 32;
            auto throughput = BSDFTestUtils::integrate_over_thin_sheet(ggx_sampler, wo, path_count, max_bounce_count);
            auto expected_throughput = BSDFTestUtils::smooth_thin_sheet_reflectance(cos_theta_o, medium_IOR, transmission_tint);

            EXPECT_RGB_EQ_EPS(expected_throughput.reflected, throughput.reflected, 0.01f);
            EXPECT_RGB_EQ_EPS(expected_throughput.transmitted, throughput.transmitted, 0.01f);
        }
    }
}

GTEST_TEST(Assets_Shading_Fittings_ThinSheetTest, approx_smooth_ggx_thin_sheet_is_nearly_exact_for_smooth_surfaces) {
    using namespace Bifrost::Math;

    float roughness = 0.0; // Smooth surface
    RGB transmission_tint = { 1.0f, 0.5f, 0.25f };

    for (float medium_IOR : { ice_ior, coat_ior, diamond_ior }) {
        for (float cos_theta_o : { 0.3f, 0.5f, 1.0f }) {
            auto expected_throughput = BSDFTestUtils::smooth_thin_sheet_reflectance(cos_theta_o, medium_IOR, transmission_tint);
            auto approximate_throughput = BSDFTestUtils::approx_thin_sheet_reflectance(cos_theta_o, roughness, medium_IOR, transmission_tint);

            // The test needs a small epsilon as the approximate implementation discretizes the output, which causes inaccuracies.
            float epsilon = 0.025f;
            EXPECT_RGB_EQ_EPS(expected_throughput.reflected, approximate_throughput.reflected, epsilon);
            EXPECT_RGB_EQ_EPS(expected_throughput.transmitted, approximate_throughput.transmitted, epsilon);
        }
    }
}

GTEST_TEST(Assets_Shading_Fittings_ThinSheetTest, approx_rough_ggx_thin_sheet_RMSE_regression_test) {
    using namespace Bifrost::Math;

    auto sqrt = [=](RGB c) -> RGB { return { sqrtf(c.r), sqrtf(c.g), sqrtf(c.b)}; };

    RGB transmission_tint = { 1.0f, 0.5f, 0.25f };
    RGB transmission_tint_per_side = sqrt(transmission_tint);

    float tested_IORs[6] = { 1.0f / ice_ior, 1.0f / coat_ior, 1.0f / diamond_ior, ice_ior, coat_ior, diamond_ior };

    RGB summed_squared_reflection_error = { 0, 0, 0 };
    RGB summed_squared_transmission_error = { 0, 0, 0 };
    float sample_count = 0;
    for (float roughness : { 0.0f, 0.5f, 1.0f }) {
        float alpha = Shading::BSDFs::GGX::alpha_from_roughness(roughness);
        for (float medium_IOR : tested_IORs) {
            float specularity = dielectric_specularity(air_ior, medium_IOR);
            float ior_air_over_medium = air_ior / medium_IOR;
            float ior_medium_over_air = medium_IOR / air_ior;

            auto ggx_sampler = [=](Vector3f wo, Vector3f random_sample) -> BSDFSample {
                bool entering = wo.z >= 0.0f;
                float ior_i_over_o = entering ? ior_medium_over_air : ior_air_over_medium;

                return Shading::BSDFs::GGX::sample(transmission_tint_per_side, alpha, specularity, ior_i_over_o, wo, random_sample);
            };

            for (float cos_theta_o : { 0.3f, 0.5f, 1.0f}) {
                Vector3f wo = BSDFTestUtils::w_from_cos_theta(cos_theta_o);
                unsigned int path_count = 16384; // High path count as both surfaces can have high variance, so a large sample size is needed to make it converge.
                unsigned int max_bounce_count = 32;
                auto expected_throughput = BSDFTestUtils::integrate_over_thin_sheet(ggx_sampler, wo, path_count, max_bounce_count);

                // The red channel doesn't lose energy while reflecting or transmitting, so normalize the throughput by the summed red.
                float energy_loss_adjustment = 1.0f / (expected_throughput.reflected.r + expected_throughput.transmitted.r);
                expected_throughput.reflected *= energy_loss_adjustment;
                expected_throughput.transmitted *= energy_loss_adjustment;

                auto approximate_throughput = BSDFTestUtils::approx_thin_sheet_reflectance(cos_theta_o, roughness, ior_medium_over_air, transmission_tint);

                summed_squared_reflection_error += pow2(expected_throughput.reflected - approximate_throughput.reflected);
                summed_squared_transmission_error += pow2(expected_throughput.transmitted - approximate_throughput.transmitted);
                sample_count++;
            }
        }
    }

    RGB root_mean_squared_reflection_error = sqrt(summed_squared_reflection_error / sample_count);
    RGB root_mean_squared_transmission_error = sqrt(summed_squared_transmission_error / sample_count);

    float epsilon = 0.025f;
    RGB expected_reflection_error = { 0.2135f, 0.2001f, 0.1973f };
    RGB expected_transmission_error = { 0.2001f, 0.1005f, 0.0503f };
    EXPECT_RGB_EQ_EPS(expected_reflection_error, root_mean_squared_reflection_error, epsilon);
    EXPECT_RGB_EQ_EPS(expected_transmission_error, root_mean_squared_transmission_error, epsilon);
}

} // NS Bifrost::Assets::Shading

#endif // _BIFROST_ASSETS_SHADING_THIN_SHEET_TEST_H_
