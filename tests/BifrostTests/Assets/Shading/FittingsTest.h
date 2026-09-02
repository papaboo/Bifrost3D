// Test Bifrost BSDF precomputations.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _BIFROST_ASSETS_SHADING_FITTINGS_TEST_H_
#define _BIFROST_ASSETS_SHADING_FITTINGS_TEST_H_

#include <Assets/Shading/BSDFs/GGXTest.h>
#include <Assets/Shading/BSDFs/LambertTest.h>
#include <Assets/Shading/BSDFs/OrenNayarTest.h>
#include <Expects.h>

#include <Bifrost/Assets/Shading/Fittings.h>
#include <Bifrost/Assets/Shading/LinearlyTransformedCosines.h>
#include <Bifrost/Math/RNG.h>

namespace Bifrost::Assets::Shading {

// Validate that a few select samples in the GGX and GGX with Fresnel precomputed Rho tables have the correct values and can be looked up correct.
// We test the corner and middle samples and make sure that the sample coordinates match with the original precomputed sample coords.
GTEST_TEST(Assets_Shading_Fittings, validate_ggx_reflection_rho_precomputations) {
    int sample_count = 4096;

    float middle_cos_theta = (Rho::GGX_angle_sample_count / 2) / (Rho::GGX_angle_sample_count - 1.0f);
    float middle_roughness = (Rho::GGX_roughness_sample_count / 2) / (Rho::GGX_roughness_sample_count - 1.0f);
    float middle_alpha = BSDFs::GGX::alpha_from_roughness(middle_roughness);

    for (float cos_theta_o : { 0.000001f, middle_cos_theta, 1.0f }) {
        Math::Vector3f wo = BSDFTestUtils::w_from_cos_theta(cos_theta_o);
        for (float alpha : { 1e-9f, middle_alpha, 1.0f }) {
            float roughness = BSDFs::GGX::roughness_from_alpha(alpha);

            auto no_specularity_ggx = BSDFs::GGXReflectionWrapper(alpha, 0.0f);
            float expected_no_specularity_rho = BSDFTestUtils::directional_hemispherical_reflectance_function(no_specularity_ggx, wo, sample_count).reflectance.r;
            float actual_no_specularity_rho = Rho::sample_GGX_with_fresnel(cos_theta_o, roughness);
            EXPECT_FLOAT_EQ_EPS(expected_no_specularity_rho, actual_no_specularity_rho, 0.0001f) << "for cos_theta: " << cos_theta_o << " and roughness: " << roughness;

            auto full_specularity_ggx = BSDFs::GGXReflectionWrapper(alpha, 1.0f);
            float expected_full_specularity_rho = BSDFTestUtils::directional_hemispherical_reflectance_function(full_specularity_ggx, wo, sample_count).reflectance.r;
            float actual_full_specularity_rho = Rho::sample_GGX(cos_theta_o, roughness);
            EXPECT_FLOAT_EQ_EPS(expected_full_specularity_rho, actual_full_specularity_rho, 0.0001f) << "for cos_theta: " << cos_theta_o << " and roughness: " << roughness;
        }
    }
}

// Validate that a few select samples in the dielectric GGX precomputed Rho tables have the correct values and can be looked up correct.
// We test the corner and middle samples and make sure that the sample coordinates match with the original precomputed sample coords.
GTEST_TEST(Assets_Shading_Fittings, validate_dielectric_GGX_rho_precomputations) {
    int sample_count = 8192;

    float middle_cos_theta = (Rho::dielectric_GGX_angle_sample_count / 2) / (Rho::dielectric_GGX_angle_sample_count - 1.0f);

    float middle_roughness = (Rho::dielectric_GGX_roughness_sample_count / 2) / (Rho::dielectric_GGX_roughness_sample_count - 1.0f);
    float middle_alpha = BSDFs::GGX::alpha_from_roughness(middle_roughness);

    float common_IOR = 1.5f; // glass and coat IOR
    float tested_IORs[6] = {
        Rho::dielectric_GGX_minimum_IOR_into_light_medium, 1.0f / common_IOR, Rho::dielectric_GGX_maximum_IOR_into_light_medium,
        Rho::dielectric_GGX_minimum_IOR_into_dense_medium, common_IOR, Rho::dielectric_GGX_maximum_IOR_into_dense_medium,
    };

    for (float cos_theta_o : { 1 / 15.0f, middle_cos_theta, 1.0f }) {
        Math::Vector3f wo = BSDFTestUtils::w_from_cos_theta(cos_theta_o);
        for (float alpha : { BSDFs::GGX::MIN_ALPHA, middle_alpha, 1.0f }) {
            float roughness = BSDFs::GGX::roughness_from_alpha(alpha);
            for (float medium_IOR : tested_IORs) {
                float ior_i_over_o = medium_IOR / air_ior;
                float specularity = dielectric_specularity(air_ior, medium_IOR);

                Math::RGB transmission_tint = { 1, 0, 0 };
                auto ggx = BSDFs::GGXWrapper(alpha, ior_i_over_o, transmission_tint);
                Math::RGB expected_rho = BSDFTestUtils::directional_hemispherical_reflectance_function(ggx, wo, sample_count).reflectance;
                float expected_total_reflectance = expected_rho.r; // Red contains both reflected and transmitted contribution.
                float expected_reflected_reflectance = expected_rho.g; // Green only contains reflected contribution as green transmitted tint is 0.

                auto actual_rho = Rho::sample_dielectric_GGX(cos_theta_o, roughness, ior_i_over_o);
                float actual_total_reflectance = actual_rho.total_rho;
                float actual_reflected_reflectance = actual_rho.reflected_rho;
                EXPECT_FLOAT_EQ_EPS(expected_total_reflectance, actual_total_reflectance, 0.0029f) << "for cos_theta: " << cos_theta_o << ", roughness: " << roughness << ", ior_i_over_o: " << ior_i_over_o;
                EXPECT_FLOAT_EQ_EPS(expected_reflected_reflectance, actual_reflected_reflectance, 0.0024f) << "for cos_theta: " << cos_theta_o << ", roughness: " << roughness << ", ior_i_over_o: " << ior_i_over_o;
            }
        }
    }
}

// Compute the LTC fitting error as described in
// Real-Time Polygonal-Light Shading with Linearly Transformed Cosines, Heitz et al., 2016.
// The error is the mean of (brdf_reflectance - ltc_reflectance)^3.
template <typename BSDF>
float LTC_error(Math::Vector3f wo, BSDF bsdf, Math::IsotropicLTC ltc, int max_sample_count = 256) {
    double summed_error = 0.0;
    int valid_sample_count = 0;
    for (int i = 0; i < max_sample_count; ++i) {
        Math::Vector3f random_sample = BSDFTestUtils::bsdf_rng_sample3f(i, max_sample_count);

        { // LTC error
            auto ltc_sample = ltc.sample({ random_sample.x, random_sample.y });
            float ltc_evaluation = ltc_sample.PDF; // LTC's are perfectly sampled
            auto wi = ltc_sample.direction;
            float cos_theta_i = abs(wi.z);

            BSDFResponse bsdf_response = bsdf.evaluate_with_PDF(wo, wi);
            float ltc_target = bsdf_response.reflectance.r * cos_theta_i;

            // error with MIS weight
            if (bsdf_response.PDF.is_valid()) {
                float error = fabsf(ltc_target - ltc_evaluation);
                error = error * error * error;
                float mis_weight = Math::MonteCarlo::balance_heuristic(ltc_sample.PDF, bsdf_response.PDF.value());
                summed_error += error * mis_weight / ltc_sample.PDF;
                ++valid_sample_count;
            }
        }

        { // BSDF error
            BSDFSample bsdf_sample = bsdf.sample(wo, random_sample);
            if (bsdf_sample.PDF.is_valid()) {
                Math::Vector3f wi = bsdf_sample.direction;
                float cos_theta_i = abs(wi.z);

                float ltc_target = bsdf_sample.reflectance.r * cos_theta_i;
                float ltc_evaluation = ltc.evaluate(wi);
                float ltc_PDF = ltc_evaluation; // LTC's are perfectly sampled

                // error with MIS weight
                float error = fabsf(ltc_target - ltc_evaluation);
                error = error * error * error;
                float mis_weight = Math::MonteCarlo::balance_heuristic(bsdf_sample.PDF.value(), ltc_PDF);
                summed_error += error * mis_weight / bsdf_sample.PDF.value();
                ++valid_sample_count;
            }
        }
    }

    return float(summed_error / valid_sample_count);
}

GTEST_TEST(Assets_Shading_Fittings, validate_lambert_LTC_fitting) {
    auto brdf = BSDFs::LambertWrapper();
    auto ltc = LTC::lambert_LTC_coefficients();
    for (float cos_theta_o : { 0.1f, 0.5f, 0.9f }) {
        Math::Vector3f wo = BSDFTestUtils::w_from_cos_theta(cos_theta_o);
        float error = LTC_error(wo, brdf, ltc);

        // Lambertian / cosine distribution is the identity distribution of LTC and should be perfectly sampled.
        EXPECT_FLOAT_EQ_EPS(0.0f, error, 1e-20f) << "cos(theta_o): " << cos_theta_o;
    }
}

GTEST_TEST(Assets_Shading_Fittings, validate_oren_nayar_LTC_error) {
    auto error_statistics = Bifrost::Math::Statistics<float>();

    for (float roughness : { 0.1f, 0.5f, 0.9f }) {
        auto brdf = BSDFs::OrenNayarWrapper(roughness);
        for (float cos_theta_o : { 0.1f, 0.5f, 0.9f }) {
            auto ltc = LTC::oren_nayar_LTC_coefficients(cos_theta_o, roughness);

            Math::Vector3f wo = BSDFTestUtils::w_from_cos_theta(cos_theta_o);
            float error = LTC_error(wo, brdf, ltc);

            error_statistics.add(error);
        }
    }

    EXPECT_LT(error_statistics.mean(), 0.0016f);
    EXPECT_LT(error_statistics.standard_deviation(), 0.0044f);
}

GTEST_TEST(Assets_Shading_Fittings, validate_GGX_LTC_error) {
    auto error_statistics = Bifrost::Math::Statistics<float>();

    for (float roughness : { 0.1f, 0.5f, 0.9f }) {
        const float full_specularity = 1.0f; // The LTCs are fitted without the Fresnel term
        auto brdf = BSDFs::GGXReflectionWrapper(BSDFs::GGX::alpha_from_roughness(roughness), full_specularity);
        brdf.normalized_rho(true);
        for (float cos_theta_o : { 0.1f, 0.5f, 0.9f }) {
            auto ltc = LTC::GGX_reflection_LTC_coefficients(cos_theta_o, roughness);

            Math::Vector3f wo = BSDFTestUtils::w_from_cos_theta(cos_theta_o);
            float error = LTC_error(wo, brdf, ltc);

            error_statistics.add(error);
        }
    }

    EXPECT_FLOAT_EQ_EPS(error_statistics.mean(), 621.0f, 0.5f);
    EXPECT_FLOAT_EQ_EPS(error_statistics.standard_deviation(), 1755.0f, 0.5f);
}

GTEST_TEST(Assets_Shading_Fittings, correct_GGX_LTC_bounds) {
    using namespace Bifrost::Math;

    int count = LTC::GGX_reflection_angle_sample_count * LTC::GGX_reflection_roughness_sample_count;
    Vector4f expected_minimum = Vector4f(INFINITY);
    Vector4f expected_maximum = Vector4f(-INFINITY);
    for (int i = 0; i < count; ++i) {
        Vector4f value = LTC::GGX_reflection_LTC_params[i];
        expected_minimum = min(expected_minimum, value);
        expected_maximum = max(expected_maximum, value);
    }

    Vector4f actual_minimum = LTC::GGX_reflection_minimum_param;
    Vector4f actual_maximum = LTC::GGX_reflection_maximum_param;

    EXPECT_VECTOR4F_EQ_EPS(expected_minimum, actual_minimum, 1e-6f);
    EXPECT_VECTOR4F_EQ_EPS(expected_maximum, actual_maximum, 1e-6f);
}

} // NS Bifrost::Assets::Shading

#endif // _BIFROST_ASSETS_SHADING_FITTINGS_TEST_H_
