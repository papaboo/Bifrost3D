// Test LTC applications in OptiXRenderer.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _OPTIXRENDERER_LTC_TEST_H_
#define _OPTIXRENDERER_LTC_TEST_H_

#include <Bifrost/Assets/Shading/LinearlyTransformedCosines.h>

#include <BSDFs/GGXTest.h>
#include <BSDFs/LambertTest.h>
#include <BSDFs/OrenNayarTest.h>
#include <BSDFTestUtils.h>

#include <Bifrost/Math/Statistics.h>

#include <gtest/gtest.h>

namespace OptiXRenderer {

// Compute the LTC fitting error as described in
// Real-Time Polygonal-Light Shading with Linearly Transformed Cosines, Heitz et al., 2016.
// The error is the mean of (brdf_reflectance - ltc_reflectance)^3.
template <typename BSDF>
float LTC_error(optix::float3 wo, BSDF bsdf, Bifrost::Math::IsotropicLTC ltc, int max_sample_count = 256) {
    double summed_error = 0.0;
    int valid_sample_count = 0;
    for (int i = 0; i < max_sample_count; ++i) {
        optix::float3 random_sample = BSDFTestUtils::bsdf_rng_sample3f(i, max_sample_count);

        { // LTC error
            Bifrost::Math::Vector3f wi = ltc.sample({ random_sample.x, random_sample.y });
            float cos_theta_i = abs(wi.z);

            BSDFResponse bsdf_response = bsdf.evaluate_with_PDF(wo, { wi.x, wi.y, wi.z });
            float ltc_target = bsdf_response.reflectance.x * cos_theta_i;
            float ltc_evaluation = ltc.evaluate(wi);
            float ltc_PDF = ltc_evaluation; // LTC's are perfectly sampled

            // error with MIS weight
            if (bsdf_response.PDF.is_valid()) {
                float error = fabsf(ltc_target - ltc_evaluation);
                summed_error += pow3(error) / (ltc_PDF + bsdf_response.PDF.value());
                ++valid_sample_count;
            }
        }

        { // BSDF error
            BSDFSample bsdf_sample = bsdf.sample(wo, random_sample);
            optix::float3 wi = bsdf_sample.direction;
            float cos_theta_i = abs(wi.z);

            float ltc_target = bsdf_sample.reflectance.x * cos_theta_i;
            float ltc_evaluation = ltc.evaluate({ wi.x, wi.y, wi.z });
            float ltc_PDF = ltc_evaluation; // LTC's are perfectly sampled

            // error with MIS weight
            if (bsdf_sample.PDF.is_valid()) {
                float error = fabsf(ltc_target - ltc_evaluation);
                summed_error += pow3(error) / (ltc_PDF + bsdf_sample.PDF.value());
                ++valid_sample_count;
            }
        }
    }

    return float(summed_error / valid_sample_count);
}

GTEST_TEST(LTC, lambert_error) {
    auto brdf = LambertWrapper();
    auto ltc = Bifrost::Assets::Shading::LTC::lambert_LTC_coefficients();
    for (float cos_theta_o : { 0.1f, 0.5f, 0.9f } ) {
        optix::float3 wo = BSDFTestUtils::w_from_cos_theta(cos_theta_o);
        float error = LTC_error(wo, brdf, ltc);

        // Lambertian / cosine distribution is the identity distribution of LTC and should be perfectly sampled.
        EXPECT_FLOAT_EQ_EPS(0.0f, error, 1e-20f) << "cos(theta_o): " << cos_theta_o;
    }
}

GTEST_TEST(LTC, oren_nayar_error) {
    auto error_statistics = Bifrost::Math::Statistics<float>();

    for (float roughness : { 0.1f, 0.5f, 0.9f }) {
        auto brdf = OrenNayarWrapper(roughness);
        for (float cos_theta_o : { 0.1f, 0.5f, 0.9f }) {
            auto ltc = Bifrost::Assets::Shading::LTC::oren_nayar_LTC_coefficients(cos_theta_o, roughness);

            optix::float3 wo = BSDFTestUtils::w_from_cos_theta(cos_theta_o);
            float error = LTC_error(wo, brdf, ltc);

            error_statistics.add(error);
        }
    }

    EXPECT_LT(error_statistics.mean(), 0.0045f);
    EXPECT_LT(error_statistics.standard_deviation(), 0.007f);
}

GTEST_TEST(LTC, GGX_error) {
    const float full_specularity = 1.0f; // The LTCs are fitted without the Fresnel term

    auto error_statistics = Bifrost::Math::Statistics<float>();

    for (float roughness : { 0.1f, 0.5f, 0.9f }) {
        auto brdf = GGXReflectionWrapper(roughness, full_specularity);
        brdf.normalized_rho(true);
        for (float cos_theta_o : { 0.1f, 0.5f, 0.9f }) {
            auto ltc = Bifrost::Assets::Shading::LTC::GGX_reflection_LTC_coefficients(cos_theta_o, roughness);

            optix::float3 wo = BSDFTestUtils::w_from_cos_theta(cos_theta_o);
            float error = LTC_error(wo, brdf, ltc);

            error_statistics.add(error);
        }
    }

    EXPECT_LT(error_statistics.mean(), 46.0f);
    EXPECT_LT(error_statistics.standard_deviation(), 107.0f);
}

} // NS OptiXRenderer

#endif // _OPTIXRENDERER_LTC_TEST_H_