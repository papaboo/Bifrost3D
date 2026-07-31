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
#include <Bifrost/Math/Statistics.h>

#include <OPtiXRenderer/Distributions.h>

#include <BSDFs/GGXTest.h>
#include <BSDFs/LambertTest.h>
#include <BSDFs/OrenNayarTest.h>
#include <BSDFTestUtils.h>

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
            auto ltc_sample = ltc.sample({ random_sample.x, random_sample.y });
            float ltc_evaluation = ltc_sample.PDF; // LTC's are perfectly sampled
            auto wi = ltc_sample.direction;
            float cos_theta_i = abs(wi.z);

            BSDFResponse bsdf_response = bsdf.evaluate_with_PDF(wo, { wi.x, wi.y, wi.z });
            float ltc_target = bsdf_response.reflectance.x * cos_theta_i;

            // error with MIS weight
            if (bsdf_response.PDF.is_valid()) {
                float error = fabsf(ltc_target - ltc_evaluation);
                summed_error += pow3(error) / (ltc_sample.PDF + bsdf_response.PDF.value());
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

GTEST_TEST(LTC, lambert_integrate_triangle_light) {
    using namespace optix;

    // Define the surface plane to illuminate.
    // The surface passes through origo and the normal is along positive z.
    float3 surface_point = { 0, 0, 0 };
    float3 surface_normal = { 0, 0, 1 };

    // Define triangle light above surface plane.
    float distance_to_surface = 1.0f;
    float emission = 1;
    float3 v0 = { -1, 1, distance_to_surface };
    float3 v1 = { 1, 1, distance_to_surface };
    float3 v2 = { 1, -1, distance_to_surface };
    float3 light_surface_normal = { 0, 0, -1 };
    float light_surface_area = 2;

    // Integrate light
    int max_sample_count = 1024;
    float integral = 0.0f;
    for (int s = 0; s < max_sample_count; ++s) {
        // Sample barycentric coordinates
        float3 bc = Distributions::Triangle::sample_barycentric_coords(BSDFTestUtils::bsdf_rng_sample2f(s));
        float area_pdf = Distributions::Triangle::PDF(light_surface_area);

        float3 light_sample_point = bc.x * v0 + bc.y * v1 + bc.z * v2;
        float3 wi = normalize(light_sample_point - surface_point);

        // Converting from local area PDF to solid angle PDF wrt the surface point.
        float area_PDF_to_solid_angle_PDF = abs(dot(wi, light_surface_normal)) / length_squared(light_sample_point - surface_point);
        float solid_angle_PDF = area_pdf * area_PDF_to_solid_angle_PDF;

        float f = 1 / PIf; // TODO Block if light is on the backside / evaluate by material
        float abs_cos_theta_i = abs(dot(surface_normal, wi));

        integral += f * emission * abs_cos_theta_i / solid_angle_PDF;
    }

    integral /= max_sample_count;
}

} // NS OptiXRenderer

#endif // _OPTIXRENDERER_LTC_TEST_H_