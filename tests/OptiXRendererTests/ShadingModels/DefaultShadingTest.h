// Test OptiXRenderer's default shading model.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _OPTIXRENDERER_SHADING_MODEL_DEFAULT_TEST_H_
#define _OPTIXRENDERER_SHADING_MODEL_DEFAULT_TEST_H_

#include <Utils.h>

#include <Bifrost/Math/Utils.h>

#include <OptiXRenderer/RNG.h>
#include <OptiXRenderer/Shading/ShadingModels/DefaultShading.h>
#include <OptiXRenderer/Utils.h>

#include <gtest/gtest.h>

namespace OptiXRenderer {

Material gold_parameters() {
    Material gold_params = {};
    gold_params.tint = optix::make_float3(1.0f, 0.766f, 0.336f);
    gold_params.roughness = 0.02f;
    gold_params.metallic = 1.0f;
    gold_params.specularity = 0.25f;
    return gold_params;
}

Material plastic_parameters() {
    Material plastic_params = {};
    plastic_params.tint = optix::make_float3(0.02f, 0.27f, 0.33f);
    plastic_params.roughness = 0.7f;
    plastic_params.metallic = 0.0f;
    plastic_params.specularity = 0.25f;
    return plastic_params;
}

Material coated_plastic_parameters() {
    Material plastic_params = plastic_parameters();
    plastic_params.coat = 1.0f;
    plastic_params.coat_roughness = plastic_params.roughness;
    return plastic_params;
}

GTEST_TEST(DefaultShadingModel, power_conservation) {
    using namespace Shading::ShadingModels;
    using namespace optix;

    const unsigned int MAX_SAMPLES = 4096u;

    // A white material to stress test power_conservation.
    Material material_params = {};
    material_params.tint = optix::make_float3(1.0f, 1.0f, 1.0f);
    material_params.roughness = 0.7f;
    material_params.metallic = 0.0f;
    material_params.specularity = 0.25f;

    for (float cos_theta : { 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f }) {
        const float3 wo = { sqrt(1 - pow2(cos_theta)), 0.0f, cos_theta };
        auto material = DefaultShading(material_params, wo.z);
        float ws[MAX_SAMPLES];
        for (unsigned int s = 0u; s < MAX_SAMPLES; ++s) {
            float3 rng_sample = make_float3(RNG::sample02(s), float(s) / float(MAX_SAMPLES));
            BSDFSample sample = material.sample_all(wo, rng_sample);
            if (is_PDF_valid(sample.PDF))
                ws[s] = sample.reflectance.x * sample.direction.z / sample.PDF; // f * ||cos_theta|| / pdf
            else
                ws[s] = 0.0f;
        }

        float average_w = Bifrost::Math::sort_and_pairwise_summation(ws, ws + MAX_SAMPLES) / float(MAX_SAMPLES);
        EXPECT_LE(average_w, 1.0011f);
    }
}

/*
 * DefaultShading currently ignores the Helmholtz reciprocity rule.
GTEST_TEST(DefaultShadingModel, Helmholtz_reciprocity) {
    using namespace Shading::ShadingModels;
    using namespace optix;

    const unsigned int MAX_SAMPLES = 128u;

    for (int i = 0; i < 10; ++i) {
        const float3 wo = normalize(make_float3(float(i), 0.0f, 1.001f - float(i) * 0.1f));
        auto plastic_material = DefaultShading(plastic_parameters(), wo.z);
        for (unsigned int s = 0u; s < MAX_SAMPLES; ++s) {
            float3 rng_sample = make_float3(RNG::sample02(s), float(s) / float(MAX_SAMPLES));
            BSDFSample sample = plastic_material.sample_all(wo, rng_sample);
            if (is_PDF_valid(sample.PDF)) {
                // Re-evaluate contribution from both directions to avoid 
                // floating point imprecission between sampling and evaluating.
                // (Yes, they can actually get quite high on smooth materials.)
                float3 f0 = plastic_material.evaluate(wo, sample.direction);
                float3 f1 = plastic_material.evaluate(sample.direction, wo);

                EXPECT_FLOAT_EQ_PCT(f0.x, f1.x, 0.000013f);
                EXPECT_FLOAT_EQ_PCT(f0.y, f1.y, 0.000013f);
                EXPECT_FLOAT_EQ_PCT(f0.z, f1.z, 0.000013f);
            }
        }
    }
}
*/

GTEST_TEST(DefaultShadingModel, consistent_PDF) {
    using namespace Shading::ShadingModels;

    static auto default_shading_model_consistent_PDF_test = [](Material& material_params) {
        using namespace optix;

        const float3 wo = normalize(make_float3(1.0f, 0.0f, 1.0f));
        auto material = DefaultShading(material_params, wo.z);

        const unsigned int MAX_SAMPLES = 64;
        for (unsigned int i = 0u; i < MAX_SAMPLES; ++i) {
            float3 rng_sample = make_float3(RNG::sample02(i), float(i) / float(MAX_SAMPLES));
            BSDFSample sample = material.sample_all(wo, rng_sample);
            if (is_PDF_valid(sample.PDF)) {
                float PDF = material.PDF(wo, sample.direction);
                EXPECT_TRUE(almost_equal_eps(sample.PDF, PDF, 0.0001f));
            }
        }
    };

    // This test can only be performed with rough materials, as the PDF of smooth materials 
    // is highly sensitive to floating point precision.
    default_shading_model_consistent_PDF_test(plastic_parameters());
    default_shading_model_consistent_PDF_test(coated_plastic_parameters());
}

GTEST_TEST(DefaultShadingModel, evaluate_with_PDF) {
    using namespace optix;
    using namespace Shading::ShadingModels;


    static auto evaluate_with_PDF_test = [](Material& material_params) {
        const unsigned int MAX_SAMPLES = 128u;
        const float3 wo = normalize(make_float3(1.0f, 1.0f, 1.0f));

        for (float roughness : { 0.2f, 0.4f, 0.6f, 0.8f, 1.0f }) {
            material_params.roughness = roughness;
            auto material = DefaultShading(material_params, wo.z);
            for (unsigned int i = 0u; i < MAX_SAMPLES; ++i) {
                float3 rng_sample = make_float3(RNG::sample02(i), float(i) / float(MAX_SAMPLES));
                BSDFSample sample = material.sample_all(wo, rng_sample);

                if (is_PDF_valid(sample.PDF)) {
                    BSDFResponse response = material.evaluate_with_PDF(wo, sample.direction);
                    EXPECT_COLOR_EQ_PCT(material.evaluate(wo, sample.direction), response.reflectance, make_float3(0.00002f));
                    EXPECT_FLOAT_EQ(material.PDF(wo, sample.direction), response.PDF);
                }
            }

        }
    };

    evaluate_with_PDF_test(gold_parameters());
    evaluate_with_PDF_test(plastic_parameters());
    evaluate_with_PDF_test(coated_plastic_parameters());
}

GTEST_TEST(DefaultShadingModel, Fresnel) {
    using namespace Shading::ShadingModels;
    using namespace optix;

    { // Test that specular reflections on non-metals are white and incident reflections are diffuse.
        Material material_params = {};
        material_params.tint = make_float3(1.0f, 0.0f, 0.0f);
        material_params.roughness = 0.02f;
        material_params.metallic = 0.0f;
        material_params.specularity = 0.0f; // Testing specularity. Physically-based fubar value.

        { // Test that incident reflectivity is red.
            float3 wo = make_float3(0.0f, 0.0f, 1.0f);
            auto material = DefaultShading(material_params, wo.z);
            float3 weight = material.evaluate(wo, wo);
            EXPECT_GT(weight.x, 0.0f);
            EXPECT_FLOAT_EQ(weight.y, 0.0f);
            EXPECT_FLOAT_EQ(weight.z, 0.0f);
        }

        { // Test that specular reflectivity is white.
            float3 wo = normalize(make_float3(0.0f, 1.0f, 0.001f));
            float3 wi = normalize(make_float3(0.0f, -1.0f, 0.001f));
            auto material = DefaultShading(material_params, wo.z);
            float3 weight = material.evaluate(wo, wi);
            EXPECT_GT(weight.x, 0.99f);
            EXPECT_FLOAT_EQ(weight.x, weight.y);
            EXPECT_FLOAT_EQ(weight.x, weight.z);
        }
    }

    { // Test that specular reflections on metals are tinted.
        Material material_params = gold_parameters();

        { // Test that incident reflectivity is tint scaled.
            float3 wo = make_float3(0.0f, 0.0f, 1.0f);
            auto material = DefaultShading(material_params, wo.z);
            float3 weight = material.evaluate(wo, wo);
            float scale = material_params.tint.x / weight.x;
            EXPECT_FLOAT_EQ(weight.x * scale, material_params.tint.x);
            EXPECT_FLOAT_EQ(weight.y * scale, material_params.tint.y);
            EXPECT_FLOAT_EQ(weight.z * scale, material_params.tint.z);
        }

        { // Test that grazing angle reflectivity is nearly white.
            float3 wo = normalize(make_float3(0.0f, 1.0f, 0.001f));
            float3 wi = normalize(make_float3(0.0f, -1.0f, 0.001f));
            auto material = DefaultShading(material_params, wo.z);
            float3 weight = material.evaluate(wo, wi);
            EXPECT_GT(weight.y, 0.99f);
            EXPECT_FLOAT_EQ_PCT(weight.y, weight.x, 0.01f);
            EXPECT_FLOAT_EQ_PCT(weight.y, weight.z, 0.01f);
        }
    }
}

GTEST_TEST(DefaultShadingModel, directional_hemispherical_reflectance_estimation) {
    using namespace Shading::ShadingModels;
    using namespace optix;

    // Test albedo is properly estimated.
    static auto test_albedo = [](float3 wo, float roughness, float metallic, float coat = 0.0f, float coat_roughness = 0.0f) {
        Material material_params = {};
        material_params.tint = make_float3(1.0f, 1.0f, 1.0f);
        material_params.roughness = roughness;
        material_params.metallic = metallic;
        material_params.specularity = 0.5f;
        material_params.coat = coat;
        material_params.coat_roughness = coat_roughness;
        auto material = DefaultShading(material_params, wo.z);

        const unsigned int MAX_SAMPLES = 2048;
        double* ws = new double[MAX_SAMPLES];
        for (unsigned int i = 0u; i < MAX_SAMPLES; ++i) {
            float3 rng_sample = make_float3(RNG::sample02(i), float(i) / float(MAX_SAMPLES));
            BSDFSample sample = material.sample_all(wo, rng_sample);
            if (is_PDF_valid(sample.PDF))
                ws[i] = sample.reflectance.x * abs(sample.direction.z) / sample.PDF; // f * ||cos_theta|| / pdf
            else
                ws[i] = 0.0f;
        }

        double sample_mean = Bifrost::Math::sort_and_pairwise_summation(ws, ws + MAX_SAMPLES) / double(MAX_SAMPLES);
        float rho = material.rho(wo.z).x;

        // The error is slightly higher for low roughness materials.
        float error_percentage = 0.015f * (2 - roughness) * (2 - coat_roughness);
        EXPECT_FLOAT_EQ_PCT(float(sample_mean), rho, error_percentage) << "Material params: roughness: " << roughness << ", metallic: " << metallic << ", coat: " << coat << ", coat roughness: " << coat_roughness;
    };

    const float3 incident_wo = make_float3(0.0f, 0.0f, 1.0f);
    const float3 average_wo = normalize(make_float3(1.0f, 0.0f, 1.0f));
    const float3 grazing_wo = normalize(make_float3(1.0f, 0.0f, 0.01f));

    for (auto wo : { incident_wo, average_wo, grazing_wo })
        for (float roughness : { 0.25f, 0.75f })
            for (float metallic : { 0.0f, 0.5f, 1.0f })
                for (float coat : { 0.0f, 0.5f, 1.0f })
                    for (float coat_roughness : { 0.25f, 0.75f })
                        test_albedo(wo, roughness, metallic, coat, coat_roughness);
}

GTEST_TEST(DefaultShadingModel, white_hot_room) {
    using namespace Shading::ShadingModels;
    using namespace optix;

    // A white material to stress test power_conservation.
    Material white_material_params = {};
    white_material_params.tint = optix::make_float3(1.0f, 1.0f, 1.0f);
    white_material_params.specularity = 0.5f;

    for (float metallic : { 0.0f, 0.5f, 1.0f }) {
        white_material_params.metallic = metallic;
        for (float roughness : { 0.0f, 0.5f, 1.0f }) {
            white_material_params.roughness = roughness;

            for (int a = 0; a < 5; ++a) {
                float abs_cos_theta = 1.0f - float(a) * 0.2f;
                auto material = DefaultShading(white_material_params, abs_cos_theta);
                EXPECT_FLOAT_EQ(1.0f, material.rho(abs_cos_theta).x);
            }
        }
    }
}

GTEST_TEST(DefaultShadingModel, metallic_interpolation) {
    using namespace Shading::ShadingModels;
    using namespace optix;

    Material material_params = {};
    material_params.tint = optix::make_float3(1.0f, 0.5f, 0.25f);
    material_params.specularity = 0.5f;

    for (float roughness : { 0.0f, 0.5f, 1.0f }) {
        material_params.roughness = roughness;
        for (float metallic : { 0.25f, 0.5f, 0.75f }) {
            for (float abs_cos_theta : { 0.2f, 0.4f, 0.6f, 0.8f, 1.0f }) {

                material_params.metallic = metallic;
                auto material = DefaultShading(material_params, abs_cos_theta);
                float3 rho = material.rho(abs_cos_theta);

                material_params.metallic = 0;
                auto dielectric_material = DefaultShading(material_params, abs_cos_theta);
                float3 dielectric_rho = dielectric_material.rho(abs_cos_theta);

                material_params.metallic = 1;
                auto conductor_material = DefaultShading(material_params, abs_cos_theta);
                float3 conductor_rho = conductor_material.rho(abs_cos_theta);

                // Test that the directional-hemispherical reflectance of the semi-metallic material equals
                // the one evaluated by interpolating a fully dielectric and fully conductor material.
                float3 interpolated_rho = lerp(dielectric_rho, conductor_rho, metallic);
                EXPECT_FLOAT_EQ(interpolated_rho.x, rho.x);
                EXPECT_FLOAT_EQ(interpolated_rho.y, rho.y);
                EXPECT_FLOAT_EQ(interpolated_rho.z, rho.z);
            }
        }
    }
}

// Test that a partial coat is the linear interpolation of the material with no coat and full coat.
GTEST_TEST(DefaultShadingModel, coat_interpolation) {
    using namespace Shading::ShadingModels;
    using namespace optix;

    Material material_params = plastic_parameters();

    for (float coat_roughness : { 0.0f, 0.5f, 1.0f }) {
        material_params.coat_roughness = coat_roughness;
        for (float coat : { 0.25f, 0.5f, 0.75f }) {
            for (float abs_cos_theta : { 0.2f, 0.4f, 0.6f, 0.8f, 1.0f }) {
                material_params.coat = coat;
                auto material = DefaultShading(material_params, abs_cos_theta);
                float3 rho = material.rho(abs_cos_theta);

                material_params.coat = 0;
                auto non_coat_material = DefaultShading(material_params, abs_cos_theta);
                float3 non_coat_rho = non_coat_material.rho(abs_cos_theta);

                material_params.coat = 1;
                auto coated_material = DefaultShading(material_params, abs_cos_theta);
                float3 coated_rho = coated_material.rho(abs_cos_theta);

                // Test that the directional-hemispherical reflectance of the semi-coated material equals
                // the one evaluated by interpolating between a material with no coat and coated material.
                float3 interpolated_rho = lerp(non_coat_rho, coated_rho, coat);
                EXPECT_FLOAT_EQ_EPS(interpolated_rho.x, rho.x, 0.000001f);
                EXPECT_FLOAT_EQ_EPS(interpolated_rho.y, rho.y, 0.000001f);
                EXPECT_FLOAT_EQ_EPS(interpolated_rho.z, rho.z, 0.000001f);
            }
        }
    }
}

GTEST_TEST(DefaultShadingModel, sampling_variance) {
    using namespace Shading::ShadingModels;
    using namespace optix;

    const unsigned int MAX_SAMPLES = 8196;
    const float3 wo = normalize(make_float3(1.0f, 0.0f, 1.0f));

    double* ws = new double[MAX_SAMPLES];
    double* ws_squared = new double[MAX_SAMPLES];

    // Test that sample_all has lower variance than sample_one and that they converge to the same result.
    static auto sampling_variance_test = [=](Material& material_params) {

        auto material = DefaultShading(material_params, wo.z);

        for (unsigned int i = 0u; i < MAX_SAMPLES; ++i) {
            float3 rng_sample = make_float3(RNG::sample02(i), float(i) / float(MAX_SAMPLES));
            BSDFSample sample = material.sample_one(wo, rng_sample);
            if (is_PDF_valid(sample.PDF)) {
                ws[i] = sample.reflectance.x * abs(sample.direction.z) / sample.PDF; // f * ||cos_theta|| / pdf
                ws_squared[i] = ws[i] * ws[i];
            } else
                ws_squared[i] = ws[i] = 0.0f;
        }

        double sample_one_mean = Bifrost::Math::sort_and_pairwise_summation(ws, ws + MAX_SAMPLES) / double(MAX_SAMPLES);
        double sample_one_mean_squared = Bifrost::Math::sort_and_pairwise_summation(ws_squared, ws_squared + MAX_SAMPLES) / double(MAX_SAMPLES);
        double sample_one_variance = sample_one_mean_squared - sample_one_mean * sample_one_mean;

        for (unsigned int i = 0u; i < MAX_SAMPLES; ++i) {
            float3 rng_sample = make_float3(RNG::sample02(i), float(i) / float(MAX_SAMPLES));
            BSDFSample sample = material.sample_all(wo, rng_sample);
            if (is_PDF_valid(sample.PDF)) {
                ws[i] = sample.reflectance.x * abs(sample.direction.z) / sample.PDF; // f * ||cos_theta|| / pdf
                ws_squared[i] = ws[i] * ws[i];
            } else
                ws_squared[i] = ws[i] = 0.0f;
        }

        double sample_all_mean = Bifrost::Math::sort_and_pairwise_summation(ws, ws + MAX_SAMPLES) / double(MAX_SAMPLES);
        double sample_all_mean_squared = Bifrost::Math::sort_and_pairwise_summation(ws_squared, ws_squared + MAX_SAMPLES) / double(MAX_SAMPLES);
        double sample_all_variance = sample_all_mean_squared - sample_all_mean * sample_all_mean;

        EXPECT_TRUE(almost_equal_eps(float(sample_one_mean), float(sample_all_mean), 0.0001f));
        EXPECT_LT(sample_all_variance, sample_one_variance);
    };

    sampling_variance_test(plastic_parameters());
    sampling_variance_test(coated_plastic_parameters());

    delete[] ws;
    delete[] ws_squared;
}

} // NS OptiXRenderer

#endif // _OPTIXRENDERER_SHADING_MODEL_DEFAULT_TEST_H_