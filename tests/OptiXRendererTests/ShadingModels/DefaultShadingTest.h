// Test OptiXRenderer's default shading model.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _OPTIXRENDERER_SHADING_MODEL_DEFAULT_TEST_H_
#define _OPTIXRENDERER_SHADING_MODEL_DEFAULT_TEST_H_

#include <ShadingModels/ShadingModelTestUtils.h>
#include <Utils.h>

#include <Bifrost/Math/Utils.h>

#include <OptiXRenderer/RNG.h>
#include <OptiXRenderer/Shading/ShadingModels/DefaultShading.h>
#include <OptiXRenderer/Utils.h>

#include <gtest/gtest.h>

namespace OptiXRenderer {

GTEST_TEST(DefaultShadingModel, power_conservation) {
    using namespace Shading::ShadingModels;
    using namespace optix;

    // A white material to stress test power_conservation.
    Material material_params = {};
    material_params.tint = optix::make_float3(1.0f, 1.0f, 1.0f);
    material_params.roughness = 0.7f;
    material_params.metallic = 0.0f;
    material_params.specularity = 0.02f;

    for (float cos_theta : { 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f }) {
        const float3 wo = { sqrt(1 - pow2(cos_theta)), 0.0f, cos_theta };
        auto shading_model = DefaultShading(material_params, wo.z);
        auto result = ShadingModelTestUtils::directional_hemispherical_reflectance_function(shading_model, wo);
        EXPECT_LE(result.reflectance, 1.00029f);
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
            BSDFSample sample = plastic_material.sample(wo, rng_sample);
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

    static auto consistent_PDF_test = [](Material& material_params) {
        auto wo = optix::normalize(optix::make_float3(1.0f, 0.0f, 1.0f));
        auto shading_model = DefaultShading(material_params, wo.z);
        ShadingModelTestUtils::PDF_consistency_test(shading_model, wo, 32);
    };

    // This test can only be performed with rough materials, as the PDF of smooth materials 
    // is highly sensitive to floating point precision.
    consistent_PDF_test(ShadingModelTestUtils::plastic_parameters());
    consistent_PDF_test(ShadingModelTestUtils::coated_plastic_parameters());
}

GTEST_TEST(DefaultShadingModel, consistent_evaluate_with_PDF) {
    using namespace Shading::ShadingModels;

    static auto evaluate_with_PDF_test = [](Material& material_params) {
        auto wo = optix::normalize(optix::make_float3(1.0f, 1.0f, 1.0f));

        for (float roughness : { 0.2f, 0.4f, 0.6f, 0.8f, 1.0f }) {
            material_params.roughness = roughness;
            auto shading_model = DefaultShading(material_params, wo.z);
            ShadingModelTestUtils::evaluate_with_PDF_consistency_test(shading_model, wo, 32);
        }
    };

    evaluate_with_PDF_test(ShadingModelTestUtils::gold_parameters());
    evaluate_with_PDF_test(ShadingModelTestUtils::plastic_parameters());
    evaluate_with_PDF_test(ShadingModelTestUtils::coated_plastic_parameters());
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

        { // Test that grazing angle reflectivity is white.
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
        Material material_params = ShadingModelTestUtils::gold_parameters();

        { // Test that incident reflectivity equals a scaled tint.
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
        material_params.specularity = 0.04f;
        material_params.coat = coat;
        material_params.coat_roughness = coat_roughness;
        auto shading_model = DefaultShading(material_params, wo.z);
        float sample_mean = ShadingModelTestUtils::directional_hemispherical_reflectance_function(shading_model, wo).reflectance;

        float rho = shading_model.rho(wo.z).x;

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
    white_material_params.specularity = 0.04f;

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
    material_params.specularity = 0.04f;

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

// Helper function to generate a coated material with a specific target roughness.
// The input roughness for the material is found through a binary search.
Shading::ShadingModels::DefaultShading generate_interpolated_coated_material(Material material_params, float cos_theta, float target_roughness) {
    using namespace Shading::ShadingModels;
    using namespace optix;

    double roughness = 0.5;
    double roughness_adjustment = 0.25;
    material_params.roughness = (float)roughness;
    do {
        auto material = DefaultShading(material_params, cos_theta);
        float roughness_delta = abs(material.get_roughness() - target_roughness);
        if (roughness_delta < 1e-8f)
            return material;

        bool decrease_roughness = material.get_roughness() > target_roughness;
        roughness += decrease_roughness ? -roughness_adjustment : roughness_adjustment;
        if (material_params.roughness == (float)roughness)
            return material;

        material_params.roughness = (float)roughness;
        roughness_adjustment *= 0.5;
    } while (true);
}

// Test that a partial coat is the linear interpolation of the material with no coat and full coat.
// A rough coat will modulate the roughness of the base layer. Strictly speaking this breaks
// the linear interpolation property of the coat, but that is how the material is defined.
// A coated material and a non-coated material can still be linearly interpolated,
// if we ensure that they both have the same roughness after being created.
GTEST_TEST(DefaultShadingModel, coat_interpolation) {
    using namespace Shading::ShadingModels;
    using namespace optix;

    Material material_params = ShadingModelTestUtils::plastic_parameters();
    material_params.roughness = 0.85f;
    float original_roughness = material_params.roughness;

    for (float coat_roughness : { 0.0f, 0.5f, 1.0f }) {
        material_params.coat_roughness = coat_roughness;
        for (float cos_theta : { 0.2f, 0.4f, 0.6f, 0.8f, 1.0f }) {
            material_params.coat = 1;
            material_params.roughness = original_roughness;
            auto coated_material = DefaultShading(material_params, cos_theta);
            float3 coated_rho = coated_material.rho(cos_theta);

            // Verify that material roughness is lower when the coat isn't perfectly smooth.
            if (coat_roughness > 0.0)
                EXPECT_LT(material_params.roughness, coated_material.get_roughness());

            material_params.coat = 0;
            material_params.roughness = coated_material.get_roughness();
            auto non_coated_material = DefaultShading(material_params, cos_theta);
            float3 non_coated_rho = non_coated_material.rho(cos_theta);

            // Verify that the two materials have the same roughness.
            EXPECT_FLOAT_EQ_EPS(non_coated_material.get_roughness(), coated_material.get_roughness(), 0.000001f);

            for (float coat : { 0.25f, 0.5f, 0.75f }) {
                // Generate a material with a partial coat that has the same roughness as the coated material.
                material_params.coat = coat;
                auto material = generate_interpolated_coated_material(material_params, cos_theta, coated_material.get_roughness());
                float3 rho = material.rho(cos_theta);

                // Verify that the two materials have the same roughness.
                EXPECT_FLOAT_EQ_EPS(material.get_roughness(), coated_material.get_roughness(), 0.000001f);

                // Test that the directional-hemispherical reflectance of the semi-coated material equals
                // the one evaluated by interpolating between a material with no coat and coated material.
                float3 interpolated_rho = lerp(non_coated_rho, coated_rho, coat);
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

    // Test that evaluating all BRDFs with a given sample has lower variance than just sampling one and that they converge to the same result.
    static auto sampling_variance_test = [=](Material& material_params) {

        auto shading_model = DefaultShading(material_params, wo.z);

        for (unsigned int i = 0u; i < MAX_SAMPLES; ++i) {
            float3 rng_sample = make_float3(RNG::sample02(i), float(i) / float(MAX_SAMPLES));
            BSDFSample sample = shading_model.sample_one(wo, rng_sample);
            if (is_PDF_valid(sample.PDF)) {
                ws[i] = sample.reflectance.x * abs(sample.direction.z) / sample.PDF; // f * ||cos_theta|| / pdf
                ws_squared[i] = ws[i] * ws[i];
            } else
                ws_squared[i] = ws[i] = 0.0f;
        }

        double sample_one_mean = Bifrost::Math::sort_and_pairwise_summation(ws, ws + MAX_SAMPLES) / double(MAX_SAMPLES);
        double sample_one_mean_squared = Bifrost::Math::sort_and_pairwise_summation(ws_squared, ws_squared + MAX_SAMPLES) / double(MAX_SAMPLES);
        double sample_one_variance = sample_one_mean_squared - sample_one_mean * sample_one_mean;
        
        auto sample_all_result = ShadingModelTestUtils::directional_hemispherical_reflectance_function(shading_model, wo);

        EXPECT_FLOAT_EQ_EPS(float(sample_one_mean), sample_all_result.reflectance, 0.0001f);
        EXPECT_LT(sample_all_result.std_dev, sqrt(sample_one_variance));
    };

    sampling_variance_test(ShadingModelTestUtils::plastic_parameters());
    sampling_variance_test(ShadingModelTestUtils::coated_plastic_parameters());

    delete[] ws;
    delete[] ws_squared;
}

GTEST_TEST(DefaultShadingModel, regression_test) {
    using namespace Shading::ShadingModels;
    using namespace optix;

    const unsigned int MAX_SAMPLES = 2;

    Material materials[3] = { ShadingModelTestUtils::gold_parameters(), ShadingModelTestUtils::plastic_parameters(), ShadingModelTestUtils::coated_plastic_parameters() };
    float3 wos[3] = { make_float3(0.0f, 0.0f, 1.0f), normalize(make_float3(1.0f, 0.0f, 1.0f)), normalize(make_float3(1.0f, 0.0f, 0.01f)) };

    BSDFResponse bsdf_responses[] = {
        // Gold
        {497357.906250f, 380976.156250f, 167112.25f, 497357.90625f}, {124339.273438f, 95243.882813f, 41777.996094f, 124339.210938f},
        {994712.5f, 762451.4375f, 335647.0625f, 703368.8125f}, {249076.625f, 190918.921875f, 84047.960938f, 175982.875f},
        {4938916864.0f, 4882190336.0f, 4777949696.0f, 49537124.0f}, {1234685696.0f, 1220504576.0f, 1194445312.0f, 12383978.0f},
        // Plastic
        {0.014787f, 0.092766f, 0.111481f, 0.319138f}, {0.012301f, 0.090280f, 0.108995f, 0.220644f},
        {0.015975f, 0.093622f, 0.112258f, 0.300184f}, {0.021748f, 0.099395f, 0.118031f, 0.241324f},
        {0.012864f, 0.083453f, 0.100394f, 0.290481f}, {0.099139f, 0.169727f, 0.186669f, 0.375691f},
        // Coated plastic
        {0.029084f, 0.103935f, 0.121900f, 0.313488f}, {0.024119f, 0.098970f, 0.116935f, 0.213343f},
        {0.031833f, 0.106137f, 0.123970f, 0.296725f}, {0.043611f, 0.117915f, 0.135748f, 0.248994f},
        {0.022988f, 0.086616f, 0.101887f, 0.273984f}, {0.024616f, 0.088245f, 0.103516f, 0.142139f} };

    int response_index = 0;
    for (int i = 0; i < 3; ++i)
        for (float3 wo : wos) {
            auto material = DefaultShading(materials[i], wo.z);
            for (int s = 0; s < MAX_SAMPLES; ++s) {
                float3 rng_sample = make_float3(RNG::sample02(s), float(s) / float(MAX_SAMPLES));
                BSDFSample sample = material.sample(wo, rng_sample);
                // printf("{%.6ff, %.6ff, %.6ff, %.6ff},\n", sample.reflectance.x, sample.reflectance.y, sample.reflectance.z, sample.PDF);
                auto response = bsdf_responses[response_index++];

                EXPECT_FLOAT_EQ_PCT(response.reflectance.x, sample.reflectance.x, 0.0001f);
                EXPECT_FLOAT_EQ_PCT(response.reflectance.y, sample.reflectance.y, 0.0001f);
                EXPECT_FLOAT_EQ_PCT(response.reflectance.z, sample.reflectance.z, 0.0001f);
                EXPECT_FLOAT_EQ_PCT(response.PDF, sample.PDF, 0.0001f);
            }
        }
}

} // NS OptiXRenderer

#endif // _OPTIXRENDERER_SHADING_MODEL_DEFAULT_TEST_H_