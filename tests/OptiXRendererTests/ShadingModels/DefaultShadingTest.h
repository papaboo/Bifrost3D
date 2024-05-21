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

#include <OptiXRenderer/RNG.h>
#include <OptiXRenderer/Shading/ShadingModels/DefaultShading.h>
#include <OptiXRenderer/Utils.h>

#include <gtest/gtest.h>

namespace OptiXRenderer {

class DefaultShadingWrapper {
public:
    Material m_material_params;
    float m_cos_theta;
    Shading::ShadingModels::DefaultShading m_shading_model;

    DefaultShadingWrapper(const Material& material_params, float cos_theta)
        : m_material_params(material_params), m_cos_theta(cos_theta), m_shading_model(material_params, cos_theta) {}

    BSDFResponse evaluate_with_PDF(optix::float3 wo, optix::float3 wi) const { return m_shading_model.evaluate_with_PDF(wo, wi); }

    BSDFSample sample(optix::float3 wo, optix::float3 random_sample) const { return m_shading_model.sample(wo, random_sample); }

    float get_roughness() const { return m_shading_model.get_roughness(); }

    optix::float3 rho(float abs_cos_theta_wi) const { return m_shading_model.rho(abs_cos_theta_wi); }

    std::string to_string() const {
        std::ostringstream out;
        out << "Default shading:" << std::endl;
        out << "  Tint: " << m_material_params.tint.x << ", " << m_material_params.tint.y << ", " << m_material_params.tint.z << std::endl;
        out << "  Roughness: " << m_material_params.roughness << std::endl;
        out << "  Metalness: " << m_material_params.metallic << std::endl;
        out << "  Coverage: " << m_material_params.coverage << std::endl;
        out << "  Coat: " << m_material_params.coat << std::endl;
        out << "  Coat roughness: " << m_material_params.coat_roughness << std::endl;
        return out.str();
    }
};

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
        EXPECT_FLOAT3_LE(result.reflectance, 1.00084f) << "cos_theta: " << cos_theta;
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

GTEST_TEST(DefaultShadingModel, function_consistency) {
    using namespace Shading::ShadingModels;

    static auto evaluate_with_PDF_test = [](Material& material_params) {
        auto wo = optix::normalize(optix::make_float3(1.0f, 1.0f, 1.0f));

        for (float roughness : { 0.2f, 0.4f, 0.6f, 0.8f, 1.0f }) {
            material_params.roughness = roughness;
            auto shading_model = DefaultShadingWrapper(material_params, wo.z);
            ShadingModelTestUtils::consistency_test(shading_model, wo, 32);
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
            float3 weight = material.evaluate_with_PDF(wo, wo).reflectance;
            EXPECT_GT(weight.x, 0.0f);
            EXPECT_FLOAT_EQ_EPS(weight.y, 0.0f, 1e-6f);
            EXPECT_FLOAT_EQ_EPS(weight.z, 0.0f, 1e-6f);
        }

        { // Test that grazing angle reflectivity is white.
            float3 wo = normalize(make_float3(0.0f, 1.0f, 0.001f));
            float3 wi = normalize(make_float3(0.0f, -1.0f, 0.001f));
            auto material = DefaultShading(material_params, wo.z);
            float3 weight = material.evaluate_with_PDF(wo, wi).reflectance;
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
            float3 weight = material.evaluate_with_PDF(wo, wo).reflectance;
            float scale = material_params.tint.x / weight.x;
            EXPECT_FLOAT3_EQ_EPS(scale * weight, material_params.tint, 1e-6f);
        }

        { // Test that grazing angle reflectivity is nearly white.
            float3 wo = normalize(make_float3(0.0f, 1.0f, 0.001f));
            float3 wi = normalize(make_float3(0.0f, -1.0f, 0.001f));
            auto material = DefaultShading(material_params, wo.z);
            float3 weight = material.evaluate_with_PDF(wo, wi).reflectance;
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
        float3 sample_mean = ShadingModelTestUtils::directional_hemispherical_reflectance_function(shading_model, wo).reflectance;

        float3 rho = shading_model.rho(wo.z);

        // The error is slightly higher for low roughness materials.
        float error_percentage = 0.015f * (2 - roughness) * (2 - coat_roughness);
        EXPECT_FLOAT3_EQ_PCT(sample_mean, rho, error_percentage) << "Material params: roughness: " << roughness << ", metallic: " << metallic << ", coat: " << coat << ", coat roughness: " << coat_roughness;
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
                auto material = DefaultShadingWrapper(white_material_params, abs_cos_theta);
                EXPECT_FLOAT_EQ(1.0f, material.rho(abs_cos_theta).x) << material.to_string();
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
                auto material = DefaultShadingWrapper(material_params, abs_cos_theta);
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
                EXPECT_FLOAT3_EQ_EPS(interpolated_rho, rho, 1e-6f) << material.to_string();
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
            EXPECT_FLOAT_EQ_EPS(non_coated_material.get_roughness(), coated_material.get_roughness(), 1e-6f);

            for (float coat : { 0.25f, 0.5f, 0.75f }) {
                // Generate a material with a partial coat that has the same roughness as the coated material.
                material_params.coat = coat;
                auto material = generate_interpolated_coated_material(material_params, cos_theta, coated_material.get_roughness());
                float3 rho = material.rho(cos_theta);

                // Verify that the two materials have the same roughness.
                EXPECT_FLOAT_EQ_EPS(material.get_roughness(), coated_material.get_roughness(), 1e-6f);

                // Test that the directional-hemispherical reflectance of the semi-coated material equals
                // the one evaluated by interpolating between a material with no coat and coated material.
                float3 interpolated_rho = lerp(non_coated_rho, coated_rho, coat);
                EXPECT_FLOAT3_EQ_EPS(interpolated_rho, rho, 1e-6f);
            }
        }
    }
}

GTEST_TEST(DefaultShadingModel, regression_test) {
    using namespace Shading::ShadingModels;
    using namespace optix;

    const unsigned int MAX_SAMPLES = 2;

    Material materials[3] = { ShadingModelTestUtils::gold_parameters(), ShadingModelTestUtils::plastic_parameters(), ShadingModelTestUtils::coated_plastic_parameters() };
    float3 wos[3] = { make_float3(0.0f, 0.0f, 1.0f), normalize(make_float3(1.0f, 0.0f, 1.0f)), normalize(make_float3(1.0f, 0.0f, 0.01f)) };

    BSDFResponse bsdf_responses[] = {
        // Gold
        { 497358.250000f, 380976.437500f, 167112.35938f, 497357.968750f }, { 124339.296875f, 95243.906250f, 41778.00000f, 124339.195313f },
        { 994714.562500f, 762453.062500f, 335647.75000f, 703369.687500f }, { 249080.015625f, 190921.531250f, 84049.10156f, 175985.171875f },
        { 4957715968.0f, 4900811776.0f, 4796245504.0f, 49668972.0f }, { 1455763712.0f, 1439698560.0f, 1410177152.0f, 13442245.0f },
        // Plastic
        { 0.012626f, 0.090607f, 0.10932f, 0.006954f }, { 0.012176f, 0.090156f, 0.10887f, 0.222093f },
        { 0.010227f, 0.087904f, 0.10655f, 0.005283f }, { 0.025274f, 0.102950f, 0.12159f, 0.255087f },
        { 0.054774f, 0.125411f, 0.14236f, 0.354429f }, { 0.100238f, 0.170874f, 0.18783f, 0.379421f },
        // Coated plastic
        { 0.026947f, 0.101801f, 0.11977f, 0.017392f }, { 0.023793f, 0.098647f, 0.11661f, 0.217566f },
        { 0.021004f, 0.095354f, 0.11320f, 0.013394f }, { 0.051778f, 0.126128f, 0.14397f, 0.262095f },
        { 0.090937f, 0.154627f, 0.16991f, 0.355085f }, { 0.157052f, 0.220742f, 0.23603f, 0.385136f } };

    int response_index = 0;
    for (int i = 0; i < 3; ++i)
        for (float3 wo : wos) {
            auto material = DefaultShading(materials[i], wo.z);
            for (int s = 0; s < MAX_SAMPLES; ++s) {
                float3 rng_sample = make_float3(RNG::sample02(s), (s + 0.5f) / MAX_SAMPLES);
                BSDFSample sample = material.sample(wo, rng_sample);
                // printf("{ %.6ff, %.6ff, %.5ff, %.6ff },\n", sample.reflectance.x, sample.reflectance.y, sample.reflectance.z, sample.PDF);
                auto response = bsdf_responses[response_index++];

                EXPECT_FLOAT3_EQ_PCT(response.reflectance, sample.reflectance, 0.0001f);
                EXPECT_FLOAT_EQ_PCT(response.PDF, sample.PDF, 0.0001f);
            }
        }
}

} // NS OptiXRenderer

#endif // _OPTIXRENDERER_SHADING_MODEL_DEFAULT_TEST_H_