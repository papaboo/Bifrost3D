// Test Bifrost's default shading model.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _BIFROST_ASSETS_SHADING_SHADING_MODELS_DEFAULT_TEST_H_
#define _BIFROST_ASSETS_SHADING_SHADING_MODELS_DEFAULT_TEST_H_

#include <Assets/Shading/BSDFTestUtils.h>
#include <Expects.h>

#include <Bifrost/Assets/Shading/ShadingModels/DefaultShading.h>
#include <Bifrost/Assets/Shading/Constants.h>
#include <Bifrost/Assets/Shading/Utils.h>
#include <Bifrost/Assets/Material.h>
#include <Bifrost/Math/RNG.h>

#include <gtest/gtest.h>

namespace Bifrost::Assets::Shading::ShadingModels {

class DefaultShadingWrapper {
public:
    Material m_material_params;
    Math::Vector2f m_texcoord;
    float m_cos_theta;
    DefaultShading m_shading_model;

    DefaultShadingWrapper(const Material material_params, Math::Vector2f texcoord, float cos_theta_o)
        : m_material_params(material_params), m_texcoord(texcoord), m_cos_theta(cos_theta_o), m_shading_model(material_params, texcoord, cos_theta_o) {}
    DefaultShadingWrapper(const Material material_params, float cos_theta_o)
        : m_material_params(material_params), m_texcoord({0.5f, 0.5f}), m_cos_theta(cos_theta_o), m_shading_model(material_params, {0.5f, 0.5f}, cos_theta_o) {}

    BSDFResponse evaluate_with_PDF(Math::Vector3f wo, Math::Vector3f wi) const { return m_shading_model.evaluate_with_PDF(wo, wi); }

    BSDFSample sample(Math::Vector3f wo, Math::Vector3f random_sample) const { return m_shading_model.sample(wo, random_sample); }

    DefaultShading::BsdfLtcStack get_LTC_representation(float cos_theta_o) const { return m_shading_model.get_LTC_representation(cos_theta_o); }

    float get_roughness() const { return m_shading_model.get_roughness(); }

    float get_diffuse_probability() const { return m_shading_model.get_diffuse_probability(); }
    float get_specular_probability() const { return m_shading_model.get_specular_probability(); }
    float get_coat_probability() const { return m_shading_model.get_coat_probability(); }

    Math::RGB rho(float abs_cos_theta_wi) const { return m_shading_model.rho(abs_cos_theta_wi); }
    Math::RGB diffuse_rho(float abs_cos_theta_wi) const { return m_shading_model.diffuse_rho(abs_cos_theta_wi); }
    Math::RGB specular_rho(float abs_cos_theta_wi) const { return m_shading_model.specular_rho(abs_cos_theta_wi); }
    float coat_rho(float abs_cos_theta_wi) const { return m_shading_model.coat_rho(abs_cos_theta_wi); }

    std::string to_string() const {
        auto [tint, roughness] = m_material_params.get_tint_roughness(m_texcoord);
        std::ostringstream out;
        out << "Default shading:" << std::endl;
        out << "  Tint: " << tint.r << ", " << tint.g << ", " << tint.b << std::endl;
        out << "  Roughness: " << roughness << std::endl;
        out << "  Metalness: " << m_material_params.get_metallic(m_texcoord) << std::endl;
        out << "  Coat: " << m_material_params.get_coat() << std::endl;
        out << "  Coat roughness: " << m_material_params.get_coat_roughness() << std::endl;
        return out.str();
    }
};

Material create_gold_material() {
    return Material::create_metal("gold", gold_tint, 0.02f);
}

Material create_plastic_material() {
    return Material::create_dielectric("plastic", Math::RGB(0.02f, 0.27f, 0.33f), 0.7f);
}

Material create_coated_plastic_material() {
    Material plastic_params = create_plastic_material();
    plastic_params.set_coat(1.0f);
    plastic_params.set_coat_roughness(plastic_params.get_roughness());
    return plastic_params;
}

class Assets_Shading_ShadingModels_DefaultShadingTests : public ::testing::Test {
    // Per-test set-up and tear-down logic.
    virtual void SetUp() {
        Materials::allocate(8u);
        Textures::allocate(8u);
        Images::allocate(8u);
    }
    virtual void TearDown() {
        Materials::deallocate();
        Textures::deallocate();
        Images::deallocate();
    }
};

TEST_F(Assets_Shading_ShadingModels_DefaultShadingTests, power_conservation) {
    // A white material to stress test power_conservation.
    Material white_material = Material::create_dielectric("White", Math::RGB::white(), 1.0f);

    for (float roughness : { 0.0f, 0.5f, 0.9f })
        for (float cos_theta_o : { 0.1f, 0.4f, 0.7f, 1.0f }) {
            white_material.set_roughness(roughness);
            Math::Vector3f wo = BSDFTestUtils::w_from_cos_theta(cos_theta_o);
            auto shading_model = DefaultShading(white_material, wo.z);
            auto result = BSDFTestUtils::directional_hemispherical_reflectance_function(shading_model, wo, 16384u);
            EXPECT_RGB_EQ_EPS(result.reflectance, Math::RGB::white(), 1e-3f) << "cos_theta: " << cos_theta_o;
        }
}

/*
 * DefaultShading currently ignores the Helmholtz reciprocity rule.
TEST_F(Assets_Shading_ShadingModels_DefaultShadingTests, Helmholtz_reciprocity) {
    using namespace Bifrost::Math;

    const unsigned int MAX_SAMPLES = 128u;

    for (int i = 0; i < 10; ++i) {
        const Vector3f wo = normalize(Vector3f(float(i), 0.0f, 1.001f - float(i) * 0.1f));
        auto plastic_material = DefaultShading(plastic_parameters(), wo.z);
        for (unsigned int s = 0u; s < MAX_SAMPLES; ++s) {
            Vector3f rng_sample = Vector3f(RNG::sample02(s), float(s) / float(MAX_SAMPLES));
            BSDFSample sample = plastic_material.sample(wo, rng_sample);
            if (sample.PDF.is_valid()) {
                // Re-evaluate contribution from both directions to avoid 
                // floating point imprecission between sampling and evaluating.
                // (Yes, they can actually get quite high on smooth materials.)
                RGB f0 = plastic_material.evaluate_with_PDF(wo, sample.direction).reflectance;
                RGB f1 = plastic_material.evaluate_with_PDF(sample.direction, wo).reflectance;

                EXPECT_FLOAT_EQ_PCT(f0.r, f1.r, 0.000013f);
                EXPECT_FLOAT_EQ_PCT(f0.g, f1.g, 0.000013f);
                EXPECT_FLOAT_EQ_PCT(f0.b, f1.b, 0.000013f);
            }
        }
    }
}
*/

TEST_F(Assets_Shading_ShadingModels_DefaultShadingTests, function_consistency) {
    static auto evaluate_with_PDF_test = [](Material material_params) {
        auto wo = Math::normalize(Math::Vector3f(1.0f, 1.0f, 1.0f));

        for (float roughness : { 0.2f, 0.4f, 0.6f, 0.8f, 1.0f }) {
            material_params.set_roughness(roughness);
            auto shading_model = DefaultShadingWrapper(material_params, wo.z);
            BSDFTestUtils::shading_model_consistency_test(shading_model, wo, 32);
        }
    };

    evaluate_with_PDF_test(create_gold_material());
    evaluate_with_PDF_test(create_plastic_material());
    evaluate_with_PDF_test(create_coated_plastic_material());
}

TEST_F(Assets_Shading_ShadingModels_DefaultShadingTests, PDF_positivity) {
    static auto PDF_positivity_test = [](Material material_params) {
        for (float cos_theta_o : {-0.8f, -0.4f, 0.1f, 0.5f, 0.9f}) {
            auto wo = BSDFTestUtils::w_from_cos_theta(cos_theta_o);

            for (float roughness : { 0.2f, 0.6f, 1.0f }) {
                material_params.set_roughness(roughness);
                auto shading_model = DefaultShadingWrapper(material_params, wo.z);
                BSDFTestUtils::PDF_positivity_test(shading_model, wo, 128);
            }
        }
    };

    PDF_positivity_test(create_gold_material());
    PDF_positivity_test(create_plastic_material());
    PDF_positivity_test(create_coated_plastic_material());
}

TEST_F(Assets_Shading_ShadingModels_DefaultShadingTests, Fresnel) {
    using namespace Bifrost::Math;

    { // Test that specular reflections on non-metals are white and incident reflections are diffuse.
        float specularity = 0.0f; // Testing specularity. Physically-based fubar value.
        Material material_params = Material::create_dielectric("material", Math::RGB::red(), 0.02f, specularity);

        { // Test that incident reflectivity is red.
            Vector3f wo = Vector3f(0.0f, 0.0f, 1.0f);
            auto material = DefaultShading(material_params, wo.z);
            RGB weight = material.evaluate_with_PDF(wo, wo).reflectance;
            EXPECT_GT(weight.r, 0.0f);
            EXPECT_FLOAT_EQ_EPS(weight.g, 0.0f, 1e-6f);
            EXPECT_FLOAT_EQ_EPS(weight.b, 0.0f, 1e-6f);
        }

        { // Test that grazing angle reflectivity is white.
            Vector3f wo = normalize(Vector3f(0.0f, 1.0f, 0.001f));
            Vector3f wi = normalize(Vector3f(0.0f, -1.0f, 0.001f));
            auto material = DefaultShading(material_params, wo.z);
            RGB weight = material.evaluate_with_PDF(wo, wi).reflectance;
            EXPECT_GT(weight.r, 0.99f);
            EXPECT_FLOAT_EQ(weight.r, weight.g);
            EXPECT_FLOAT_EQ(weight.r, weight.b);
        }
    }

    { // Test that specular reflections on metals are tinted.
        Material material_params = create_gold_material();

        { // Test that incident reflectivity equals a scaled tint.
            Vector3f wo = Vector3f(0.0f, 0.0f, 1.0f);
            auto material = DefaultShading(material_params, wo.z);
            RGB weight = material.evaluate_with_PDF(wo, wo).reflectance;
            float scale = material_params.get_tint().r / weight.r;
            EXPECT_RGB_EQ_EPS(weight * scale, material_params.get_tint(), 1e-6f);
        }

        { // Test that grazing angle reflectivity is nearly white.
            Vector3f wo = normalize(Vector3f(0.0f, 1.0f, 0.001f));
            Vector3f wi = normalize(Vector3f(0.0f, -1.0f, 0.001f));
            auto material = DefaultShading(material_params, wo.z);
            RGB weight = material.evaluate_with_PDF(wo, wi).reflectance;
            EXPECT_GT(weight.g, 0.99f);
            EXPECT_FLOAT_EQ_PCT(weight.g, weight.r, 0.01f);
            EXPECT_FLOAT_EQ_PCT(weight.g, weight.b, 0.01f);
        }
    }
}

TEST_F(Assets_Shading_ShadingModels_DefaultShadingTests, directional_hemispherical_reflectance_estimation) {
    using namespace Bifrost::Math;

    // Test albedo is properly estimated.
    static auto test_albedo = [](Vector3f wo, float roughness, float metallic, float coat_strength, float coat_roughness) {
        Material material_params = Material::create_dielectric("material", RGB(1.0f, 0.5f, 0.25f), roughness);
        material_params.set_metallic(metallic);
        material_params.set_coat(coat_strength);
        material_params.set_coat_roughness(coat_roughness);
        auto shading_model = DefaultShadingWrapper(material_params, wo.z);

        RGB expected_rho = BSDFTestUtils::directional_hemispherical_reflectance_function(shading_model, wo, 8192).reflectance;
        RGB actual_rho = shading_model.rho(wo.z);

        // The error is slightly higher for low roughness materials.
        float error_percentage = 0.015f * (2 - roughness) * (2 - coat_roughness);
        EXPECT_RGB_EQ_PCT(expected_rho, actual_rho, error_percentage) << shading_model.to_string();
    };

    const Vector3f incident_wo = Vector3f(0.0f, 0.0f, 1.0f);
    const Vector3f average_wo = normalize(Vector3f(1.0f, 0.0f, 1.0f));
    const Vector3f grazing_wo = BSDFTestUtils::w_from_cos_theta(1.0f / (Rho::GGX_angle_sample_count - 1.0f)); // Map cos_theta to an angle that has been sampled in the rho precomputation

    for (Vector3f wo : { incident_wo, average_wo, grazing_wo })
        for (float roughness : { 0.25f, 0.75f })
            for (float metallic : { 0.0f, 0.5f, 1.0f })
                for (float coat : { 0.0f, 0.5f, 1.0f })
                    for (float coat_roughness : { 0.25f, 0.75f })
                        test_albedo(wo, roughness, metallic, coat, coat_roughness);
}

TEST_F(Assets_Shading_ShadingModels_DefaultShadingTests, white_hot_room) {
    // A white material to stress test power_conservation.
    Material white_material_params = Material::create_dielectric("material", Bifrost::Math::RGB::white(), 0.0f);

    for (float metallic : { 0.0f, 0.5f, 1.0f }) {
        white_material_params.set_metallic(metallic);
        for (float roughness : { 0.0f, 0.5f, 1.0f }) {
            white_material_params.set_roughness(roughness);

            for (int a = 0; a < 5; ++a) {
                float abs_cos_theta = 1.0f - float(a) * 0.2f;
                auto material = DefaultShadingWrapper(white_material_params, abs_cos_theta);
                EXPECT_FLOAT_EQ(1.0f, material.rho(abs_cos_theta).r) << material.to_string();
            }
        }
    }
}

TEST_F(Assets_Shading_ShadingModels_DefaultShadingTests, sampling_probability_match_reflectance_contribution) {
    using namespace Bifrost::Math;

    for (float cos_theta_o : { 0.5f, 1.0f })
        for (float roughness : { 0.25f, 0.75f })
            for (float metallic : { 0.0f, 1.0f })
                for (float coat_strength : { 0.0f, 0.5f, 1.0f })
                    for (float coat_roughness : { 0.25f, 0.75f }) {
                        Material material_params = Material::create_dielectric("material", RGB::white(), roughness);
                        material_params.set_metallic(metallic);
                        material_params.set_coat(coat_strength);
                        material_params.set_coat_roughness(coat_roughness);
                        auto material = DefaultShadingWrapper(material_params, cos_theta_o);

                        RGB diffuse_rho = material.diffuse_rho(cos_theta_o);
                        RGB specular_rho = material.specular_rho(cos_theta_o);
                        RGB coat_rho = RGB(material.coat_rho(cos_theta_o));
                        RGB total_rho = material.rho(cos_theta_o);

                        float expected_diffuse_contribution = sum(diffuse_rho) / sum(total_rho);
                        float expected_specular_contribution = sum(specular_rho) / sum(total_rho);
                        float expected_coat_contribution = sum(coat_rho) / sum(total_rho);

                        float actual_diffuse_probability = material.get_diffuse_probability();
                        float actual_specular_probability = material.get_specular_probability();
                        float actual_coat_probability = material.get_coat_probability();
                        float total_actual_probability = actual_diffuse_probability + actual_specular_probability + actual_coat_probability;

                        EXPECT_FLOAT_EQ_EPS(total_actual_probability, 1.0f, 2e-5f); // The BRDF layer probabilities sum to 1
                        EXPECT_FLOAT_EQ_EPS(expected_diffuse_contribution, actual_diffuse_probability, 2e-5f);
                        EXPECT_FLOAT_EQ_EPS(expected_specular_contribution, actual_specular_probability, 2e-5f);
                        EXPECT_FLOAT_EQ_EPS(expected_coat_contribution, actual_coat_probability, 2e-5f);
                    }
}

TEST_F(Assets_Shading_ShadingModels_DefaultShadingTests, metallic_interpolation) {
    using namespace Bifrost::Math;

    Material material_params = Material::create_dielectric("material", RGB(1.0f, 0.5f, 0.25f), 0.0f);

    for (float roughness : { 0.0f, 0.5f, 1.0f }) {
        material_params.set_roughness(roughness);
        for (float metallic : { 0.25f, 0.5f, 0.75f }) {
            for (float abs_cos_theta : { 0.2f, 0.4f, 0.6f, 0.8f, 1.0f }) {

                material_params.set_metallic(metallic);
                auto material = DefaultShadingWrapper(material_params, abs_cos_theta);
                RGB rho = material.rho(abs_cos_theta);

                material_params.set_metallic(0);
                auto dielectric_material = DefaultShading(material_params, abs_cos_theta);
                RGB dielectric_rho = dielectric_material.rho(abs_cos_theta);

                material_params.set_metallic(1);
                auto conductor_material = DefaultShading(material_params, abs_cos_theta);
                RGB conductor_rho = conductor_material.rho(abs_cos_theta);

                // Test that the directional-hemispherical reflectance of the semi-metallic material equals
                // the one evaluated by interpolating a fully dielectric and fully conductor material.
                RGB interpolated_rho = lerp(dielectric_rho, conductor_rho, metallic);
                EXPECT_RGB_EQ_EPS(interpolated_rho, rho, 1e-6f) << material.to_string();
            }
        }
    }
}

// Helper function to generate a coated material with a specific target specular roughness.
// The input roughness for the material is found through a binary search.
DefaultShading generate_interpolated_coated_material(Material material_params, float cos_theta, float target_roughness) {

    double roughness = 0.5;
    double roughness_adjustment = 0.25;
    material_params.set_roughness((float)roughness);
    do {
        auto material = DefaultShading(material_params, cos_theta);
        float roughness_delta = abs(material.get_roughness() - target_roughness);
        if (roughness_delta < 1e-8f)
            return material;

        bool decrease_roughness = material.get_roughness() > target_roughness;
        roughness += decrease_roughness ? -roughness_adjustment : roughness_adjustment;
        if (material_params.get_roughness() == (float)roughness)
            return material;

        material_params.set_roughness((float)roughness);
        roughness_adjustment *= 0.5;
    } while (true);
}

// Test that a partial coat is the linear interpolation of the material with no coat and full coat.
// A rough coat will modulate the roughness of the base layer. Strictly speaking this breaks
// the linear interpolation property of the coat, but that is how the material is defined.
// A coated material and a non-coated material can still be linearly interpolated,
// if we ensure that they both have the same roughness after being created.
// The specularity of the base material will be affected by the coat, as the base is now viewed through a coat instead of air.
// This means that interpolated rho values are not exact, but the difference is expected
// and expressed as an acceptable deviation in final rho values.
TEST_F(Assets_Shading_ShadingModels_DefaultShadingTests, coat_interpolation) {
    using namespace Bifrost::Math;

    Material material_params = create_plastic_material();
    material_params.set_specularity(0.02f);

    for (float coat_roughness : { 0.0f, 0.5f, 1.0f }) {
        for (float cos_theta : { 0.2f, 0.4f, 0.6f, 0.8f, 1.0f }) {
            material_params.set_coat(1.0f);
            material_params.set_coat_roughness(coat_roughness);

            auto coated_material = DefaultShading(material_params, cos_theta);
            float coated_specularity = coated_material.get_specularity().r;
            RGB coated_rho = coated_material.rho(cos_theta);

            // Verify that material roughness is lower when the coat isn't perfectly smooth.
            if (coat_roughness > 0.0)
                EXPECT_LT(material_params.get_roughness(), coated_material.get_roughness());

            material_params.set_coat(0.0f);
            material_params.set_roughness(coated_material.get_roughness());
            auto non_coated_material = DefaultShading(material_params, cos_theta);
            float non_coated_specularity = non_coated_material.get_specularity().r;
            RGB non_coated_rho = non_coated_material.rho(cos_theta);

            // Verify that the two materials have the same roughness.
            EXPECT_FLOAT_EQ(non_coated_material.get_roughness(), coated_material.get_roughness());

            for (float coat_strength : { 0.25f, 0.5f, 0.75f }) {
                // Generate a material with a partial coat that has the same roughness as the coated material.
                material_params.set_coat(coat_strength);
                auto material = generate_interpolated_coated_material(material_params, cos_theta, coated_material.get_roughness());
                float interpolated_specularity = material.get_specularity().r;
                RGB interpolated_rho = material.rho(cos_theta);

                // Verify that the two materials have the same roughness.
                EXPECT_FLOAT_EQ_EPS(material.get_roughness(), coated_material.get_roughness(), 1e-6f);

                // Verify that the interpolated materials specularity is bounded by the coated and non-coated specularity.
                EXPECT_LE(interpolated_specularity, non_coated_specularity);
                EXPECT_GE(interpolated_specularity, coated_specularity);

                // Test that the directional-hemispherical reflectance of the semi-coated material equals
                // the one evaluated by interpolating between a material with no coat and coated material.
                float acceptable_pct_deviation = 0.01f;
                RGB expected_rho = lerp(non_coated_rho, coated_rho, coat_strength);
                EXPECT_RGB_EQ_PCT(expected_rho, interpolated_rho, acceptable_pct_deviation);
            }
        }
    }
}

TEST_F(Assets_Shading_ShadingModels_DefaultShadingTests, regression_test) {
    using namespace Bifrost::Math;

    const unsigned int MAX_SAMPLES = 2;

    Material materials[3] = { create_gold_material(), create_plastic_material(), create_coated_plastic_material() };
    Vector3f wos[3] = { Vector3f(0.0f, 0.0f, 1.0f), normalize(Vector3f(1.0f, 0.0f, 1.0f)), normalize(Vector3f(1.0f, 0.0f, 0.01f)) };

    BSDFResponse bsdf_responses[] = {
        // Gold
        { { 497358.250000f, 380976.437500f, 167112.35938f }, 497357.968750f }, { { 124339.328125f, 95243.929688f, 41778.01172f }, 124339.226562f },
        { { 994714.312500f, 762452.875000f, 335647.65625f }, 703369.500000f }, { { 249079.953125f, 190921.484375f, 84049.07812f }, 175985.125000f },
        { { 4957683712.0f, 4900780032.0f, 4796214272.0f }, 49668972.0f }, { { 1455754624.0f, 1439689728.0f, 1410168448.0f }, 13442245.0f },
        // Plastic
        { { 0.017321f, 0.080929f, 0.09619f }, 0.015246f }, { { 0.018466f, 0.096538f, 0.11528f }, 0.228278f },
        { { 0.016705f, 0.124393f, 0.15024f }, 0.034463f }, { { 0.013662f, 0.121644f, 0.14756f }, 0.239035f },
        { { 0.064515f, 0.097390f, 0.10528f }, 0.302231f }, { { 0.017234f, 0.145679f, 0.17651f }, 0.210165f },
        // Coated plastic
        { { 0.017964f, 0.080150f, 0.09507f }, 0.015335f }, { { 0.018487f, 0.096746f, 0.11553f }, 0.229421f },
        { { 0.016956f, 0.127775f, 0.15437f }, 0.037424f }, { { 0.013921f, 0.125297f, 0.15203f }, 0.241837f },
        { { 0.086971f, 0.113758f, 0.12019f }, 0.316902f }, { { 0.017554f, 0.147099f, 0.17819f }, 0.192900f } };

    int response_index = 0;
    for (int i = 0; i < 3; ++i)
        for (Vector3f wo : wos) {
            auto material = DefaultShading(materials[i], wo.z);
            for (int s = 0; s < MAX_SAMPLES; ++s) {
                Vector3f rng_sample = Vector3f(RNG::sample02(s), (s + 0.5f) / MAX_SAMPLES);
                BSDFSample sample = material.sample(wo, rng_sample);
                // printf("{ { %.6ff, %.6ff, %.5ff }, %.6ff },\n", sample.reflectance.r, sample.reflectance.g, sample.reflectance.b, sample.PDF.value());
                auto response = bsdf_responses[response_index++];

                EXPECT_RGB_EQ_PCT(response.reflectance, sample.reflectance, 0.0001f);
                EXPECT_PDF_EQ_PCT(response.PDF, sample.PDF, 0.0001f);
            }
        }
}

} // NS Bifrost::Assets::Shading::ShadingModels

#endif // _BIFROST_ASSETS_SHADING_SHADING_MODELS_DEFAULT_TEST_H_