// Test OptiXRenderer's rough glass shading model.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _OPTIXRENDERER_SHADING_MODEL_ROUGH_GLASS_TEST_H_
#define _OPTIXRENDERER_SHADING_MODEL_ROUGH_GLASS_TEST_H_

#include <ShadingModels/ShadingModelTestUtils.h>
#include <Utils.h>

#include <OptiXRenderer/RNG.h>
#include <OptiXRenderer/Shading/ShadingModels/RoughGlassShading.h>
#include <OptiXRenderer/Utils.h>

#include <gtest/gtest.h>

namespace OptiXRenderer {

Material smooth_glass_parameters() {
    Material glass_params = {};
    glass_params.tint = optix::make_float3(0.95f, 0.97f, 0.95f);
    glass_params.roughness = 0.0f;
    glass_params.metallic = 0.0f;
    glass_params.specularity = 0.04f;
    return glass_params;
}

Material frosted_glass_parameters() {
    Material glass_params = smooth_glass_parameters();
    glass_params.roughness = 0.2f;
    return glass_params;
}

class RoughGlassShadingWrapper {
public:
    Material m_material_params;
    Shading::ShadingModels::RoughGlassShading m_shading_model;

    RoughGlassShadingWrapper(const Material& material_params, float cos_theta)
        : m_material_params(material_params), m_shading_model(material_params, cos_theta) {}

    BSDFResponse evaluate_with_PDF(optix::float3 wo, optix::float3 wi) const { return m_shading_model.evaluate_with_PDF(wo, wi); }

    BSDFSample sample(optix::float3 wo, optix::float3 random_sample) const { return m_shading_model.sample(wo, random_sample); }

    std::string to_string() const {
        std::ostringstream out;
        out << "Rough glass shading:" << std::endl;
        out << "  Tint: " << m_material_params.tint.x << ", " << m_material_params.tint.y << ", " << m_material_params.tint.z << std::endl;
        out << "  Roughness:" << m_material_params.roughness << std::endl;
        return out.str();
    }
};
GTEST_TEST(RoughGlassShadingModel, power_conservation) {
    using namespace Shading::ShadingModels;

    // A white material to stress test power_conservation.
    Material material_params = {};
    material_params.tint = optix::make_float3(1.0f, 1.0f, 1.0f);
    material_params.specularity = 0.02f;

    for (float roughness : { 0.2f, 0.7f }) {
        material_params.roughness = roughness;
        for (float cos_theta_o : { -1.0f, -0.7f, -0.4f, -0.1f, 0.1f, 0.4f, 0.7f, 1.0f }) {
            optix::float3 wo = BSDFTestUtils::w_from_cos_theta(cos_theta_o);
            auto shading_model = RoughGlassShading(material_params, wo.z);
            auto rho = ShadingModelTestUtils::directional_hemispherical_reflectance_function(shading_model, wo).reflectance;
            EXPECT_LE(rho.x, 1.0f) << " with roughness " << roughness << " and cos_theta " << cos_theta_o;
        }
    }
}

GTEST_TEST(RoughGlassShadingModel, function_consistency) {
    using namespace optix;

    for (float cos_theta_o : { -1.0f, -0.7f, -0.4f, -0.1f, 0.1f, 0.4f, 0.7f, 1.0f }) {
        float3 wo = BSDFTestUtils::w_from_cos_theta(cos_theta_o);

        Material material_params = smooth_glass_parameters();
        for (float roughness : { 0.2f, 0.4f, 0.6f, 0.8f, 1.0f }) {
            material_params.roughness = roughness;
            auto shading_model = RoughGlassShadingWrapper(material_params, wo.z);
            ShadingModelTestUtils::consistency_test(shading_model, wo, 32);
        }
    }
}

GTEST_TEST(RoughGlassShadingModel, PDF_positivity) {
    for (float cos_theta_o : {-0.8f, -0.4f, 0.1f, 0.5f, 0.9f}) {
        optix::float3 wo = BSDFTestUtils::w_from_cos_theta(cos_theta_o);

        Material material_params = smooth_glass_parameters();
        for (float roughness : { 0.2f, 0.6f, 1.0f }) {
            material_params.roughness = roughness;
            auto shading_model = RoughGlassShadingWrapper(material_params, wo.z);
            BSDFTestUtils::PDF_positivity_test(shading_model, wo, 1024);
        }
    }
}

GTEST_TEST(RoughGlassShadingModel, Fresnel) {
    using namespace Shading::ShadingModels;
    using namespace optix;

    // Test that specular reflections are white and incident reflections are black, i.e. all light is transmitted, when specularity is 0.
    Material material_params = smooth_glass_parameters();
    material_params.specularity = 0.0f; // Testing specularity. Physically-based fubar value.

    { // Test that incident reflectivity is black.
        float3 wo = make_float3(0.0f, 0.0f, 1.0f);
        auto shading_model = RoughGlassShading(material_params, wo.z);
        float3 weight = shading_model.evaluate_with_PDF(wo, wo).reflectance;
        EXPECT_FLOAT_EQ(weight.x, 0.0f);
        EXPECT_FLOAT_EQ(weight.y, 0.0f);
        EXPECT_FLOAT_EQ(weight.z, 0.0f);
    }

    { // Test that grazing angle reflectivity is white.
        float3 wo = normalize(make_float3(0.0f, 1.0f, 0.001f));
        float3 wi = normalize(make_float3(0.0f, -1.0f, 0.001f));
        auto shading_model = RoughGlassShading(material_params, wo.z);
        float3 weight = shading_model.evaluate_with_PDF(wo, wi).reflectance;
        EXPECT_GT(weight.x, 0.99f);
        EXPECT_FLOAT_EQ(weight.x, weight.y);
        EXPECT_FLOAT_EQ(weight.x, weight.z);
    }
}

GTEST_TEST(RoughGlassShadingModel, snells_law) {
    using namespace optix;
    using namespace Shading::ShadingModels;

    float transmission_random_sample = 1; // We know that if we pass 1 as the third random value, then the BTDF will always be sampled.
    float ior_o = 1;
    float ior_i = 2;

    Material material_params = smooth_glass_parameters();
    material_params.specularity = dielectric_specularity(ior_o, ior_i);

    for (float cos_theta_o : { 0.2f, 0.5f, 0.9f }) {
        float3 wo = BSDFTestUtils::w_from_cos_theta(cos_theta_o);
        auto shading_model = RoughGlassShading(material_params, wo.z);

        float3 wi = shading_model.sample(wo, make_float3(0.5f, 0.5f, transmission_random_sample)).direction;

        // Test that wi was sampled as a transmission.
        EXPECT_LT(wi.z, 0.0f);

        float sin_theta_o = sin_theta(wo);
        float sin_theta_i = sin_theta(wi);

        EXPECT_FLOAT_EQ_EPS(ior_o * sin_theta_o, ior_i * sin_theta_i, 1e-6f);
    }
}

GTEST_TEST(RoughGlassShadingModel, regression_test) {
    using namespace Shading::ShadingModels;
    using namespace optix;

    const unsigned int MAX_SAMPLES = 2;

    BSDFResponse bsdf_responses[] = {
        {400.362793f, 400.362793f, 400.362793f, 83.876236f},
        {77.502472f, 79.134109f, 77.502472f, 63.220791f},
        {904.332703f, 923.371338f, 904.332703f, 805.072449f},
        {220.748367f, 225.395706f, 220.748367f, 199.066086f},
        {1632.925171f, 1667.302612f, 1632.925171f, 1718.868652f},
        {408.665649f, 417.269165f, 408.665649f, 430.136108f} };

    Material material_params = frosted_glass_parameters();
    int response_index = 0;
    for (float cos_theta_o : { 0.2f, 0.6f, 1.0f }) {
        float3 wo = BSDFTestUtils::w_from_cos_theta(cos_theta_o);
        auto material = RoughGlassShading(material_params, wo.z);
        for (int s = 0; s < MAX_SAMPLES; ++s) {
            float3 rng_sample = make_float3(RNG::sample02(s), (s + 0.5f) / MAX_SAMPLES);
            BSDFSample sample = material.sample(wo, rng_sample);
            // printf("{%.6ff, %.6ff, %.6ff, %.6ff},\n", sample.reflectance.x, sample.reflectance.y, sample.reflectance.z, sample.PDF);
            auto response = bsdf_responses[response_index++];

            EXPECT_COLOR_EQ_PCT(response.reflectance, sample.reflectance, 0.0001f);
            EXPECT_FLOAT_EQ_PCT(response.PDF, sample.PDF, 0.0001f);
        }
    }
}

} // NS OptiXRenderer

#endif // _OPTIXRENDERER_SHADING_MODEL_ROUGH_GLASS_TEST_H_