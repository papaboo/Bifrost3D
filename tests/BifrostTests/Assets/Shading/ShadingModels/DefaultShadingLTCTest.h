// Test Bifrost's LTC approximation of the default shading model.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _BIFROST_ASSETS_SHADING_SHADING_MODELS_DEFAULT_LTC_TEST_H_
#define _BIFROST_ASSETS_SHADING_SHADING_MODELS_DEFAULT_LTC_TEST_H_

#include <Assets/Shading/ShadingModels/DefaultShadingTest.h> // To the DefaultShadingWrapper

#include <Bifrost/Assets/Shading/LightSources/TriangleLight.h>

namespace Bifrost::Assets::Shading::ShadingModels {

class Assets_Shading_ShadingModels_DefaultShadingLTCTests : public ::testing::Test {
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

TEST_F(Assets_Shading_ShadingModels_DefaultShadingLTCTests, ltc_stack_only_includes_non_zero_contributing_BRDFs) {
    float cos_theta_o = 1.0f;

    { // Metals only consist of a tinted specular layer and can be represented by a single LTC.
        Material gold_material = create_gold_material();
        auto gold_shading_model = DefaultShadingWrapper(gold_material, cos_theta_o);
        auto gold_LTCs = gold_shading_model.get_LTC_representation(cos_theta_o);
        EXPECT_EQ(1, gold_LTCs.count);
        EXPECT_RGB_GT(gold_LTCs[0].tint, 0);
    }

    { // Dielectrics are represented by a diffuse and specular LTC.
        Material dielectric_material = create_plastic_material();
        auto dielectric_shading_model = DefaultShadingWrapper(dielectric_material, cos_theta_o);
        auto dielectric_LTCs = dielectric_shading_model.get_LTC_representation(cos_theta_o);
        EXPECT_EQ(2, dielectric_LTCs.count);
        EXPECT_RGB_GT(dielectric_LTCs[0].tint, 0);
        EXPECT_RGB_GT(dielectric_LTCs[1].tint, 0);
    }

    { // Coated dielectrics are represented by a diffuse, specular and coat LTC.
        Material coated_material = create_coated_plastic_material();
        auto coated_shading_model = DefaultShadingWrapper(coated_material, cos_theta_o);
        auto coated_LTCs = coated_shading_model.get_LTC_representation(cos_theta_o);
        EXPECT_EQ(3, coated_LTCs.count);
        EXPECT_RGB_GT(coated_LTCs[0].tint, 0);
        EXPECT_RGB_GT(coated_LTCs[1].tint, 0);
        EXPECT_RGB_GT(coated_LTCs[2].tint, 0);
    }
}

TEST_F(Assets_Shading_ShadingModels_DefaultShadingLTCTests, LTC_triangle_light_error) {
    using namespace Bifrost::Math;

    static auto test_ltc_triangle_error = [](Vector3f wo, Material material_params, float expected_error) {

        // Create a two-sided light with emitted radiance set to white.
        Trianglef light_surface = Trianglef(Vector3f(0, 0, 0), Vector3f(0, 1, 0), Vector3f(1, 0, 0));
        RGB emitted_radiance = RGB::white();
        bool is_two_sided = false;
        RGB power = emitted_radiance * PIf * light_surface.get_surface_area();
        auto light = LightSources::TriangleLight(light_surface, power, is_two_sided);

        // Define the surface plane to illuminate.
        // The surface passes through origo and the normal is along positive z.
        Vector3f surface_point = { 0, 0, -1 };
        Vector3f surface_normal = { 0, 0, 1 };

        float cos_theta_o = wo.z;
        auto shading = DefaultShadingWrapper(material_params, cos_theta_o);

        // Integrate over light source
        RGB monte_carlo_estimation = BSDFTestUtils::integrate_light_over_surface(wo, surface_point, surface_normal, shading, light, 4096);

        // Approximate light source with LTC
        auto LTC_shading = shading.get_LTC_representation(cos_theta_o);
        RGB ltc_estimation = RGB::black();
        for (int bsdf_index = 0; bsdf_index < LTC_shading.count; ++bsdf_index) {
            auto ltc_bsdf = LTC_shading[bsdf_index];
            ltc_estimation += ltc_bsdf.tint * light.evaluate(ltc_bsdf.shading_to_ltc, wo, surface_point, surface_normal);
        }

        // Compare error
        RGB error_diff = monte_carlo_estimation - ltc_estimation;
        float estimation_error = (abs(error_diff.r) + abs(error_diff.g) + abs(error_diff.b)) / 3.0f;
        EXPECT_FLOAT_EQ_EPS(expected_error, estimation_error, 0.0001f);
    };

    // View directions
    Vector3f wos[3] = { Vector3f(0, 0, 1), normalize(Vector3f(1, 0, 1)), normalize(Vector3f(1, 4, 1)) };

    Material gold_material = create_gold_material();
    gold_material.set_roughness(0.25f); // Increase roughness of the smooth gold material to always reflect something.
    test_ltc_triangle_error(wos[0], gold_material, 0.000447f);
    test_ltc_triangle_error(wos[1], gold_material, 0.000301f);
    test_ltc_triangle_error(wos[2], gold_material, 0.000906f);

    Material dielectric_material = create_plastic_material();
    test_ltc_triangle_error(wos[0], dielectric_material, 0.000242f);
    test_ltc_triangle_error(wos[1], dielectric_material, 0.000210f);
    test_ltc_triangle_error(wos[2], dielectric_material, 0.001734f);

    Material coated_material = create_coated_plastic_material();
    test_ltc_triangle_error(wos[0], coated_material, 0.000255f);
    test_ltc_triangle_error(wos[1], coated_material, 0.000453f);
    test_ltc_triangle_error(wos[2], coated_material, 0.003683f);
}

} // NS Bifrost::Assets::Shading::ShadingModels

#endif // _BIFROST_ASSETS_SHADING_SHADING_MODELS_DEFAULT_LTC_TEST_H_