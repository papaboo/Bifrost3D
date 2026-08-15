// Test Bifrost shading utilities.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _BIFROST_ASSETS_SHADING_UTILS_TEST_H_
#define _BIFROST_ASSETS_SHADING_UTILS_TEST_H_

#include <Bifrost/Assets/Shading/Constants.h>
#include <Bifrost/Assets/Shading/Utils.h>
#include <Bifrost/Math/Distributions.h>
#include <Bifrost/Math/RNG.h>

#include <Expects.h>

namespace Bifrost::Assets::Shading {

GTEST_TEST(Assets_Shading_Specularity, dielectric_conversions_to_and_from_index_of_refraction) {
    // Index of refraction
    const float air_ior = 1.0f;
    const float water_ior = 1.333f;
    const float glass_ior = 1.50f;

    // Specularity of medium when transitioning from air to medium.
    const float water_specularity = 0.02037318784f;
    const float glass_specularity = 0.04f;

    // Test conversion from index of refraction to specularity.
    float computed_water_specularity = dielectric_specularity(air_ior, water_ior);
    float computed_glass_specularity = dielectric_specularity(air_ior, glass_ior);

    EXPECT_FLOAT_EQ(water_specularity, computed_water_specularity);
    EXPECT_FLOAT_EQ(glass_specularity, computed_glass_specularity);

    // Test conversion from specularity to index of refraction
    float computed_water_ior = dielectric_ior_from_specularity(water_specularity);
    float computed_glass_ior = dielectric_ior_from_specularity(glass_specularity);

    EXPECT_FLOAT_EQ(water_ior, computed_water_ior);
    EXPECT_FLOAT_EQ(glass_ior, computed_glass_ior);
}

GTEST_TEST(Assets_Shading_Specularity, conductor_conversions_to_and_from_index_of_refraction) {
    float accuracy = 1e-5f;

    const Math::RGB air_ior = { 1.0f, 1.0f, 1.0f };

    // Specularity of medium when transitioning from air to medium at wavelengths 630nm (red), 532nm (green) and 465nm (blue)
    const Math::RGB gold_specularity = { 0.932999f, 0.687356f, 0.384839f };
    const Math::RGB titanium_specularity = { 0.61167696422f, 0.57501477894f, 0.54852055032f };

    // Test conversion from index of refraction to specularity.
    Math::RGB computed_gold_specularity = conductor_specularity(air_ior, gold_ior, gold_extinction);
    Math::RGB computed_titanium_specularity = conductor_specularity(air_ior, titanium_ior, titanium_extinction);

    EXPECT_RGB_EQ_PCT(gold_specularity, computed_gold_specularity, accuracy);
    EXPECT_RGB_EQ_PCT(titanium_specularity, computed_titanium_specularity, accuracy);

    // Test conversion from specularity to index of refraction
    Math::RGB computed_gold_ior = conductor_ior_from_specularity(gold_specularity, gold_extinction);
    Math::RGB computed_titanium_ior = conductor_ior_from_specularity(titanium_specularity, titanium_extinction);

    EXPECT_RGB_EQ_PCT(gold_ior, computed_gold_ior, accuracy);
    EXPECT_RGB_EQ_PCT(titanium_ior, computed_titanium_ior, accuracy);
}

GTEST_TEST(Assets_Shading_Specularity, scaling_dielectric_specularity_under_coat) {
    // adjust_dielectric_specularity_to_exterior_medium makes the assumption that air's Index of Refraction is exactly 1.
    const float air_ior = 1.0f;

    for (float base_ior : { ice_ior, coat_ior, diamond_ior }) {
        // The expected specularity of the base material viewed through the coat instead of air.
        float expected_base_specularity_through_coat = dielectric_specularity(coat_ior, base_ior);

        float base_specularity_through_air = dielectric_specularity(air_ior, base_ior);
        float actual_base_specularity_through_coat = adjust_dielectric_specularity_to_exterior_medium(coat_ior, base_specularity_through_air);

        EXPECT_FLOAT_EQ_EPS(expected_base_specularity_through_coat, actual_base_specularity_through_coat, 1e-7f);
    }
}

GTEST_TEST(Assets_Shading_Specularity, scaling_conductor_specularity_under_coat) {
    const Math::RGB air_ior_3 = Math::RGB(air_ior);
    const Math::RGB coat_ior_3 = Math::RGB(coat_ior);

    for (Math::RGB base_ior : { gold_ior, titanium_ior }) {
        for (Math::RGB base_extinction : { gold_extinction, titanium_extinction }) {
            // The expected specularity of the base material viewed through the coat instead of air.
            Math::RGB expected_base_specularity_through_coat = conductor_specularity(coat_ior_3, base_ior, base_extinction);

            Math::RGB base_specularity_through_air = conductor_specularity(air_ior_3, base_ior, base_extinction);
            Math::RGB actual_base_specularity_through_coat = adjust_conductor_specularity_to_exterior_medium(coat_ior_3, base_specularity_through_air, base_extinction);

            EXPECT_RGB_EQ_EPS(expected_base_specularity_through_coat, actual_base_specularity_through_coat, 0.02f);
        }
    }
}

GTEST_TEST(Assets_Shading_Trigonometry, refract_overloads_gives_same_result_as_full_implementation) {
    Math::Vector3f up = Math::Vector3f(0, 0, 1);

    for (int wo_s = 0; wo_s < 16; wo_s++) {
        Math::Vector3f wo = Math::Distributions::UniformHemisphere::sample(Math::RNG::sample02(wo_s)).direction;
        for (float ior_i_over_o : { 0.33f, 0.7f, 1.5f, 3.0f }) {
            Math::Vector3f expected_direction;
            bool expected_success = refract(expected_direction, wo, up, ior_i_over_o);

            { // Test vector implementation
                Math::Vector3f actual_refraction;
                bool actual_success = refract(actual_refraction, wo, ior_i_over_o);

                EXPECT_EQ(expected_success, actual_success);
                if (actual_success) // The reference implementation returns the zero vector if refraction failed, while the OptiXRenderer implementation returns an undefined result.
                    EXPECT_VECTOR3F_EQ_EPS(expected_direction, actual_refraction, 1e-6f);
            }

            { // Test angle implementation
                float actual_refracted_cos_theta;
                bool actual_success = refract(actual_refracted_cos_theta, wo.z, ior_i_over_o);

                EXPECT_EQ(expected_success, actual_success);
                if (actual_success) // The reference implementation returns the zero vector if refraction failed, while the OptiXRenderer implementation returns an undefined result.
                    EXPECT_FLOAT_EQ_EPS(expected_direction.z, actual_refracted_cos_theta, 1e-6f);
            }
        }
    }
}

} // NS Bifrost::Assets::Shading

#endif // _BIFROST_ASSETS_SHADING_UTILS_TEST_H_
