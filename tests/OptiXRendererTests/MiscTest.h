// Test miscellaneous parts of the OptiXRenderer.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _OPTIXRENDERER_MISC_TEST_H_
#define _OPTIXRENDERER_MISC_TEST_H_

#include <Utils.h>

#include <Bifrost/Math/OctahedralNormal.h>

#include <OptiXRenderer/Distributions.h>
#include <OptiXRenderer/RNG.h>

#include <gtest/gtest.h>

#include <limits.h>

namespace OptiXRenderer {

// Index of refraction and extinction coefficient for mediums at wavelengths 630nm (red), 532nm (green) and 465nm (blue)
const optix::float3 gold_ior = { 0.1986f, 0.54463f, 1.2515f };
const optix::float3 gold_extinction = { 3.228f, 2.1406f, 1.7517f };
const optix::float3 titanium_ior = { 2.6979f, 2.4793f, 2.3050f };
const optix::float3 titanium_extinction = { 3.7571f, 3.3511f, 3.0820f };

GTEST_TEST(OctahedralNormal, equality_with_bifrost_implementation) {
    using namespace Bifrost;

    for (int x = -10; x < 11; ++x)
        for (int y = -10; y < 11; ++y)
            for (int z = -10; z < 11; ++z) {
                if (x == 0 && y == 0 && z == 0)
                    continue;
                
                optix::float3 normal = optix::normalize(optix::make_float3(float(x), float(y), float(z)));
                Math::OctahedralNormal bifrost_encoded_normal = Math::OctahedralNormal::encode_precise(normal.x, normal.y, normal.z);
                Math::Vector3f bifrost_decoded_normal = bifrost_encoded_normal.decode();

                OctahedralNormal optix_encoded_normal = { bifrost_encoded_normal.encoding.x, bifrost_encoded_normal.encoding.y };
                optix::float3 optix_decoded_normal = optix_encoded_normal.decode();

                EXPECT_FLOAT_EQ(bifrost_decoded_normal.x, optix_decoded_normal.x);
                EXPECT_FLOAT_EQ(bifrost_decoded_normal.y, optix_decoded_normal.y);
                EXPECT_FLOAT_EQ(bifrost_decoded_normal.z, optix_decoded_normal.z);
            }
}

GTEST_TEST(Specularity, dielectric_conversions_to_and_from_index_of_refraction) {
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

GTEST_TEST(Specularity, conductor_conversions_to_and_from_index_of_refraction) {
    using namespace optix;
    float accuracy = 1e-5f;

    const float3 air_ior = { 1.0f, 1.0f, 1.0f };

    // Specularity of medium when transitioning from air to medium at wavelengths 630nm (red), 532nm (green) and 465nm (blue)
    const float3 gold_specularity = { 0.932999f, 0.687356f, 0.384839f };
    const float3 titanium_specularity = { 0.61167696422f, 0.57501477894f, 0.54852055032f };

    // Test conversion from index of refraction to specularity.
    float3 computed_gold_specularity = conductor_specularity(air_ior, gold_ior, gold_extinction);
    float3 computed_titanium_specularity = conductor_specularity(air_ior, titanium_ior, titanium_extinction);

    EXPECT_FLOAT3_EQ_PCT(gold_specularity, computed_gold_specularity, accuracy);
    EXPECT_FLOAT3_EQ_PCT(titanium_specularity, computed_titanium_specularity, accuracy);

    // Test conversion from specularity to index of refraction
    float3 computed_gold_ior = conductor_ior_from_specularity(gold_specularity, gold_extinction);
    float3 computed_titanium_ior = conductor_ior_from_specularity(titanium_specularity, titanium_extinction);

    EXPECT_FLOAT3_EQ_PCT(gold_ior, computed_gold_ior, accuracy);
    EXPECT_FLOAT3_EQ_PCT(titanium_ior, computed_titanium_ior, accuracy);
}

GTEST_TEST(Specularity, scaling_dielectric_specularity_under_coat) {
    using namespace optix;

    const float air_ior = 1.0f;
    const float coat_ior = 1.5f;

    for (float base_ior : { 1.0f, 1.2f, 1.4f }) {
        // The expected specularity of the base material viewed through the coat instead of air.
        float expected_base_specularity_through_coat = dielectric_specularity(coat_ior, base_ior);

        float base_specularity_through_air = dielectric_specularity(air_ior, base_ior);
        float actual_base_specularity_through_coat = adjust_dielectric_specularity_to_exterior_medium(coat_ior, base_specularity_through_air);

        EXPECT_FLOAT_EQ_EPS(expected_base_specularity_through_coat, actual_base_specularity_through_coat, 1e-7f);
    }
}

GTEST_TEST(Specularity, scaling_conductor_specularity_under_coat) {
    using namespace optix;

    const float3 air_ior = { 1.0f, 1.0f, 1.0f };
    const float3 coat_ior = { 1.5f, 1.5f, 1.5f };

    for (float3 base_ior : { gold_ior, titanium_ior }) {
        for (float3 base_extinction : { gold_extinction, titanium_extinction }) {
            // The expected specularity of the base material viewed through the coat instead of air.
            float3 expected_base_specularity_through_coat = conductor_specularity(coat_ior, base_ior, base_extinction);

            float3 base_specularity_through_air = conductor_specularity(air_ior, base_ior, base_extinction);
            float3 actual_base_specularity_through_coat = adjust_conductor_specularity_to_exterior_medium(coat_ior, base_specularity_through_air, base_extinction);

            EXPECT_FLOAT3_EQ_EPS(expected_base_specularity_through_coat, actual_base_specularity_through_coat, 0.02f);
        }
    }
}

GTEST_TEST(Trigonometry, fix_backfacing_shading_normal) {
    using namespace optix;

    float3 normal = { 0, 0, 1 };
    float3 wo_in_hemisphere = normalize(make_float3(1, 0, 1));
    float3 wo_orthogonal = { 1, 0, 0 };
    float3 wo_below_hemipshere = normalize(make_float3(1, 0, -0.1f));

    float3 uncorrected_normal = fix_backfacing_shading_normal(wo_in_hemisphere, normal);
    EXPECT_FLOAT3_EQ(normal, uncorrected_normal);

    uncorrected_normal = fix_backfacing_shading_normal(wo_orthogonal, normal);
    EXPECT_FLOAT3_EQ(normal, uncorrected_normal);

    float3 corrected_normal = fix_backfacing_shading_normal(wo_below_hemipshere, normal);
    float cos_theta = dot(wo_below_hemipshere, corrected_normal);
    EXPECT_FLOAT_EQ_EPS(0.0f, cos_theta, 1e-6f);
}

GTEST_TEST(Trigonometry, fix_backfacing_shading_normal_with_target_cos_theta) {
    using namespace optix;

    float target_cos_theta = 0.002f;
    float3 normal = { 0, 0, 1 };
    float3 wo_in_hemisphere = normalize(make_float3(1, 0, 1));
    float3 wo_orthogonal = { 1, 0, 0 };
    float3 wo_below_hemipshere = normalize(make_float3(1, 0, -0.1f));

    float3 uncorrected_normal = fix_backfacing_shading_normal(wo_in_hemisphere, normal, target_cos_theta);
    EXPECT_FLOAT3_EQ(normal, uncorrected_normal);

    float3 corrected_normal = fix_backfacing_shading_normal(wo_orthogonal, normal, target_cos_theta);
    float actual_cos_theta = dot(wo_orthogonal, corrected_normal);
    EXPECT_FLOAT_EQ_EPS(target_cos_theta, actual_cos_theta, 1e-5f);

    corrected_normal = fix_backfacing_shading_normal(wo_below_hemipshere, normal, target_cos_theta);
    actual_cos_theta = dot(wo_below_hemipshere, corrected_normal);
    EXPECT_FLOAT_EQ_EPS(target_cos_theta, actual_cos_theta, 1e-5f);
}

GTEST_TEST(Trigonometry, refract_overload_gives_same_result_as_optix_implementation) {
    using namespace optix;

    float3 up = make_float3(0, 0, 1);

    for (int wo_s = 0; wo_s < 16; wo_s++) {
        float3 wo = Distributions::UniformHemisphere::sample(RNG::sample02(wo_s)).direction;
        for (float ior_i_over_o : { 0.33f, 0.7f, 1.5f, 3.0f }) {
            float3 expected_direction;
            bool expected_success = optix::refract(expected_direction, wo, up, ior_i_over_o);

            { // Test vector implementation
                float3 actual_refraction;
                bool actual_success = OptiXRenderer::refract(actual_refraction, wo, ior_i_over_o);

                EXPECT_EQ(expected_success, actual_success);
                if (actual_success) // The reference implementation returns the zero vector if refraction failed, while the OptiXRenderer implementation returns an undefined result.
                    EXPECT_FLOAT3_EQ_EPS(expected_direction, actual_refraction, 1e-6f);
            }

            { // Test angle implementation
                float actual_refracted_cos_theta;
                bool actual_success = OptiXRenderer::refract(actual_refracted_cos_theta, wo.z, ior_i_over_o);

                EXPECT_EQ(expected_success, actual_success);
                if (actual_success) // The reference implementation returns the zero vector if refraction failed, while the OptiXRenderer implementation returns an undefined result.
                    EXPECT_FLOAT_EQ_EPS(expected_direction.z, actual_refracted_cos_theta, 1e-6f);
            }
        }
    }
}

} // NS OptiXRenderer

#endif // _OPTIXRENDERER_MISC_TEST_H_