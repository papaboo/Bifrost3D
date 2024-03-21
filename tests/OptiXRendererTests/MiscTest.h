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

#include <OptiXRenderer/MonteCarlo.h>
#include <OptiXRenderer/Types.h>

#include <gtest/gtest.h>

#include <limits.h>

namespace OptiXRenderer {

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
                optix::float3 optix_decoded_normal = optix::normalize(optix_encoded_normal.decode_unnormalized());

                EXPECT_FLOAT_EQ(bifrost_decoded_normal.x, optix_decoded_normal.x);
                EXPECT_FLOAT_EQ(bifrost_decoded_normal.y, optix_decoded_normal.y);
                EXPECT_FLOAT_EQ(bifrost_decoded_normal.z, optix_decoded_normal.z);
            }
}

GTEST_TEST(MonteCarlo, Balance_heuristic_invariants) {
    // Sanity checks.
    EXPECT_FLOAT_EQ(0.5f, MonteCarlo::balance_heuristic(1.0f, 1.0f));
    EXPECT_FLOAT_EQ(0.25f, MonteCarlo::balance_heuristic(1.0f, 3.0f));

    // The balance heuristic should return 1 if the second pdf is NAN, as then the first sample trivially wins.
    EXPECT_EQ(1.0f, MonteCarlo::balance_heuristic(1.0f, NAN));

    float almost_inf = FLT_MAX;
    EXPECT_TRUE(isinf(almost_inf + almost_inf));

    // The balance heuristic should handle values close to infinity.
    EXPECT_FLOAT_EQ(1.0f / almost_inf, MonteCarlo::balance_heuristic(1.0f, almost_inf));
    EXPECT_FLOAT_EQ(1.0f, MonteCarlo::balance_heuristic(almost_inf, 1.0f));
    EXPECT_FLOAT_EQ(0.0f, MonteCarlo::balance_heuristic(0.5f * almost_inf, almost_inf));
    EXPECT_FLOAT_EQ(1.0f, MonteCarlo::balance_heuristic(almost_inf, 0.5f * almost_inf));

    // The balance heuristic should handle infinity.
    EXPECT_FLOAT_EQ(0.0f, MonteCarlo::balance_heuristic(1.0f, INFINITY));
    EXPECT_FLOAT_EQ(1.0f, MonteCarlo::balance_heuristic(INFINITY, 1.0f));

    // Zero should be a valid first parameter and always return zero.
    EXPECT_FLOAT_EQ(0.0f, MonteCarlo::balance_heuristic(0.0f, 0.0f));
    EXPECT_FLOAT_EQ(0.0f, MonteCarlo::balance_heuristic(0.0f, 1.0f));
    EXPECT_FLOAT_EQ(0.0f, MonteCarlo::balance_heuristic(0.0f, almost_inf));
}

GTEST_TEST(MonteCarlo, Power_heuristic_invariants) {
    // Sanity checks.
    EXPECT_FLOAT_EQ(0.5f, MonteCarlo::power_heuristic(1.0f, 1.0f));
    EXPECT_FLOAT_EQ(0.1f, MonteCarlo::power_heuristic(1.0f, 3.0f));

    // The power heuristic should return 1 if the second pdf is NAN, as then the first sample trivially wins.
    EXPECT_EQ(1.0f, MonteCarlo::power_heuristic(1.0f, NAN));

    float almost_inf = FLT_MAX;
    EXPECT_TRUE(isinf(almost_inf * almost_inf));

    // The power heuristic should handle values that squared become infinity.
    EXPECT_FLOAT_EQ(0.0f, MonteCarlo::power_heuristic(1.0f, almost_inf));
    EXPECT_FLOAT_EQ(1.0f, MonteCarlo::power_heuristic(almost_inf, 1.0f));

    // Zero should be a valid first parameter and always return zero.
    EXPECT_FLOAT_EQ(0.0f, MonteCarlo::power_heuristic(0.0f, 0.0f));
    EXPECT_FLOAT_EQ(0.0f, MonteCarlo::power_heuristic(0.0f, 1.0f));
    EXPECT_FLOAT_EQ(0.0f, MonteCarlo::power_heuristic(0.0f, almost_inf));

    // Hacking the power heuristic by giving it pdf's that'll force the divisor to become infinite.
    EXPECT_FLOAT_EQ(0.0f, MonteCarlo::power_heuristic(0.9f * sqrt(almost_inf), sqrt(almost_inf)));
    EXPECT_FLOAT_EQ(1.0f, MonteCarlo::power_heuristic(sqrt(almost_inf), 0.9f * sqrt(almost_inf)));
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

    // Index of refraction and extinction coefficient for mediums at wavelengths 630nm (red), 532nm (green) and 465nm (blue)
    const float3 air_ior = { 1.0f, 1.0f, 1.0f };
    const float3 gold_ior = { 0.1986f, 0.54463f, 1.2515f };
    const float3 gold_extinction = { 3.228f, 2.1406f, 1.7517f };
    const float3 titanium_ior = { 2.6979f, 2.4793f, 2.3050f };
    const float3 titanium_extinction = { 3.7571f, 3.3511f, 3.0820f };

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
    EXPECT_FLOAT_EQ(0.0f, cos_theta);
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

} // NS OptiXRenderer

#endif // _OPTIXRENDERER_MISC_TEST_H_