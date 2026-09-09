// Test sphere lights in Bifrost.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _BIFROST_ASSETS_SHADING_LIGHTSOURCES_SPHERE_LIGHT_TEST_H_
#define _BIFROST_ASSETS_SHADING_LIGHTSOURCES_SPHERE_LIGHT_TEST_H_

#include <Assets/Shading/LightSources/LightTestUtils.h>
#include <Expects.h>

#include <Bifrost/Assets/Shading/LightSources/SphereLight.h>
#include <Bifrost/Math/Utils.h>

#include <gtest/gtest.h>

namespace Bifrost::Assets::Shading::LightSources {

GTEST_TEST(Assets_Shading_LightSources_SphereLight, power_preservation_when_radius_changes) {
    using namespace Bifrost::Math;

    const unsigned int RADIUS_COUNT = 5;
    const unsigned int MAX_SAMPLES = 1024u;

    Vector3f shading_position = Vector3f(0.0f);
    Vector3f shading_normal = Vector3f(0.0f, 1.0f, 0.0f);

    RGB expected_power = RGB(10);
    float radiuses[RADIUS_COUNT] = { 0.0f, 1.0f, 2.0f, 5.0f, 9.0f };

    float actual_power[RADIUS_COUNT];
    for (int i = 0; i < RADIUS_COUNT; ++i) {
        float radius = radiuses[i];
        SphereLight light = SphereLight(Vector3f(0.0f, 10.0f, 0.0f), radius, expected_power);
        double summed_radiance = 0.0;
        for (unsigned int i = 0u; i < MAX_SAMPLES; ++i) {
            Vector2f random_sample = RNG::sample02(i);

            LightSample sample = light.sample_radiance(shading_position, random_sample);
            summed_radiance += sample.radiance.r * (dot(shading_normal, sample.direction_to_light) / sample.PDF.value());
        }
        float radiance = summed_radiance / float(MAX_SAMPLES);

        // Map radiance arriving at intersection point to light source power by applying an inverse quadratic fall off and assume that all lights are point lights.
        float distance_to_light = magnitude(shading_position - light.get_position());
        float light_power = radiance * (4.0f * PIf * distance_to_light * distance_to_light);
        actual_power[i] = light_power;
    }

    // Test that the estimated power equals the power of the light source.
    EXPECT_FLOAT_EQ(expected_power.r, actual_power[0]);
    EXPECT_FLOAT_EQ_EPS(expected_power.r, actual_power[1], 0.0001f);
    EXPECT_FLOAT_EQ_EPS(expected_power.r, actual_power[2], 0.001f);
    EXPECT_FLOAT_EQ_EPS(expected_power.r, actual_power[3], 0.001f);
    EXPECT_FLOAT_EQ_EPS(expected_power.r, actual_power[4], 0.004f);
}

GTEST_TEST(Assets_Shading_LightSources_SphereLight, function_consistency) {
    using namespace Bifrost::Math;

    const unsigned int MAX_SAMPLES = 32u;
    const Vector3f lit_position = Vector3f(10.0f, 0.0f, 0.0f);

    Vector3f light_position = Vector3f(0.0f, 10.0f, 0.0f);
    RGB light_power = RGB(10.0f);

    for (float radius : { 0.0f, 1.0f, 13.0f }) {
        SphereLight light = { light_position, radius, light_power };
        LightTestUtils::light_consistency_test(light, lit_position, MAX_SAMPLES);
    }
}

GTEST_TEST(Assets_Shading_LightSources_SphereLight, evaluation_and_PDF_rejects_rays_that_miss) {
    using namespace Bifrost::Math;

    SphereLight light = SphereLight(Vector3f(0.0f, 10.0f, 0.0f), 2.0f, RGB(10.0f));

    Vector3f lit_position = Vector3f(0.0f, 0.0f, 0.0f);
    Vector3f hit_light_direction = normalize(Vector3f(1.0f, 10.0f, 0.0f));
    Vector3f miss_light_direction = normalize(Vector3f(3.0f, 10.0f, 0.0f));

    auto hit_light_PDF = light.pdf(lit_position, hit_light_direction);
    EXPECT_TRUE(hit_light_PDF.is_valid());
    auto hit_light_response = light.evaluate_with_PDF(lit_position, hit_light_direction);
    EXPECT_TRUE(hit_light_response.PDF.is_valid());
    EXPECT_GT(hit_light_response.radiance.r, 0.0f);

    auto miss_light_PDF = light.pdf(lit_position, miss_light_direction);
    EXPECT_FALSE(miss_light_PDF.is_valid());
    auto miss_light_response = light.evaluate_with_PDF(lit_position, miss_light_direction);
    EXPECT_FALSE(miss_light_response.PDF.is_valid());
    EXPECT_RGB_EQ(miss_light_response.radiance, RGB::black());
}

} // NS Bifrost::Assets::Shading::LightSources

#endif // _BIFROST_ASSETS_SHADING_LIGHTSOURCES_SPHERE_LIGHT_TEST_H_