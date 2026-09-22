// Test disk lights in Bifrost.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _BIFROST_ASSETS_SHADING_LIGHTSOURCES_DISK_LIGHT_TEST_H_
#define _BIFROST_ASSETS_SHADING_LIGHTSOURCES_DISK_LIGHT_TEST_H_

#include <Assets/Shading/LightSources/LightTestUtils.h>
#include <Expects.h>

#include <Bifrost/Assets/Shading/LightSources/DiskLight.h>
#include <Bifrost/Math/Utils.h>

#include <gtest/gtest.h>

namespace Bifrost::Assets::Shading::LightSources {

GTEST_TEST(Assets_Shading_LightSources_DiskLight, two_sided_lights_shine_with_half_radiance_on_lit_surface) {
    using namespace Bifrost::Math;

    Disk surface = Disk(Vector3f::zero(), 1, Vector3f(0, 0, -1));
    RGB light_power = RGB(10.0f);
    DiskLight one_sided_light = DiskLight(surface, light_power, false);
    DiskLight two_sided_light = DiskLight(surface, light_power, true);

    Vector3f lit_position = Vector3f(0.5f, 0.5f, -2.0f);
    Vector3f direction_to_light = Vector3f(0, 0, 1);

    auto one_sided_light_response = one_sided_light.evaluate_with_PDF(lit_position, direction_to_light);
    EXPECT_TRUE(one_sided_light_response.PDF.is_valid());
    EXPECT_GT(one_sided_light_response.radiance.r, 0.0f);

    auto two_sided_light_response = two_sided_light.evaluate_with_PDF(lit_position, direction_to_light);
    EXPECT_TRUE(two_sided_light_response.PDF.is_valid());
    EXPECT_GT(two_sided_light_response.radiance.r, 0.0f);

    EXPECT_RGB_EQ(two_sided_light_response.radiance * 2, one_sided_light_response.radiance);
}

GTEST_TEST(Assets_Shading_LightSources_DiskLight, shaded_position_in_light_plane_is_not_lit) {
    using namespace Bifrost::Math;

    Disk surface = Disk(Vector3f(0, 0, 0), 1, Vector3f(0, 0, -1));
    DiskLight area_light = DiskLight(surface, RGB(10), false);

    DiskLight delta_light = DiskLight(surface.center, surface.normal, RGB(10), false);

    for (auto& light : { area_light, delta_light }) {
        // Position in the same plane as the light source is not lit
        Vector3f position_in_plane = surface.center; // Picked as the worst case for the delta light, as the position is the same as the delta light's.
        LightSample light_sample = light.sample_radiance(position_in_plane, { 0.5f, 0.5f });
        EXPECT_FALSE(light_sample.PDF.is_valid());
        EXPECT_RGB_EQ(RGB::black(), light_sample.radiance);

        auto light_response = light.evaluate_with_PDF(position_in_plane, light_sample.direction_to_light);
        EXPECT_FALSE(light_response.PDF.is_valid());
        EXPECT_RGB_EQ(RGB::black(), light_response.radiance);
    }
}

GTEST_TEST(Assets_Shading_LightSources_DiskLight, one_sided_light_only_lights_in_front) {
    using namespace Bifrost::Math;

    Disk surface = Disk(Vector3f(0, 0, 0), 1, Vector3f(0, 0, 1));
    DiskLight area_light = DiskLight(surface, RGB(10), false);

    DiskLight delta_light = DiskLight(surface.center, surface.normal, RGB(10), false);

    for (auto& light : { area_light, delta_light }) {

        { // Position in front of light source can be lit
            Vector3f position_in_front = Vector3f(0, 0, 2);
            Vector3f direction_to_light = normalize(surface.center - position_in_front);

            LightSample sample = light.sample_radiance(position_in_front, { 0.5f, 0.5f });
            EXPECT_TRUE(sample.PDF.is_valid());
            EXPECT_GT(sample.radiance.r, 0.0f);

            LightResponse response = light.evaluate_with_PDF(position_in_front, direction_to_light);
            if (light.is_delta_light()) {
                // Delta lights can only be sampled and not evaluated.
                EXPECT_FALSE(response.PDF.is_valid());
                EXPECT_EQ(response.radiance.r, 0.0f);
            } else {
                EXPECT_TRUE(response.PDF.is_valid());
                EXPECT_GT(response.radiance.r, 0.0f);
            }
        }

        { // Position behind light source can't be lit
            Vector3f position_behind = Vector3f(0, 0, -2);
            Vector3f direction_to_light = normalize(surface.center - position_behind);

            LightSample sample = light.sample_radiance(position_behind, { 0.5f, 0.5f });
            EXPECT_FALSE(sample.PDF.is_valid());
            EXPECT_EQ(sample.radiance.r, 0.0f);

            LightResponse response = light.evaluate_with_PDF(position_behind, direction_to_light);
            EXPECT_FALSE(response.PDF.is_valid());
            EXPECT_EQ(response.radiance.r, 0.0f);
        }
    }
}

GTEST_TEST(Assets_Shading_LightSources_DiskLight, radiance_follows_the_inverse_square_law) {
    using namespace Bifrost::Math;

    const unsigned int DISTANCE_COUNT = 4;
    const unsigned int MAX_SAMPLES = 1024u;

    Vector3f shaded_position = Vector3f(0.0f);
    Vector3f shaded_normal = Vector3f(0.0f, 0.0f, 1.0f);

    RGB light_power = RGB(10);
    float distance[DISTANCE_COUNT] = { 1.0f, 2.0f, 4.0f, 16.0f };

    RGB radiance_at_distance[DISTANCE_COUNT];
    for (int i = 0; i < DISTANCE_COUNT; ++i) {
        // Linking the size of the light to the distance means that the light will have the same relative size and subtended solid angle
        // from the shaded position at origo.
        float radius = distance[i];
        Disk surface = Disk(Vector3f(0, 0, distance[i]), radius, Vector3f(0, 0, -1));
        DiskLight light = DiskLight(surface, light_power);
        EXPECT_FLOAT_EQ(dot(shaded_normal, light.get_normal()), -1.0f) << "Light should point towards shaded position.";

        RGB summed_radiance = RGB(0.0f);
        for (unsigned int i = 0u; i < MAX_SAMPLES; ++i) {
            Vector2f random_sample = RNG::sample02(i);

            LightSample sample = light.sample_radiance(shaded_position, random_sample);
            summed_radiance += sample.radiance * (dot(shaded_normal, sample.direction_to_light) / sample.PDF.value());
        }
        RGB radiance = summed_radiance / float(MAX_SAMPLES);

        // Map radiance arriving at intersection point to light source power by applying an inverse quadratic fall off and assume that all lights are point lights.
        radiance_at_distance[i] = radiance;
    }

    RGB radiance_at_unit_distance = radiance_at_distance[0] * pow2(distance[0]);
    for (int i = 1; i < DISTANCE_COUNT; ++i) {
        RGB radiance_at_target_distance = radiance_at_unit_distance / pow2(distance[i]);
        EXPECT_RGB_EQ(radiance_at_target_distance, radiance_at_distance[i]);
    }
}

GTEST_TEST(Assets_Shading_LightSources_DiskLight, light_function_consistency) {
    using namespace Bifrost::Math;

    Vector3f lit_position = Vector3f(0.0f, 0.0f, 2.0f);
    RGB light_power = RGB(1, 0.5f, 0.0f);

    for (float radius : { 0.0f, 1.0f, 13.0f }) {
        const unsigned int sample_count = radius == 0.0f ? 1 : 32;
        for (bool is_two_sided : { true, false })
            for (Vector3f normal : { Vector3f(0, 0, 1), Vector3f(0, 0, -1) }) {
                Disk surface = Disk(Vector3f::zero(), radius, normal);
                DiskLight light = DiskLight(surface, light_power, is_two_sided);
                LightTestUtils::light_consistency_test(light, lit_position, sample_count);
            }
    }
}

GTEST_TEST(Assets_Shading_LightSources_DiskLight, evaluation_and_PDF_rejects_rays_that_miss) {
    using namespace Bifrost::Math;

    Disk surface = Disk(Vector3f::zero(), 1, Vector3f(0, 0, 1));
    DiskLight light = DiskLight(surface, RGB(1), true);

    Vector3f lit_position = Vector3f(0, 0, 1);
    Vector3f hit_light_direction = Vector3f(0.0f, 0.0f, -1.0f);
    Vector3f miss_light_direction = normalize(Vector3f(-2.0f, 0.0f, -1.0f));

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

GTEST_TEST(Assets_Shading_LightSources_DiskLight, delta_light_contribution_similar_to_tiny_area_light) {
    using namespace Bifrost::Math;

    int MAX_SAMPLES = 1024;

    Disk surface = Disk(Vector3f(0, 0, 0), 1, Vector3f(0, 0, -1));
    DiskLight area_light = DiskLight(surface, RGB(10), false);

    DiskLight delta_light = DiskLight(surface.center, surface.normal, RGB(10), false);

    Vector3f shaded_normal = Vector3f(0, 0, 1);
    EXPECT_FLOAT_EQ(dot(shaded_normal, area_light.get_normal()), -1.0f) << "Light should point towards shaded position.";

    for (int y = -1; y <= 1; ++y)
        for (int x = -1; x <= 1; ++x) {
            // Shaded position far from the light source.
            Vector3f shaded_position = Vector3f(x, y, -1) * 1000;

            RGB summed_radiance_area_light = RGB(0.0f);
            for (unsigned int i = 0u; i < MAX_SAMPLES; ++i) {
                Vector2f random_sample = RNG::sample02(i);

                LightSample sample = area_light.sample_radiance(shaded_position, random_sample);
                summed_radiance_area_light += sample.radiance * (dot(shaded_normal, sample.direction_to_light) / sample.PDF.value());
            }
            RGB radiance_area_light = summed_radiance_area_light / float(MAX_SAMPLES);

            LightSample delta_sample = delta_light.sample_radiance(shaded_position, { 0.5f, 0.5f });
            RGB radiance_delta_light = delta_sample.radiance * (dot(shaded_normal, delta_sample.direction_to_light) / delta_sample.PDF.value());

            EXPECT_RGB_EQ_PCT(radiance_delta_light, radiance_area_light, 0.001f);
        }
}

} // NS Bifrost::Assets::Shading::LightSources

#endif // _BIFROST_ASSETS_SHADING_LIGHTSOURCES_DISK_LIGHT_TEST_H_