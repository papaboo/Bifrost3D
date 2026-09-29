// Test Bifrost's LTC disk light approximation.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _BIFROST_ASSETS_SHADING_LIGHTSOURCES_DISK_LIGHT_LTC_TEST_H_
#define _BIFROST_ASSETS_SHADING_LIGHTSOURCES_DISK_LIGHT_LTC_TEST_H_

#include <Assets/Shading/BSDFs/LambertTest.h>
#include <Assets/Shading/BSDFTestUtils.h>
#include <Expects.h>

#include <Bifrost/Assets/Shading/LightSources/DiskLight.h>
#include <Bifrost/Assets/Shading/LinearlyTransformedCosines.h>

#include <gtest/gtest.h>

namespace Bifrost::Assets::Shading::LightSources {

GTEST_TEST(Assets_Shading_LightSources_DiskLight_LTC, single_sided_disk_light_only_shades_in_front) {
    using namespace Bifrost::Math;

    // Define the surface plane to illuminate.
    // The surface passes through origo and the normal is along positive z.
    Vector3f surface_point = { 0, 0, 0 };
    Vector3f surface_normal = { 0, 0, 1 };
    Vector3f wo = { 0, 0, 1 };

    // Define triangle light above surface plane at (0, 0, 0) with normal pointing upwards.
    float distance_to_surface = 1.0f;
    Vector3f light_normal = { 0, 0, 1 };
    Disk light_surface = Disk(Vector3f(0, 0, distance_to_surface), 1, light_normal);

    // Two sided disk light illuminates on both sides.
    auto two_sided_light = DiskLight(light_surface, RGB::white(), true);
    RGB radiance = two_sided_light.evaluate_radiance(wo, surface_point, surface_normal);
    EXPECT_RGB_GT(radiance, 0.0f);

    // One sided disk light should not illuminate surface behind.
    auto one_sided_light = DiskLight(light_surface, RGB::white(), false);
    radiance = one_sided_light.evaluate_radiance(wo, surface_point, surface_normal);
    EXPECT_RGB_EQ(radiance, RGB::black());
}

GTEST_TEST(Assets_Shading_LightSources_DiskLight_LTC, light_behind_surface_does_not_illuminate) {
    using namespace Bifrost::Math;

    Vector3f surface_point = { 0, 0, 0 };
    Vector3f surface_normal = { 0, 0, 1 };

    // Define triangle light below surface plane at (0, 0, 0) with normal pointing upwards.
    float distance_to_surface = -1.0f;
    Vector3f light_normal = { 0, 0, 1 };
    bool is_two_sided = true; // Ensure that the light always casts light at the surface.
    Disk light_surface = Disk(Vector3f(0, 0, distance_to_surface), 1, light_normal);
    auto light = DiskLight(light_surface, RGB::white(), is_two_sided);

    for (float cos_theta_o : { 0.2f, 0.6f, 1.0f }) {
        Vector3f wo = BSDFTestUtils::w_from_cos_theta(cos_theta_o);

        RGB radiance = light.evaluate_radiance(wo, surface_point, surface_normal);
        EXPECT_RGB_EQ(radiance, RGB::black());
    }
}

GTEST_TEST(Assets_Shading_LightSources_DiskLight_LTC, LTC_evaluation_is_shading_space_rotation_agnostic) {
    using namespace Bifrost::Math;

    Vector3f surface_point = { 0, 0, 0 };
    Vector3f surface_normal = { 0, 0, 1 };
    float surface_roughness = 0.5f;

    Vector3f wo0 = normalize(Vector3f(1, 1, 1));
    float cos_theta_o = wo0.z;
    Vector3f wo1 = { sqrt(1 - pow2(cos_theta_o)), 0, cos_theta_o };
    Vector3f wo2 = { wo1.y, -wo1.x, cos_theta_o };

    auto ltc_bsdf = Bifrost::Assets::Shading::LTC::GGX_reflection_LTC_coefficients(cos_theta_o, surface_roughness);

    Vector3f wos[3] = { wo0, wo1, wo2 };
    RGB radiances[3] = { RGB::black(), RGB::black(), RGB::black() };
    for (int i : {0, 1, 2}) {
        Vector3f wo = wos[i];

        // Create light at reflected view direction pointing towards the surface,
        // so the light has the same location relative to the view direction.
        Vector3f wi = reflect(-wo, surface_normal);
        auto light = DiskLight(Disk(wi, 1, -wi), RGB::white(), true);

        radiances[i] = light.evaluate(ltc_bsdf, wo, surface_point, surface_normal);
    }

    EXPECT_RGB_GT(radiances[0], 0.0f);
    EXPECT_RGB_EQ_EPS(radiances[0], radiances[1], 1e-5f);
    EXPECT_RGB_EQ_EPS(radiances[0], radiances[2], 1e-5f);
}

GTEST_TEST(Assets_Shading_LightSources_DiskLight_LTC, LTC_evaluation_accounts_for_geometric_term) {
    using namespace Bifrost::Math;

    Vector3f wo = Vector3f(0, 0, 1);
    float distance = 100;
    // Shaded position far from the light source.
    Vector3f shaded_position = Vector3f(0, 0, -distance);
    Vector3f shaded_normal = Vector3f(0, 0, 1);

    // Use a small triangle so the radiance contribution is dominated by the geometric term.
    float light_radius = 0.1f;
    Disk light_surface = Disk(Vector3f::zero(), light_radius, Vector3f(0, 0, 1));
    auto light = DiskLight(light_surface, RGB::white() * 1e4f, true);

    // Expected radiance at 'distance' from light source and with no reduction in light area from rotation.
    RGB expected_radiance = light.evaluate_radiance(wo, shaded_position, shaded_normal);

    for (int y = -1; y <= 1; ++y)
        for (int x = -1; x <= 1; ++x) {
            Vector3f local_shaded_position = shaded_position + Vector3f(x, y, 0) * distance;
            Vector3f direction_to_light = normalize(light_surface.center - local_shaded_position);
            // Project shaded position to keep same distance to light as reference point.
            local_shaded_position = light_surface.center - direction_to_light * distance;
            
            // Test that LTC lights have a uniform light distribution, meaning that they emit equal amounts of light in each direction
            // and only the projected area of the light has an impact on a shaded point facing towards the light.
            float light_area_reduction = abs(dot(direction_to_light, light.get_normal()));
            {
                Vector3f shaded_normal_facing_light = direction_to_light;
                RGB radiance = light.evaluate_radiance(wo, local_shaded_position, shaded_normal_facing_light);
                RGB radiance_corrected_for_area_reduction = radiance / (light_area_reduction);

                EXPECT_RGB_EQ_PCT(expected_radiance, radiance_corrected_for_area_reduction, 0.001f) << "[x: " << x << ", y: " << y << "]: light area reduction: " << light_area_reduction;
            }

            // Test that LTC lights respect the geometric term from tilting the surface,
            // such that the light is distributed over a larger surface and less light arrives at a single point.
            {
                RGB radiance = light.evaluate_radiance(wo, local_shaded_position, shaded_normal);

                // Divide out the geometric term and the light area reduction
                float geometric_term = abs(dot(direction_to_light, shaded_normal));
                RGB corrected_radiance = radiance / (geometric_term * light_area_reduction);

                EXPECT_RGB_EQ_PCT(expected_radiance, corrected_radiance, 0.001f) << "[x: " << x << ", y: " << y << "]: geometric term: " << geometric_term;
            }
        }
}

GTEST_TEST(Assets_Shading_LightSources_DiskLight_LTC, LTC_integration_yields_same_result_as_monte_carlo) {
    using namespace Bifrost::Math;

    int sample_count = 1024;
    int max_wo_sample_count = 4;

    auto lambert_bsdf = BSDFs::LambertWrapper();
    auto ltc_lambert_bsdf = Bifrost::Assets::Shading::LTC::lambert_LTC_coefficients();
    Vector3f wo = normalize(Vector3f(1, -1, 1)); // wo just needs to lie in the positive hemisphere for lambert bsdf, the direction itself is irrelevant.

    Vector3f light_center = Vector3f(0, 0, 0);

    for (float size : { 0.1f, 1.0f, 3.0f, 5.0f }) {
        for (int x : { -1, 0, 1, 2, 5 }) {
            Vector3f shaded_position = Vector3f(x, 0, -1) * 10;
            Vector3f direction_to_light = normalize(light_center - shaded_position);

            for (Vector3f shaded_normal : { Vector3f(0, 0, 1), direction_to_light }) {
                for (Vector3f light_normal : { Vector3f(0, 0, -1), -direction_to_light }) {
                    Disk surface = Disk(light_center, size, light_normal);
                    DiskLight light = DiskLight(surface, RGB(10), true);

                    RGB reflectance_area_light = BSDFTestUtils::integrate_light_over_surface(wo, shaded_position, shaded_normal, lambert_bsdf, light, sample_count);
                    RGB reflectance_ltc_light = light.evaluate(ltc_lambert_bsdf, wo, shaded_position, shaded_normal);

                    // Error increased wrt the solid angle. This should be fixed by using the exact diffuse sphere integral.
                    float error_percentage = size * size * 0.01f;
                    EXPECT_RGB_EQ_PCT(reflectance_ltc_light, reflectance_area_light, error_percentage) << "ratio: " << reflectance_area_light.r / reflectance_ltc_light.r;
                }
            }
        }
    }
}

} // NS Bifrost::Assets::Shading::LightSources

#endif // _BIFROST_ASSETS_SHADING_LIGHTSOURCES_DISK_LIGHT_LTC_TEST_H_