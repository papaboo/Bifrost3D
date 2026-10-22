// Test general properties of Bifrost's area lights.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _BIFROST_ASSETS_SHADING_LIGHTSOURCES_AREA_LIGHTS_TEST_H_
#define _BIFROST_ASSETS_SHADING_LIGHTSOURCES_AREA_LIGHTS_TEST_H_

#include <Assets/Shading/LightSources/LightTestUtils.h>
#include <Assets/Shading/BSDFTestUtils.h>
#include <Expects.h>

#include <Bifrost/Assets/Shading/LightSources/DiskLight.h>
#include <Bifrost/Assets/Shading/LightSources/SphereLight.h>
#include <Bifrost/Assets/Shading/LightSources/TriangleLight.h>

#include <gtest/gtest.h>

namespace Bifrost::Assets::Shading::LightSources {

template <typename LightSource>
Math::RGB integrate_radiance(Math::Vector3f shaded_position, Math::Vector3f shaded_normal, LightSource light, int sample_count = 1024) {
    auto summed_radiance = Math::RGB(0.0f);
    for (unsigned int i = 0u; i < sample_count; ++i) {
        auto random_sample = BSDFTestUtils::bsdf_rng_sample2f(i);

        LightSample sample = light.sample_radiance(shaded_position, random_sample);
        summed_radiance += sample.radiance * (dot(shaded_normal, sample.direction_to_light) / sample.PDF.value());
    }
    return summed_radiance / sample_count;
}

GTEST_TEST(Assets_Shading_LightSources_AreaLights, closed_form_diffuse_sphere_cap_integral_yields_same_result_as_monte_carlo_estimation) {
    using namespace Bifrost::Math;

    int sample_count = 1024;
    Vector3f surface_position = Vector3f(0, 0, 0);

    // We only care about the case where the surface position is outside the light, so we set the light radius to something small enough
    // that no light position is close enough to place the surface position inside the light.
    float light_radius = 0.01f;
    for (int light_z = -1; light_z <= 2; light_z++)
        for (int light_x = -2; light_x <= 2; light_x++) {
            // Use oriented disk as representation for sphere cap
            Vector3f light_position = { float(light_x), 0, float(light_z) };
            Vector3f light_direction = light_position - surface_position;
            float light_distance = magnitude(light_direction);
            light_direction = normalize(light_direction);
            
            // Skip cases where the surface position is inside the light
            if (light_distance <= light_radius)
                continue;

            float light_projected_solid_angle = Math::PI<float>() * pow2(light_radius / light_distance);

            // Sample sphere cap and integrate with respect to the diffuse surface
            float summed_contribution = 0.0f;
            for (unsigned int i = 0u; i < sample_count; ++i) {
                auto random_sample = BSDFTestUtils::bsdf_rng_sample2f(i);
                auto disk_sample = Distributions::Disk::sample_solid_angle(surface_position, light_position, light_radius, light_direction, random_sample);
                bool is_occluded = disk_sample.direction.z < 0.0f;
                if (!is_occluded) {
                    constexpr float diffuse_scale = RECIP_PIf;
                    summed_contribution += diffuse_scale * disk_sample.direction.z / disk_sample.PDF;
                }
            }
            float expected_contribution = summed_contribution / sample_count;

            // This seems to be the adjustment to the expected contribution so it fits the the diffuse sphere cap.
            // as the closest form solution seems to be outputting diffuse weighted integral over the hemisphere divided by projected solid angle.
            // (or in the code it is divide by PI * projected solid angle over PI.)
            // But for some reason it isn't enough.
            expected_contribution /= light_projected_solid_angle;
            expected_contribution /= PIf;
            float actual_contribution = LtcAreaLight::integrate_diffuse_sphere_cap(light_direction.z, light_projected_solid_angle * RECIP_PIf);

            EXPECT_FLOAT_EQ_EPS(expected_contribution, actual_contribution, 0.001f) << "light position: " << light_position.to_string() << ", ratio: " << expected_contribution / actual_contribution;
        }
}

GTEST_TEST(Assets_Shading_LightSources_AreaLights, tiny_flat_area_lights_with_same_power_has_similar_contribution) {
    using namespace Bifrost::Math;

    Vector3f shaded_normal = Vector3f(0, 0, 1);
    Vector3f shaded_position = Vector3f(0, 0, -1000);

    RGB light_power = RGB(10);
    bool is_two_sided = true;

    Disk disk = Disk(Vector3f(0, 0, 0), 1, Vector3f(0, 0, -1));
    DiskLight disk_light = DiskLight(disk, light_power, is_two_sided);
    RGB disk_light_radiance = integrate_radiance(shaded_position, shaded_normal, disk_light);

    Disk smaller_disk = Disk(Vector3f(0, 0, 0), 0.1f, Vector3f(0, 0, -1));
    DiskLight smaller_disk_light = DiskLight(smaller_disk, light_power, is_two_sided);
    RGB smaller_disk_light_radiance = integrate_radiance(shaded_position, shaded_normal, smaller_disk_light);

    Trianglef triangle = Trianglef(Vector3f(-1, -1, 0), Vector3f(1, 0, 0), Vector3f(0, 1, 0));
    TriangleLight triangle_light = TriangleLight(triangle, light_power, is_two_sided);
    RGB triangle_light_radiance = integrate_radiance(shaded_position, shaded_normal, triangle_light);

    EXPECT_RGB_EQ_PCT(disk_light_radiance, smaller_disk_light_radiance, 0.001f);
    EXPECT_RGB_EQ_PCT(disk_light_radiance, triangle_light_radiance, 0.001f);
}

GTEST_TEST(Assets_Shading_LightSources_AreaLights, disk_light_can_approximate_sphere_light) {
    using namespace Bifrost::Math;

    Vector3f shaded_normal = Vector3f(0, 0, 1);
    Vector3f shaded_position = Vector3f(0, 0, 0);

    RGB light_power = RGB(100);
    bool is_two_sided = true;

    // The approximation breaks down when the sphere is close to the surface, i.e. the subtended solid angle is large.
    // Perhaps moving the disk closer, but maintaining the same solid angle would improve the approximation.
    for (float light_distance : { 20, 50, 100 })
        for (float light_radius : { 1, 2, 4 }) {

            Vector3f light_position = Vector3f(0, 0, light_distance);

            SphereLight sphere_light = SphereLight(light_position, light_radius, light_power);
            RGB sphere_light_radiance = integrate_radiance(shaded_position, shaded_normal, sphere_light);

            DiskLight disk_light = DiskLight::approximate_sphere_light(sphere_light, shaded_position);
            RGB disk_light_radiance = integrate_radiance(shaded_position, shaded_normal, disk_light);

            EXPECT_RGB_EQ_PCT(sphere_light_radiance, disk_light_radiance, 0.05f);
        }
}

} // NS Bifrost::Assets::Shading::LightSources

#endif // _BIFROST_ASSETS_SHADING_LIGHTSOURCES_AREA_LIGHTS_TEST_H_