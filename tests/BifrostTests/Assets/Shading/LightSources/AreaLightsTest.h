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

    int sample_count = 2048;
    for (float average_cos_theta_i : { -1.0f, -0.5f, 0.1f, 0.0f, 0.1f, 0.2f, 0.5f, 1.0f }) {
        Vector3f light_direction = BSDFTestUtils::w_from_cos_theta(average_cos_theta_i);
        Matrix3x3f sample_to_light_direction = compute_TBN(light_direction);

        for (float light_cos_theta : { 0.0f, 0.1f, 0.5f, 0.9f }) {
            float light_sin_theta_squared = 1.0f - pow2(light_cos_theta);
            float light_projected_solid_angle = Math::PI<float>() * light_sin_theta_squared;

            // Sample sphere cap and integrate with respect to a diffuse surface
            float summed_contribution = 0.0f;
            for (unsigned int i = 0u; i < sample_count; ++i) {
                auto random_sample = BSDFTestUtils::bsdf_rng_sample2f(i);
                auto sphere_cap_sample = Distributions::Cone::sample(light_cos_theta, random_sample);

                // Transform so center of samples point along average_cos_theta_i
                sphere_cap_sample.direction = sample_to_light_direction * sphere_cap_sample.direction;

                bool is_occluded = sphere_cap_sample.direction.z < 0.0f;
                if (!is_occluded) {
                    constexpr float diffuse_scale = RECIP_PIf;
                    summed_contribution += diffuse_scale * sphere_cap_sample.direction.z / sphere_cap_sample.PDF;
                }
            }
            float expected_contribution = summed_contribution / sample_count;

            float actual_contribution = LtcAreaLight::integrate_diffuse_sphere_cap(average_cos_theta_i, light_projected_solid_angle * RECIP_PIf);

            EXPECT_FLOAT_EQ_EPS(expected_contribution, actual_contribution, 0.001f);
        }
    }
}

GTEST_TEST(Assets_Shading_LightSources_AreaLights, closed_form_diffuse_sphere_cap_integral_handles_edge_case_where_projected_solid_angle_is_zero) {
    for (float average_direction_z : {-1.0f, -0.5f, -0.25f, -0.0f, 0.0f, 0.25f, 0.5f, 1.0f}) {
        float tiny_form_factor = 1e-6f;
        float degenerate_form_factor = 0.0f;
        float tiny_result = LtcAreaLight::integrate_diffuse_sphere_cap(average_direction_z, tiny_form_factor);
        float degenerate_result = LtcAreaLight::integrate_diffuse_sphere_cap(average_direction_z, degenerate_form_factor);
        EXPECT_FLOAT_EQ_EPS(tiny_result, degenerate_result, 1e-5f) << " average direction: " << average_direction_z;
    }
}

GTEST_TEST(Assets_Shading_LightSources_AreaLights, sphere_light_visibility_handles_edge_case_where_projected_solid_angle_is_zero) {
    for (float average_direction_z : {-1.0f, -0.5f, -0.25f, -0.1f, 0.1f, 0.25f, 0.5f, 1.0f}) {
        float tiny_form_factor = 1e-6f;
        float degenerate_form_factor = 0.0f;
        float tiny_result = LtcAreaLight::diffuse_sphere_light_visibility(average_direction_z, tiny_form_factor);
        float degenerate_result = LtcAreaLight::diffuse_sphere_light_visibility(average_direction_z, degenerate_form_factor);
        EXPECT_FLOAT_EQ_EPS(tiny_result, degenerate_result, 1e-5f) << " average direction: " << average_direction_z;
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