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