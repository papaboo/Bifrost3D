// Test spot lights in OptiXRenderer.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _OPTIXRENDERER_LIGHT_SOURCES_SPOT_LIGHT_TEST_H_
#define _OPTIXRENDERER_LIGHT_SOURCES_SPOT_LIGHT_TEST_H_

#include <Utils.h>

#include <Bifrost/Math/RNG.h>
#include <Bifrost/Math/Statistics.h>
#include <Bifrost/Math/Utils.h>

#include <OptiXRenderer/Shading/LightSources/SpotLightImpl.h>

#include <gtest/gtest.h>

namespace OptiXRenderer {

GTEST_TEST(SpotLight, consistent_PDF_and_radiance) {
    using namespace optix;

    const unsigned int MAX_POSITION_SAMPLES = 16u;
    const unsigned int MAX_LIGHT_SAMPLES = 16u;

    SpotLight light;
    light.position = make_float3(0.0f, 10.0f, 0.0f);
    light.direction = make_float3(0.0f, -1.0f, 0.0f);
    light.radius = 1.0f;
    light.power = make_float3(10.0f);
    light.cos_angle = 0.5f;

    const float position_variation = 2.0f;
    const float3 center_position = light.position + position_variation * light.direction;

    for (unsigned int p = 0; p < MAX_POSITION_SAMPLES; ++p) {
        float3 position = center_position + position_variation * Distributions::Cosine::sample(RNG::sample02(p)).direction;
        for (float radius : { 1.0f, 4.0f, 13.0f }) {
            light.radius = radius;
            for (float cos_angle : { 0.1f, 0.5f, 0.9f }) {
                light.cos_angle = cos_angle;
                for (unsigned int i = 0u; i < MAX_LIGHT_SAMPLES; ++i) {
                    LightSample sample = LightSources::sample_radiance(light, position, RNG::sample02(i));
                    if (is_PDF_valid(sample.PDF)) {
                        float PDF = LightSources::PDF(light, position, sample.direction_to_light);
                        EXPECT_FLOAT_EQ_EPS(sample.PDF, PDF, 0.0001f);

                        float radiance = LightSources::evaluate(light, position, sample.direction_to_light * sample.distance).x;
                        EXPECT_FLOAT_EQ_EPS(sample.radiance.x, radiance, 0.0001f);
                    }
                }
            }
        }
    }
}

GTEST_TEST(SpotLight, pdf_rejects_rays_that_miss) {
    using namespace optix;

    SpotLight light;
    light.position = make_float3(0.0f, 10.0f, 0.0f);
    light.direction = make_float3(0.0f, -1.0f, 0.0f);
    light.radius = 2.0f;
    light.power = make_float3(10.0f);
    light.cos_angle = 0.5f;

    const float3 lit_position = make_float3(0.0f, 0.0f, 0.0f);
    const float3 hit_light_direction = normalize(make_float3(1.0f, 10.0f, 0.0f));
    const float3 miss_light_direction = normalize(make_float3(3.0f, 10.0f, 0.0f));

    float hit_light_PDF = LightSources::PDF(light, lit_position, hit_light_direction);
    EXPECT_GT(hit_light_PDF, 0.0f);

    float miss_light_PDF = LightSources::PDF(light, lit_position, miss_light_direction);
    EXPECT_FLOAT_EQ(miss_light_PDF, 0.0f);

    // Flip cone light so lit position is behind
    light.direction = -light.direction;
    EXPECT_FLOAT_EQ(LightSources::PDF(light, lit_position, hit_light_direction), 0.0f);
    EXPECT_FLOAT_EQ(LightSources::PDF(light, lit_position, miss_light_direction), 0.0f);
}

Bifrost::Math::Statistics<double> estimate_power(SpotLight light, float disk_depth, Bifrost::Math::Vector2f* light_UVs, int light_UV_count) {
    using namespace optix;

    light.position = make_float3(0, 0, 0);
    light.direction = make_float3(0, 0, 1);

    // Generate samples on a disk below the light source. The disk should be large enough to cover the illuminated area.
    float3 disk_normal = -light.direction;
    float3 disk_center = light.position + disk_depth * light.direction;
    float cos_angle_squared = pow2(light.cos_angle);
    float depth_to_radius = sqrt((1.0f - cos_angle_squared) / cos_angle_squared);
    float disk_radius = depth_to_radius * disk_depth + light.radius;

    int disk_sample_count = 1024;
    auto radiances = std::vector<float>(disk_sample_count);
    for (int i = 0; i < disk_sample_count; ++i)
    {
        auto disk_sample = Distributions::Disk::sample(disk_radius, RNG::sample02(i, { 0u, 0u }));
        float3 disk_position = make_float3(disk_sample.position, 0) + disk_center;

        // Accumulate radiance at disk position
        auto sample_radiances = std::vector<float>(light_UV_count);
        for (int j = 0; j < light_UV_count; ++j) {
            float2 light_UV = make_float2(light_UVs[j].x, light_UVs[j].y);
            auto light_sample = LightSources::sample_radiance(light, disk_position, light_UV);
            if (is_PDF_valid(light_sample.PDF)) {
                float cos_theta = dot(light_sample.direction_to_light, disk_normal);
                sample_radiances[j] = light_sample.radiance.x * cos_theta / light_sample.PDF;
            } else
                sample_radiances[j] = 0.0f;
        }
        radiances[i] = Bifrost::Math::sort_and_pairwise_summation(sample_radiances.begin(), sample_radiances.begin() + light_UV_count) / float(light_UV_count);
    }

    float disk_surface_area = PIf * pow2(disk_radius);
    auto power_statistics = Bifrost::Math::Statistics<double>(radiances.begin(), radiances.begin() + disk_sample_count, [=](auto v) -> double { return *v * disk_surface_area; });
    return power_statistics;
}

GTEST_TEST(SpotLight, power_preservation_when_radius_changes) {
    using namespace optix;

    const unsigned int MAX_LIGHT_SAMPLES = 256u;
    Bifrost::Math::Vector2f UVs[MAX_LIGHT_SAMPLES];
    Bifrost::Math::RNG::fill_progressive_multijittered_bluenoise_samples(UVs, UVs + MAX_LIGHT_SAMPLES);

    SpotLight light;
    light.position = make_float3(0, 0, 0);
    light.direction = make_float3(0, 0, 1);
    light.power = make_float3(1, 1, 1);
    light.cos_angle = 0.5f;

    for (float radius : { 0.0f, 1.0f, 2.0f, 4.0f }) {
        light.radius = radius;
        auto power_statistics = estimate_power(light, 1, UVs, MAX_LIGHT_SAMPLES);
        // printf("Radius %.0f: Power %.3f, normalized std dev: %.3f\n", radius, power_statistics.mean, power_statistics.standard_deviation());
        EXPECT_FLOAT_EQ_EPS(light.power.x, (float)power_statistics.mean, 0.0025f);
    }
}

GTEST_TEST(SpotLight, power_preservation_when_angle_changes) {
    using namespace optix;

    const unsigned int MAX_LIGHT_SAMPLES = 256u;
    Bifrost::Math::Vector2f UVs[MAX_LIGHT_SAMPLES];
    Bifrost::Math::RNG::fill_progressive_multijittered_bluenoise_samples(UVs, UVs + MAX_LIGHT_SAMPLES);

    SpotLight light;
    light.position = make_float3(0, 0, 0);
    light.direction = make_float3(0, 0, 1);
    light.power = make_float3(1, 1, 1);
    light.radius = 0.25f;

    for (float cos_angle : { 0.3f, 0.5f, 0.7f }) {
        light.cos_angle = cos_angle;
        auto power_statistics = estimate_power(light, 1, UVs, MAX_LIGHT_SAMPLES);
        // printf("cos_angle %.3f: Power %.3f, normalized std dev: %.3f\n", cos_angle, power_statistics.mean, power_statistics.standard_deviation());
        EXPECT_FLOAT_EQ_EPS(light.power.x, (float)power_statistics.mean, 0.0045f);
    }
}

} // NS OptiXRenderer

#endif // _OPTIXRENDERER_LIGHT_SOURCES_SPOT_LIGHT_TEST_H_