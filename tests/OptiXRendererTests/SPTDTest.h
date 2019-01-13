// Test spherical pivot transformed distributions.
// ------------------------------------------------------------------------------------------------
// Copyright (C) 2015-2017, Bifrost. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _OPTIXRENDERER_SPTD_TEST_H_
#define _OPTIXRENDERER_SPTD_TEST_H_

#include <Utils.h>

#include <OptiXRenderer/Shading/BSDFs/GGX.h>
#include <OptiXRenderer/SPTD.h>
#include <OptiXRenderer/Utils.h>

#include <gtest/gtest.h>

#include <functional>

namespace OptiXRenderer {

void cubic_for(int width, int height, int depth, const std::function<void(int, int, int)>& f) {
    for (int z = 0; z < depth; ++z)
        for (int y = 0; y < height; ++y)
            for (int x = 0; x < width; ++x)
                f(x, y, z);
}

// Validate that the line through a point and it's transformed point goes through rp.
GTEST_TEST(SPTD, pivet_transform) {
    using namespace optix;

    const int rp_count = 3;
    float3 rps[] = {
        make_float3(0.3f, -0.2f, 0.2f),
        make_float3(-0.1f, 0.2f, 0.1f),
        make_float3(0.4f, -0.1f, 0.3f)
    };

    for (int i = 0; i < rp_count; ++i) {
        float3 rp = rps[i];
        cubic_for(5, 5, 5, [=](int p_x, int p_y, int p_z) {
            if (p_x == 2 && p_y == 2 && p_z == 2) return;

            float3 p = normalize(make_float3(p_x - 2.0f, p_y - 2.0f, p_z - 2.0f));
            float3 pp = SPTD::pivot_transform(p, rp);

            float3 p2rp = normalize(rp - p);
            float3 rp2pp = normalize(pp - rp);

            EXPECT_NORMAL_EQ(p2rp, rp2pp, 0.000001f);
        });
    }
}

// Validate that applying the pivot transform twice around rp equals identity.
GTEST_TEST(SPTD, pivet_transform_identity) {
    using namespace optix;

    cubic_for(3, 3, 3, [](int x, int y, int z) {
        float3 rp = make_float3(x - 1.0f, y - 1.0f, z - 1.0f) * 0.2f;

        cubic_for(3, 3, 3, [=](int p_x, int p_y, int p_z) {
            float3 p = make_float3(p_x - 1.0f, p_y - 1.0f, p_z - 1.0f);
            float3 pp = SPTD::pivot_transform(SPTD::pivot_transform(p, rp), rp);
            EXPECT_FLOAT3_EQ_EPS(p, pp, make_float3(0.000001f));
        });
    });
}

// compute the error between the BRDF and the pivot using Multiple Importance Sampling
float compute_ggx_fitting_error(const SPTD::Pivot& pivot, const optix::float3& wo, float roughness) {
    using namespace optix;
    using namespace OptiXRenderer::Shading::BSDFs;

    const int sample_count = 16;

    float alpha = GGX::alpha_from_roughness(roughness);
    const float full_specularity = 1.0f;

    double error = 0.0;
    int valid_sample_count = 0;
    for (int j = 0; j < sample_count; ++j)
        for (int i = 0; i < sample_count; ++i) {

            const float U1 = (i + 0.5f) / (float)sample_count;
            const float U2 = (j + 0.5f) / (float)sample_count;

            // error with MIS weight
            auto error_from_wi = [&](float3 wi) -> double {
                float3 halfway = normalize(wo + wi);
                float eval_brdf = GGX::evaluate(alpha, full_specularity, wo, wi);
                float pdf_brdf = GGX::PDF(alpha, wo, halfway);
                float eval_pivot = pivot.eval(wi);
                float pdf_pivot = eval_pivot / pivot.amplitude;
                double error = eval_brdf - eval_pivot;
                return error * error / (pdf_pivot + pdf_brdf);
            };

            { // importance sample LTC
                const float3 wi = pivot.sample(U1, U2);
                if (wi.z >= 0.0f) {
                    error += error_from_wi(wi);
                    ++valid_sample_count;
                }
            }

            { // importance sample BRDF
                const auto sample = GGX::sample(alpha, full_specularity, wo, make_float2(U1, U2));
                if (sample.direction.z >= 0.0f && is_PDF_valid(sample.PDF)) {
                    error += error_from_wi(sample.direction);
                    ++valid_sample_count;
                }
            }
        }

    return float(error / valid_sample_count);
}

GTEST_TEST(SPTD, ggx_fitting_error) {
    using namespace optix;

    float roughness[] = { 1.0f / 64.0f, 1.0f / 16.0f, 0.25f, 0.9f };
    float cos_thetas[] = { 0.1f, 0.5f, 0.9f };

    // Errors gotten by printing the errors from a visually pleasent fit.
    float errors[] = { 503744480.0f, 4054393.0f, 732875.375f, 876561.125f,
                       8177.700684f, 3523.881104f, 3427.58374f, 27.771591f,
                       13.444297f, 0.696338f, 0.124728f, 0.039955f };

    for (int r = 0; r < 4; ++r)
        for (int t = 0; t < 3; ++t) {

            float cos_theta = cos_thetas[t];
            float sin_theta = sqrt(1.0f - cos_theta * cos_theta);
            float3 wo = make_float3(sin_theta, 0, cos_theta);

            float4 pivot_params = SPTD::GGX_fit_lookup(wo.z, roughness[r]);
            SPTD::Pivot pivot = { pivot_params.z, pivot_params.x, pivot_params.y };

            float error = compute_ggx_fitting_error(pivot, wo, roughness[r]);
            // printf("[cos theta: %f, roughness: %.4f] => pivot: [amplitude: %.3f, distance: %.3f, theta: %.3f] => error: %f\n", 
            //     cos_theta, roughness[r], pivot.amplitude, pivot.distance, pivot.theta, error);
            // EXPECT_LE(error, errors[t + r * 3] * 1.0001f);
        }
}

// Diffuse area light approximation based on Ambient Aparture Lighting, 2007
GTEST_TEST(SPTD, spherical_cap_union) {
    using namespace optix;

    Cone hemisphere_sphere_cap = Cone::make(make_float3(0.0f, 0.0f, 1.0f), 0.0f);

    { // Test that a light right above the surface (zenith) has an unchanged centroid and solidangle.
        SphereLight light;
        light.power = { 2, 2, 2 };
        light.position = { 0, 0, 1 };
        light.radius = 0.5f;

        // Test solidangle.
        Cone light_sphere_cap = SPTD::sphere_to_sphere_cap(light.position, light.radius);
        float light_solidangle = SPTD::solidangle(light_sphere_cap);
        EXPECT_EQ(light_solidangle, SPTD::solidangle_of_union(light_sphere_cap, hemisphere_sphere_cap));
        EXPECT_EQ(light_solidangle, SPTD::solidangle_of_union(hemisphere_sphere_cap, light_sphere_cap));

        // Test centroid.
        float3 centroid = SPTD::centroid_of_union(light_sphere_cap, hemisphere_sphere_cap);
        EXPECT_NORMAL_EQ(light_sphere_cap.direction, SPTD::centroid_of_union(light_sphere_cap, hemisphere_sphere_cap), 0.00001);
    }

    { // Test that a light at 45 degrees from zenith has an unchanged centroid and solidangle.
        SphereLight light;
        light.power = { 2, 2, 2 };
        light.position = normalize(make_float3(1, 0, 1));
        light.radius = 0.5f;

        // Test solidangle.
        Cone light_sphere_cap = SPTD::sphere_to_sphere_cap(light.position, light.radius);
        float light_solidangle = SPTD::solidangle(light_sphere_cap);
        EXPECT_EQ(light_solidangle, SPTD::solidangle_of_union(light_sphere_cap, hemisphere_sphere_cap));
        EXPECT_EQ(light_solidangle, SPTD::solidangle_of_union(hemisphere_sphere_cap, light_sphere_cap));

        // Test centroid.
        float3 centroid = SPTD::centroid_of_union(light_sphere_cap, hemisphere_sphere_cap);
        EXPECT_NORMAL_EQ(light_sphere_cap.direction, SPTD::centroid_of_union(light_sphere_cap, hemisphere_sphere_cap), 0.00001);
    }

    { // Test that a light at the horizon has half the solidangle and the centroid is in the center of the visible half.
        SphereLight light;
        light.power = { 2, 2, 2 };
        light.position = normalize(make_float3(1, 0, 0));
        light.radius = 0.5f;

        // Test solidangle.
        Cone light_sphere_cap = SPTD::sphere_to_sphere_cap(light.position, light.radius);
        float light_half_solidangle = 0.5f * SPTD::solidangle(light_sphere_cap);
        EXPECT_EQ(light_half_solidangle, SPTD::solidangle_of_union(light_sphere_cap, hemisphere_sphere_cap));
        EXPECT_EQ(light_half_solidangle, SPTD::solidangle_of_union(hemisphere_sphere_cap, light_sphere_cap));

        // Test centroid.
        // float3 centroid = SPTD::centroid_of_union(light_sphere_cap, hemisphere_sphere_cap);
        // EXPECT_NORMAL_EQ(light_sphere_cap.direction, SPTD::centroid_of_union(light_sphere_cap, hemisphere_sphere_cap), 0.00001);
    }
}

inline optix::float3 elongated_highlight_offset(optix::float3 direction_to_camera, optix::float3 direction_to_light, float elongation) {
    using namespace optix;

    optix::float2 camera_to_light = make_float2(direction_to_light) - make_float2(direction_to_camera);
    optix::float2 perfect_reflection_point_2D = make_float2(direction_to_camera) + camera_to_light / (direction_to_light.z + direction_to_camera.z) * direction_to_camera.z;

    float2 bitangent = normalize(camera_to_light);
    float2 tangent = make_float2(-bitangent.y, bitangent.x);

    float2 delta_x = tangent * dot(perfect_reflection_point_2D, tangent);
    float2 delta_y = bitangent * dot(perfect_reflection_point_2D, bitangent);
    float2 warped_offset = perfect_reflection_point_2D - delta_x * elongation - delta_y / elongation;

    return make_float3(warped_offset, 0.0);
}

GTEST_TEST(SPTD, elongation_transformation) {
    using namespace optix;

    float3 cam_pos = make_float3(0, -1, 1);
    float3 light_pos = make_float3(0, 2, 2);

    float3 intersections[] = { make_float3(0, 0.5f, 0), make_float3(0.5f, 0, 0), make_float3(0, 0, 0), make_float3(-1.5f, 0, 0), make_float3(0, -1.5f, 0) };
    for (float3 intersection : intersections) {
        float elongation = 2.0f;
        float3 direction_to_camera = cam_pos - intersection;
        float3 direction_to_light = light_pos - intersection;
        float3 warped_intersection = intersection + elongated_highlight_offset(direction_to_camera, direction_to_light, elongation);
        float3 expected_warped_intersection = { intersection.x * elongation, intersection.y / elongation, intersection.z };
        EXPECT_FLOAT3_EQ_EPS(expected_warped_intersection, warped_intersection, make_float3(0.0001f, 0.0001f, 0.0001f));
    }
}

GTEST_TEST(SPTD, centroid_and_solidangle_on_hemisphere) {
    using namespace optix;

    Cone hemisphere = Cone::make(make_float3(0.0f, 0.0f, 1.0f), 0.0f);

    int angle_count = 4;
    for (int a = 0; a < angle_count; ++a) {
        float angle = float(a) / angle_count;
        cubic_for(5, 5, 5, [=](int p_x, int p_y, int p_z) {
            if (p_x == 2 && p_y == 2 && p_z == 2) return;

            float3 direction = normalize(make_float3(p_x - 2.0f, p_y - 2.0f, p_z - 2.0f));
            if (direction.z == -1.0f) return;
            Cone cone = { direction, angle };

            auto centroid_and_solidangle = SPTD::centroid_and_solidangle_on_hemisphere(cone);
            EXPECT_FLOAT3_EQ_EPS(SPTD::centroid_of_union(cone, hemisphere), centroid_and_solidangle.centroid, make_float3(0.000001f, 0.000001f, 0.000001f));
            EXPECT_FLOAT_EQ_EPS(SPTD::solidangle_of_union(cone, hemisphere), centroid_and_solidangle.solidangle, 0.000001f);
        });
    }
}

} // NS OptiXRenderer

#endif // _OPTIXRENDERER_SPTD_TEST_H_