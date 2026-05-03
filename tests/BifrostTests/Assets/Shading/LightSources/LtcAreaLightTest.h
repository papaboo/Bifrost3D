// Test Bifrost's LTC area lights.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _BIFROST_ASSETS_SHADING_LIGHTSOURCES_LTC_AREA_LIGHT_TEST_H_
#define _BIFROST_ASSETS_SHADING_LIGHTSOURCES_LTC_AREA_LIGHT_TEST_H_

#include <Assets/Shading/BSDFs/GGXTest.h>
#include <Assets/Shading/BSDFs/LambertTest.h>
#include <Assets/Shading/BSDFs/OrenNayarTest.h>
#include <Assets/Shading/BSDFTestUtils.h>
#include <Expects.h>

#include <Bifrost/Assets/Shading/LightSources/LtcAreaLight.h>
#include <Bifrost/Assets/Shading/LinearlyTransformedCosines.h>
#include <Bifrost/Math/Intersect.h>

#include <gtest/gtest.h>

namespace Bifrost::Assets::Shading::LightSources {

struct EmissiveTriangle {
    Math::RGB emission;
    Math::Vector3f v0, v1, v2;
    Math::Vector3f normal;
    float surface_area;
    bool two_sided = true;

    EmissiveTriangle(Math::RGB emission, Math::Vector3f v0, Math::Vector3f v1, Math::Vector3f v2)
        : emission(emission), v0(v0), v1(v1), v2(v2) {
        Math::Vector3f light_surface_normal = 0.5f * Math::Trianglef::get_up(v0, v1, v2);
        surface_area = magnitude(light_surface_normal);
        normal = light_surface_normal / surface_area;
    }

    Math::Vector3f* vertices() { return &v0; }

    inline Math::RGB evaluate(Math::IsotropicLTC ltc_bsdf_model, Math::Vector3f wo, Math::Vector3f surface_point, Math::Vector3f surface_normal) const {
        return LtcAreaLight::evaluate_triangle_light(ltc_bsdf_model, wo, surface_point, surface_normal, &v0, emission, two_sided);
    }
};

template <typename BSDFModel>
Math::RGB triangle_light_integration_error(Math::Vector3f wo, BSDFModel bsdf_model, Math::IsotropicLTC ltc_bsdf_model, EmissiveTriangle light,
                                           int max_sample_count = 4096) {
    using namespace Bifrost::Math;

    // Define the surface plane to illuminate.
    // The surface passes through origo and the normal is along positive z.
    Vector3f surface_point = { 0, 0, 0 };
    Vector3f surface_normal = { 0, 0, 1 };

    // Integrate light
    RGB monte_carlo_estimation = RGB::black();
    {
        // Sample lights
        for (int s = 0; s < max_sample_count; ++s) {
            auto light_sample = Distributions::Triangle::sample(light.v0, light.v1, light.v2, light.surface_area, BSDFTestUtils::bsdf_rng_sample2f(s));

            Vector3f direction_to_light = light_sample.position - surface_point;
            float light_distance_squared = magnitude_squared(direction_to_light);
            Vector3f wi = direction_to_light / sqrt(light_distance_squared);

            // Converting from local area PDF to solid angle PDF wrt the surface point.
            float area_PDF_to_solid_angle_PDF = light_distance_squared / abs(dot(wi, light.normal));
            float light_solid_angle_PDF = light_sample.PDF * area_PDF_to_solid_angle_PDF;

            auto bsdf_response = bsdf_model.evaluate_with_PDF(wo, wi);
            float abs_cos_theta_i = abs(dot(surface_normal, wi));

            RGB contribution = light.emission * bsdf_response.reflectance * abs_cos_theta_i / light_solid_angle_PDF;
            float mis_weight = MonteCarlo::balance_heuristic(light_solid_angle_PDF, bsdf_response.PDF.value());
            monte_carlo_estimation += contribution * mis_weight;
        }

        // Sample BSDF
        for (int s = 0; s < max_sample_count; ++s) {
            auto bsdf_sample = bsdf_model.sample(wo, BSDFTestUtils::bsdf_rng_sample3f(s, max_sample_count));
            Vector3f wi = bsdf_sample.direction;

            auto light_hit = Intersect::ray_triangle(Ray(surface_point, wi), light.vertices(), light.two_sided);
            if (light_hit.hit()) {
                // Converting from local area PDF to solid angle PDF wrt the surface point.
                float light_surface_PDF = Distributions::Triangle::PDF(light.surface_area);
                float area_PDF_to_solid_angle_PDF = pow2(light_hit.distance) / abs(dot(wi, light.normal));
                float light_solid_angle_PDF = light_surface_PDF * area_PDF_to_solid_angle_PDF;

                float abs_cos_theta_i = abs(dot(surface_normal, wi));

                RGB contribution = light.emission * bsdf_sample.reflectance * abs_cos_theta_i / bsdf_sample.PDF.value();
                float mis_weight = MonteCarlo::balance_heuristic(bsdf_sample.PDF.value(), light_solid_angle_PDF);
                monte_carlo_estimation += contribution * mis_weight;
            }
        }

        monte_carlo_estimation /= max_sample_count;
    }

    RGB ltc_estimation = light.evaluate(ltc_bsdf_model, wo, surface_point, surface_normal);

    RGB error = monte_carlo_estimation - ltc_estimation;
    return { abs(error.r), abs(error.g), abs(error.b) };
}

GTEST_TEST(Assets_Shading_LightSources_LTC, triangle_light_only_shades_in_front) {
    using namespace Bifrost::Math;

    // Define the surface plane to illuminate.
    // The surface passes through origo and the normal is along positive z.
    Vector3f surface_point = { 0, 0, 0 };
    Vector3f surface_normal = { 0, 0, 1 };
    Vector3f wo = { 0, 0, 1 };

    auto ltc_bsdf = Bifrost::Assets::Shading::LTC::lambert_LTC_coefficients();

    // Define triangle light above surface plane at (0, 0, 0) with normal pointing upwards.
    float distance_to_surface = 1.0f;
    Vector3f light_v0 = { -1, 1, distance_to_surface };
    Vector3f light_v1 = { 1, -1, distance_to_surface };
    Vector3f light_v2 = { 1, 1, distance_to_surface };
    auto light = EmissiveTriangle(RGB::white(), light_v0, light_v1, light_v2);
    // Assert that the light's normal points upwards
    EXPECT_VECTOR3F_EQ(Vector3f(0, 0, 1), light.normal);

    light.two_sided = true;
    RGB radiance = light.evaluate(ltc_bsdf, wo, surface_point, surface_normal);
    EXPECT_RGB_GT(radiance, 0.0f);

    light.two_sided = false;
    radiance = light.evaluate(ltc_bsdf, wo, surface_point, surface_normal);
    EXPECT_RGB_EQ(radiance, RGB::black());
}

GTEST_TEST(Assets_Shading_LightSources_LTC, lambert_integration_over_triangle_light_error) {
    using namespace Bifrost::Math;

    auto lambert_bsdf = BSDFs::LambertWrapper();
    auto ltc_bsdf = Bifrost::Assets::Shading::LTC::lambert_LTC_coefficients();

    // Define triangle light above surface plane at (0, 0, 0) with normal pointing upwards.
    float distance_to_surface = 1.0f;
    Vector3f light_v0 = { -1, 1, distance_to_surface };
    Vector3f light_v1 = { 1, 1, distance_to_surface };
    Vector3f light_v2 = { 1, -1, distance_to_surface };
    auto light = EmissiveTriangle(RGB::white(), light_v0, light_v1, light_v2);

    int max_wo_sample_count = 8;
    int max_sample_count = 4096;
    for (int wo_i = 0; wo_i < max_wo_sample_count; ++wo_i) {
        Vector3f wo = Distributions::Cosine::sample(BSDFTestUtils::bsdf_rng_sample2f(wo_i)).direction;
        float cos_theta_o = wo.z;
        RGB ltc_integration_error = triangle_light_integration_error(wo, lambert_bsdf, ltc_bsdf, light, max_sample_count);

        EXPECT_RGB_EQ_EPS(RGB(0.0f), ltc_integration_error, 1e-5f);
    }
}

GTEST_TEST(Assets_Shading_LightSources_LTC, oren_nayar_integration_over_triangle_light_error) {
    using namespace Bifrost::Math;

    // Define triangle light above surface plane at (0, 0, 0) with normal pointing upwards.
    float distance_to_surface = 1.0f;
    Vector3f light_v0 = { -1, 1, distance_to_surface };
    Vector3f light_v1 = { 1, 1, distance_to_surface };
    Vector3f light_v2 = { 1, -1, distance_to_surface };
    auto light = EmissiveTriangle(RGB::white(), light_v0, light_v1, light_v2);

    int max_wo_sample_count = 8;
    int max_sample_count = 4096;

    float error = 0.0f;
    for (float roughness : { 0.0f, 0.5f, 1.0f }) {
        auto oren_nayar_bsdf = BSDFs::OrenNayarWrapper(roughness);

        for (int wo_i = 0; wo_i < max_wo_sample_count; ++wo_i) {
            Vector3f wo = Distributions::Cosine::sample(BSDFTestUtils::bsdf_rng_sample2f(wo_i)).direction;
            float cos_theta_o = wo.z;

            auto ltc_bsdf = Bifrost::Assets::Shading::LTC::oren_nayar_LTC_coefficients(cos_theta_o, roughness);

            RGB ltc_integration_error = triangle_light_integration_error(wo, oren_nayar_bsdf, ltc_bsdf, light, max_sample_count);
            error += ltc_integration_error.r;
        }
    }
    error /= max_wo_sample_count;

    EXPECT_FLOAT_EQ_EPS(0.0077f, error, 1e-4f);
}

GTEST_TEST(Assets_Shading_LightSources_LTC, GGX_integration_over_triangle_light_error) {
    using namespace Bifrost::Math;

    // Define triangle light above surface plane at (0, 0, 0) with normal pointing upwards.
    float distance_to_surface = 1.0f;
    Vector3f light_v0 = { -1, 1, distance_to_surface };
    Vector3f light_v1 = { 1, 1, distance_to_surface };
    Vector3f light_v2 = { 1, -1, distance_to_surface };
    auto light = EmissiveTriangle(RGB::white(), light_v0, light_v1, light_v2);

    int max_wo_sample_count = 8;
    int max_sample_count = 4096;

    float error = 0.0f;
    for (float roughness : { 0.2f, 0.6f, 1.0f }) {
        const float full_specularity = 1.0f; // The LTCs are fitted without the Fresnel term
        auto ggx_bsdf = BSDFs::GGXReflectionWrapper(BSDFs::GGX::alpha_from_roughness(roughness), full_specularity);
        ggx_bsdf.normalized_rho(true);

        for (int wo_i = 0; wo_i < max_wo_sample_count; ++wo_i) {
            Vector3f wo = Distributions::Cosine::sample(BSDFTestUtils::bsdf_rng_sample2f(wo_i)).direction;
            float cos_theta_o = wo.z;

            auto ltc_bsdf = Bifrost::Assets::Shading::LTC::GGX_reflection_LTC_coefficients(cos_theta_o, roughness);

            RGB ltc_integration_error = triangle_light_integration_error(wo, ggx_bsdf, ltc_bsdf, light, max_sample_count);
            error += ltc_integration_error.r;
        }
    }
    error /= max_wo_sample_count;

    EXPECT_FLOAT_EQ_EPS(0.01611f, error, 1e-4f);
}

GTEST_TEST(Assets_Shading_LightSources_LTC, light_behind_surface_does_not_illuminate) {
    using namespace Bifrost::Math;

    Vector3f surface_point = { 0, 0, 0 };
    Vector3f surface_normal = { 0, 0, 1 };

    // Define triangle light below surface plane at (0, 0, 0) with normal pointing upwards.
    float distance_to_surface = -1.0f;
    Vector3f light_v0 = { -1, 1, distance_to_surface };
    Vector3f light_v1 = { 1, -1, distance_to_surface };
    Vector3f light_v2 = { 1, 1, distance_to_surface };
    auto light = EmissiveTriangle(RGB::white(), light_v0, light_v1, light_v2);
    light.two_sided = true; // Ensure that the light always casts light at the surface.

    auto ltc_lambert_bsdf = Bifrost::Assets::Shading::LTC::lambert_LTC_coefficients();

    for (float cos_theta_o : { 0.2f, 0.6f, 1.0f }) {
        Vector3f wo = BSDFTestUtils::w_from_cos_theta(cos_theta_o);

        RGB radiance = light.evaluate(ltc_lambert_bsdf, wo, surface_point, surface_normal);
        EXPECT_RGB_EQ(radiance, RGB::black());
    }
}

// When two vertices are below the horizon, then the triangle should be cropped to a new triangle on the horizon.
GTEST_TEST(Assets_Shading_LightSources_LTC, crop_triangle_by_horizon_with_one_vertex_above_horizon) {
    using namespace Bifrost::Math;

    Vector3f vertex_above = { 1, 0, 2 };
    Vector3f vertex1_below = { 1, -1, -1 };
    Vector3f vertex2_below = { 1, 1, -1 };

    Vector3f vertex1_on_horizon = (vertex1_below * 2 + vertex_above) / 3;
    Vector3f vertex2_on_horizon = (vertex2_below * 2 + vertex_above) / 3;

    auto test_triangle_configuration = [&](int vertex1_below_index, int vertex2_below_index) {
        Vector3f clipped_vertices[4] = { vertex_above, vertex_above, vertex_above, Vector3f::zero() };
        clipped_vertices[vertex1_below_index] = vertex1_below;
        clipped_vertices[vertex2_below_index] = vertex2_below;

        Triangle original_tri = { clipped_vertices[0], clipped_vertices[1], clipped_vertices[2] };

        int clipped_vertex_count = LtcAreaLight::clip_triangle_to_horizon(clipped_vertices);
        EXPECT_EQ(3, clipped_vertex_count) << "Vertex1 below index: " << vertex1_below_index << ", vertex2 below index: " << vertex2_below_index;

        EXPECT_VECTOR3F_EQ(vertex1_on_horizon, clipped_vertices[vertex1_below_index]) << "Vertex1 below index: " << vertex1_below_index << ", vertex2 below index: " << vertex2_below_index;
        EXPECT_VECTOR3F_EQ(vertex2_on_horizon, clipped_vertices[vertex2_below_index]) << "Vertex1 below index: " << vertex1_below_index << ", vertex2 below index: " << vertex2_below_index;

        Triangle clipped_tri = { clipped_vertices[0], clipped_vertices[1], clipped_vertices[2] };
        EXPECT_VECTOR3F_EQ(original_tri.get_normal(), clipped_tri.get_normal()) << "Vertex1 below index: " << vertex1_below_index << ", vertex2 below index: " << vertex2_below_index;
    };

    // Vertex 0 is above the horizon
    test_triangle_configuration(1, 2);
    test_triangle_configuration(2, 1);

    // Vertex 1 is above the horizon
    test_triangle_configuration(0, 2);
    test_triangle_configuration(2, 0);

    // Vertex 2 is above the horizon
    test_triangle_configuration(0, 1);
    test_triangle_configuration(1, 0);
}

// When two vertices are below the horizon, then the triangle should be cropped to a new triangle on the horizon.
GTEST_TEST(Assets_Shading_LightSources_LTC, crop_triangle_by_horizon_with_two_vertices_above_horizon) {
    using namespace Bifrost::Math;

    Vector3f vertex_below = { 1, 0, -2 };
    Vector3f vertex1_above = { 1, -1, 1 };
    Vector3f vertex2_above = { 1, 1, 1 };

    Vector3f vertex1_on_horizon = (vertex1_above * 2 + vertex_below) / 3;
    Vector3f vertex2_on_horizon = (vertex2_above * 2 + vertex_below) / 3;

    auto test_triangle_configuration = [&](int vertex1_above_index, int vertex2_above_index) {
        Vector3f clipped_vertices[4] = { vertex_below, vertex_below, vertex_below, Vector3f::zero() };
        clipped_vertices[vertex1_above_index] = vertex1_above;
        clipped_vertices[vertex2_above_index] = vertex2_above;

        Triangle original_tri = { clipped_vertices[0], clipped_vertices[1], clipped_vertices[2] };

        int clipped_vertex_count = LtcAreaLight::clip_triangle_to_horizon(clipped_vertices);
        EXPECT_EQ(4, clipped_vertex_count) << "Vertex1 above index: " << vertex1_above_index << ", vertex2 above index: " << vertex2_above_index;

        // Vertices above horizon are preserved and vertices on horizon are added.
        // To preserve the winding order, the vertices index may change in the clipped array.
        for (Vector3f clipped_vertex : clipped_vertices) {
            bool vertex_found = false;
            for (Vector3f expected_vertex_position : { vertex1_above, vertex2_above, vertex1_on_horizon, vertex2_on_horizon })
                vertex_found |= almost_equal(expected_vertex_position, clipped_vertex);
            EXPECT_TRUE(vertex_found) << "clipped vertex: " << clipped_vertex << " should be one of the expected vertex positions";
        }

        // Winding order is preserved
        for (int base_index = 0; base_index < 4; ++base_index) {
            Triangle partial_tri = { clipped_vertices[base_index], clipped_vertices[(base_index + 1) % 4], clipped_vertices[(base_index + 2) % 4] };
            EXPECT_VECTOR3F_EQ(original_tri.get_normal(), partial_tri.get_normal()) << "Vertex1 above index: " << vertex1_above_index << ", vertex2 above index: " << vertex2_above_index;
        }
    };

    // Vertex 0 is below the horizon
    test_triangle_configuration(1, 2);
    test_triangle_configuration(2, 1);

    // Vertex 1 is below the horizon
    test_triangle_configuration(0, 2);
    test_triangle_configuration(2, 0);

    // Vertex 2 is below the horizon
    test_triangle_configuration(0, 1);
    test_triangle_configuration(1, 0);
}

GTEST_TEST(Assets_Shading_LightSources_LTC, light_clipping_on_lambertian_surface_has_no_error) {
    using namespace Bifrost::Math;

    auto lambert_bsdf = BSDFs::LambertWrapper();
    auto ltc_lambert_bsdf = Bifrost::Assets::Shading::LTC::lambert_LTC_coefficients();

    // Test with different light vertices above and below the horizon.
    // The three first bits in the bitmask indicates which vertex is above or below hte horizon.
    // 0 and 7 are excluded as the light would be completely above or below the horizon and not part of the test.
    for (int vertex_flip_bitmask = 1; vertex_flip_bitmask < 7; ++vertex_flip_bitmask) {
        int v0_above_horizon = vertex_flip_bitmask & 1;
        int v1_above_horizon = vertex_flip_bitmask & 2;
        int v2_above_horizon = vertex_flip_bitmask & 4;

        Vector3f light_v0 = { 1, -1, v0_above_horizon ? 1.0f : -1.0f };
        Vector3f light_v1 = { 1, 0, v1_above_horizon ? 1.0f : -1.0f };
        Vector3f light_v2 = { 1, 1, v2_above_horizon ? 1.0f : -1.0f };
        auto light = EmissiveTriangle(RGB::white(), light_v0, light_v1, light_v2);
        light.two_sided = true; // We ignore winding order and light direction to make the test simpler.

        Vector3f wo = { 0, 0, 1 };

        RGB ltc_integration_error = triangle_light_integration_error(wo, lambert_bsdf, ltc_lambert_bsdf, light);
        EXPECT_RGB_EQ_EPS(RGB(0), ltc_integration_error, 1e-4f);
    }
}

GTEST_TEST(Assets_Shading_LightSources_LTC, LTC_evaluation_is_shading_space_rotation_agnostic) {
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

        // Create light at reflected view direction, so the light has the same location relative to the view direction.
        Vector3f wi = reflect(wo, surface_normal);
        Vector3f w_tangent = cross(wo, surface_normal);
        Vector3f light_v0 = surface_normal;
        Vector3f light_v1 = surface_normal + Vector3f(wi.x, wi.y, 0);
        Vector3f light_v2 = surface_normal + Vector3f(w_tangent.x, w_tangent.y, 0);
        auto light = EmissiveTriangle(RGB::white(), light_v0, light_v1, light_v2);

        radiances[i] = light.evaluate(ltc_bsdf, wo, surface_point, surface_normal);
    }

    EXPECT_RGB_EQ_EPS(radiances[0], radiances[1], 1e-5f);
    EXPECT_RGB_EQ_EPS(radiances[0], radiances[2], 1e-5f);
}

} // NS Bifrost::Assets::Shading::LightSources

#endif // _BIFROST_ASSETS_SHADING_LIGHTSOURCES_LTC_AREA_LIGHT_TEST_H_