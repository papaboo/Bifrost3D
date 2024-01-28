// DX11Renderer testing utils.
// -------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// -------------------------------------------------------------------------------------------------

#ifndef _DX11RENDERERTEST_UTILS_H_
#define _DX11RENDERERTEST_UTILS_H_

#include <DX11Renderer/CameraEffects.h>
#include <DX11Renderer/Utils.h>

// -------------------------------------------------------------------------------------------------
// Typedefs.
// -------------------------------------------------------------------------------------------------

using half4 = Bifrost::Math::Vector4<Bifrost::Math::half>;

// -------------------------------------------------------------------------------------------------
// Utility functions
// -------------------------------------------------------------------------------------------------

inline DX11Renderer::OBuffer create_camera_effects_constants(DX11Renderer::ODevice1& device, DX11Renderer::int2 viewport_size, float min_log_luminance, float max_log_luminance, float min_histogram_percentage = 0.8f, float max_histogram_percentage = 0.95f) {

    DX11Renderer::CameraEffects::Constants constants;
    constants.input_viewport = { 0.0f, 0.0f, float(viewport_size.x), float(viewport_size.y) };
    constants.output_pixel_offset = { 0, 0 };
    constants.min_log_luminance = min_log_luminance;
    constants.max_log_luminance = max_log_luminance;
    constants.min_histogram_percentage = min_histogram_percentage;
    constants.max_histogram_percentage = max_histogram_percentage;
    constants.log_lumiance_bias = 0.0f;
    // Disable eye adaptation by setting adaptation to infinity.
    constants.eye_adaptation_brightness = constants.eye_adaptation_darkness = std::numeric_limits<float>::infinity();
    constants.bloom_threshold = std::numeric_limits<float>::infinity();
    constants.delta_time = 1.0f / 60.0f;

    DX11Renderer::OBuffer constant_buffer;
    THROW_DX11_ERROR(DX11Renderer::create_constant_buffer(device, constants, &constant_buffer));
    return constant_buffer;
}

// -------------------------------------------------------------------------------------------------
// Comparison helpers.
// -------------------------------------------------------------------------------------------------

inline bool almost_equal_eps(float lhs, float rhs, float eps) {
    return abs(lhs - rhs) < eps;
}
#define EXPECT_FLOAT_EQ_EPS(expected, actual, epsilon) EXPECT_PRED3(almost_equal_eps, expected, actual, epsilon)

inline bool almost_equal_percentage(float lhs, float rhs, float percentage) {
    float eps = lhs * percentage;
    return almost_equal_eps(lhs, rhs, eps);
}
#define EXPECT_FLOAT_EQ_PCT(expected, actual, percentage) EXPECT_PRED3(almost_equal_percentage, expected, actual, percentage)

inline bool double_almost_equal_eps(double lhs, double rhs, double eps) {
    return abs(lhs - rhs) < eps;
}
#define EXPECT_DOUBLE_EQ_EPS(expected, actual, epsilon) EXPECT_PRED3(double_almost_equal_eps, expected, actual, epsilon)

inline bool double_almost_equal_percentage(double lhs, double rhs, double percentage) {
    double eps = lhs * percentage;
    return double_almost_equal_eps(lhs, rhs, eps);
}
#define EXPECT_DOUBLE_EQ_PCT(expected, actual, percentage) EXPECT_PRED3(double_almost_equal_percentage, expected, actual, percentage)

static bool equal_vector3f(Bifrost::Math::Vector3f lhs, Bifrost::Math::Vector3f rhs) {
    return Bifrost::Math::almost_equal(lhs.x, rhs.x) && Bifrost::Math::almost_equal(lhs.y, rhs.y) && Bifrost::Math::almost_equal(lhs.z, rhs.z);
}
#define EXPECT_VECTOR3F_EQ(expected, actual) EXPECT_PRED2(equal_vector3f, expected, actual)

static bool equal_Vector3f_pct(Bifrost::Math::Vector3f expected, Bifrost::Math::Vector3f actual, Bifrost::Math::Vector3f pct) {
    auto eps = expected * pct;
    return abs(expected.x - actual.x) < eps.x && abs(expected.y - actual.y) < eps.y && abs(expected.z - actual.z) < eps.z;
}
#define EXPECT_VECTOR3F_EQ_PCT(expected, actual, pct) EXPECT_PRED3(equal_Vector3f_pct, expected, actual, pct)

#endif // _DX11RENDERERTEST_UTILS_H_