// DX11Renderer testing utils.
// -------------------------------------------------------------------------------------------------
// Copyright (C) 2015-2017, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// -------------------------------------------------------------------------------------------------

#ifndef _DX11RENDERERTEST_UTILS_H_
#define _DX11RENDERERTEST_UTILS_H_

#include <DX11Renderer/Tonemapper.h>
#include <DX11Renderer/Utils.h>

// -------------------------------------------------------------------------------------------------
// Typedefs.
// -------------------------------------------------------------------------------------------------

using half4 = Cogwheel::Math::Vector4<half>;

// -------------------------------------------------------------------------------------------------
// Utility functions
// -------------------------------------------------------------------------------------------------

inline DX11Renderer::OID3D11Buffer create_tonemapping_constants(DX11Renderer::OID3D11Device1& device, float min_log_luminance, float max_log_luminance, float min_histogram_percentage = 0.8f, float max_histogram_percentage = 0.95f) {

    DX11Renderer::Tonemapper::Constants constants;
    constants.min_log_luminance = min_log_luminance;
    constants.max_log_luminance = max_log_luminance;
    constants.min_histogram_percentage = min_histogram_percentage;
    constants.max_histogram_percentage = max_histogram_percentage;
    constants.log_lumiance_bias = 0.0f;
    // Disable eye adaptation by setting adaptation to infinity.
    constants.eye_adaptation_brightness = constants.eye_adaptation_darkness = std::numeric_limits<float>::infinity();
    constants.delta_time = 1.0f / 60.0f;

    DX11Renderer::OID3D11Buffer constant_buffer;
    THROW_ON_FAILURE(DX11Renderer::create_constant_buffer(device, constants, &constant_buffer));
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

static bool equal_vector3f(Cogwheel::Math::Vector3f lhs, Cogwheel::Math::Vector3f rhs) {
    return Cogwheel::Math::almost_equal(lhs.x, rhs.x) && Cogwheel::Math::almost_equal(lhs.y, rhs.y) && Cogwheel::Math::almost_equal(lhs.z, rhs.z);
}
#define EXPECT_VECTOR3F_EQ(expected, actual) EXPECT_PRED2(equal_vector3f, expected, actual)

static bool equal_Vector3f_pct(Cogwheel::Math::Vector3f expected, Cogwheel::Math::Vector3f actual, Cogwheel::Math::Vector3f pct) {
    auto eps = expected * pct;
    return abs(expected.x - actual.x) < eps.x && abs(expected.y - actual.y) < eps.y && abs(expected.z - actual.z) < eps.z;
}
#define EXPECT_VECTOR3F_EQ_PCT(expected, actual, pct) EXPECT_PRED3(equal_Vector3f_pct, expected, actual, pct)

#endif // _DX11RENDERERTEST_UTILS_H_