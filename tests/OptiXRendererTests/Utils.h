// OptiXRenderer testing utils.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _OPTIXRENDERERTEST_UTILS_H_
#define _OPTIXRENDERERTEST_UTILS_H_

#include <optixu/optixu_math_namespace.h>

#include <string>

//-----------------------------------------------------------------------------
// Comparison helpers.
//-----------------------------------------------------------------------------

inline bool equal_float3(optix::float3 lhs, optix::float3 rhs) {
    return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z;
}

#define EXPECT_FLOAT3_EQ(expected, actual) EXPECT_PRED2(equal_float3, expected, actual)

inline bool almost_equal_eps(float lhs, float rhs, float eps) {
    return abs(lhs - rhs) <= abs(eps);
}

#define EXPECT_FLOAT_EQ_EPS(expected, actual, epsilon) EXPECT_PRED3(almost_equal_eps, expected, actual, epsilon)

inline bool almost_equal_percentage(float lhs, float rhs, float percentage) {
    float eps = lhs * percentage;
    return almost_equal_eps(lhs, rhs, eps);
}

#define EXPECT_FLOAT_EQ_PCT(expected, actual, percentage) EXPECT_PRED3(almost_equal_percentage, expected, actual, percentage)

inline bool equal_float3_eps(optix::float3 lhs, optix::float3 rhs, float epsilon) {
    return almost_equal_eps(lhs.x, rhs.x, epsilon) &&
           almost_equal_eps(lhs.y, rhs.y, epsilon) &&
           almost_equal_eps(lhs.z, rhs.z, epsilon);
}

#define EXPECT_COLOR_EQ_EPS(expected, actual, epsilon) EXPECT_PRED3(equal_float3_eps, expected, actual, epsilon)
#define EXPECT_FLOAT3_EQ_EPS(expected, actual, epsilon) EXPECT_PRED3(equal_float3_eps, expected, actual, epsilon)

inline bool equal_float3_percentage(optix::float3 expected, optix::float3 actual, float percentage) {
    optix::float3 epsilon = expected * percentage;
    return almost_equal_eps(expected.x, actual.x, epsilon.x) &&
           almost_equal_eps(expected.y, actual.y, epsilon.y) &&
           almost_equal_eps(expected.z, actual.z, epsilon.z);
}

#define EXPECT_COLOR_EQ_PCT(expected, actual, percentage) EXPECT_PRED3(equal_float3_percentage, expected, actual, percentage)
#define EXPECT_FLOAT3_EQ_PCT(expected, actual, percentage) EXPECT_PRED3(equal_float3_percentage, expected, actual, percentage)

inline bool float3_less_or_equal(optix::float3 value, float threshold) {
    return value.x <= threshold && value.y <= threshold && value.z <= threshold;
}

#define EXPECT_FLOAT3_LE(value, threshold) EXPECT_PRED2(float3_less_or_equal, value, threshold)

inline bool float3_greater_or_equal(optix::float3 value, float threshold) {
    return value.x >= threshold && value.y >= threshold && value.z >= threshold;
}

#define EXPECT_FLOAT3_GE(value, threshold) EXPECT_PRED2(float3_greater_or_equal, value, threshold)

inline bool equal_normal_eps(optix::float3 lhs, optix::float3 rhs, double epsilon) {
    using namespace optix;

    double3 delta = { double(lhs.x) - double(rhs.x), double(lhs.y) - double(rhs.y), double(lhs.z) - double(rhs.z) };
    double length_squared = delta.x * delta.x + delta.y * delta.y + delta.z * delta.z;
    return length_squared < epsilon * epsilon;
}

#define EXPECT_NORMAL_EQ(expected, actual, epsilon) EXPECT_PRED3(equal_normal_eps, expected, actual, epsilon)

//-----------------------------------------------------------------------------
// To string functions.
//-----------------------------------------------------------------------------

inline std::ostream& operator<<(std::ostream& s, const optix::float3 v) {
    return s << "[x: " << v.x << ", y: " << v.y << ", z: " << v.z << "]";
}

#endif // _OPTIXRENDERERTEST_UTILS_H_