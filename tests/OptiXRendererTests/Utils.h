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

inline bool almost_equal_eps(float lhs, float rhs, float eps) {
    return lhs < rhs + eps && lhs + eps > rhs;
}

#define EXPECT_FLOAT_EQ_EPS(expected, actual, epsilon) EXPECT_PRED3(almost_equal_eps, expected, actual, epsilon)

inline bool almost_equal_percentage(float lhs, float rhs, float percentage) {
    float eps = lhs * percentage;
    return almost_equal_eps(lhs, rhs, eps);
}

#define EXPECT_FLOAT_EQ_PCT(expected, actual, percentage) EXPECT_PRED3(almost_equal_percentage, expected, actual, percentage)

inline bool equal_float3_eps(optix::float3 lhs, optix::float3 rhs, optix::float3 epsilon) {
    return abs(lhs.x - rhs.x) < epsilon.z && abs(lhs.y - rhs.y) < epsilon.y && abs(lhs.z - rhs.z) < epsilon.z;
}

#define EXPECT_COLOR_EQ_EPS(expected, actual, epsilon) EXPECT_PRED3(equal_float3_eps, expected, actual, epsilon)
#define EXPECT_FLOAT3_EQ_EPS(expected, actual, epsilon) EXPECT_PRED3(equal_float3_eps, expected, actual, epsilon)

inline bool equal_float3_percentage(optix::float3 expected, optix::float3 actual, optix::float3 percentage) {
    optix::float3 eps = expected * percentage;
    return equal_float3_eps(expected, actual, eps);
}

#define EXPECT_COLOR_EQ_PCT(expected, actual, percentage) EXPECT_PRED3(equal_float3_eps, expected, actual, percentage)
#define EXPECT_FLOAT3_EQ_PCT(expected, actual, percentage) EXPECT_PRED3(equal_float3_eps, expected, actual, percentage)

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