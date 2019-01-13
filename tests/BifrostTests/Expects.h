// Helper expect definitions.
// ------------------------------------------------------------------------------------------------
// Copyright (C) 2015-2016, Bifrost. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#include <Bifrost/Math/AABB.h>
#include <Bifrost/Math/Color.h>
#include <Bifrost/Math/Quaternion.h>

#include <gtest/gtest.h>

#ifndef _BIFROST_TESTS_EXPECTS_H_
#define _BIFROST_TESTS_EXPECTS_H_

// ------------------------------------------------------------------------------------------------
// Single values
// ------------------------------------------------------------------------------------------------

inline bool float_in_range(float min, float max, float actual) {
    return min <= actual && actual <= max;
}
#define EXPECT_FLOAT_IN_RANGE(min, max, actual) EXPECT_PRED3(float_in_range, min, max, actual)

inline bool almost_equal_eps(float lhs, float rhs, float eps) {
    return abs(lhs - rhs) <= eps;
}
#define EXPECT_FLOAT_EQ_EPS(expected, actual, epsilon) EXPECT_PRED3(almost_equal_eps, expected, actual, epsilon)

inline bool almost_equal_percentage(float lhs, float rhs, float percentage) {
    float eps = lhs * percentage;
    return almost_equal_eps(lhs, rhs, eps);
}
#define EXPECT_FLOAT_EQ_PCT(expected, actual, percentage) EXPECT_PRED3(almost_equal_percentage, expected, actual, percentage)

inline bool double_almost_equal_eps(double lhs, double rhs, double eps) {
    return abs(lhs - rhs) <= eps;
}
#define EXPECT_DOUBLE_EQ_EPS(expected, actual, epsilon) EXPECT_PRED3(double_almost_equal_eps, expected, actual, epsilon)

inline bool double_almost_equal_percentage(double lhs, double rhs, double percentage) {
    double eps = lhs * percentage;
    return double_almost_equal_eps(lhs, rhs, eps);
}
#define EXPECT_DOUBLE_EQ_PCT(expected, actual, percentage) EXPECT_PRED3(double_almost_equal_percentage, expected, actual, percentage)

// ------------------------------------------------------------------------------------------------
// Colors
// ------------------------------------------------------------------------------------------------

inline bool equal_rgb(Bifrost::Math::RGB lhs, Bifrost::Math::RGB rhs) {
    return Bifrost::Math::almost_equal(lhs.r, rhs.r)
        && Bifrost::Math::almost_equal(lhs.g, rhs.g)
        && Bifrost::Math::almost_equal(lhs.b, rhs.b);
}
#define EXPECT_RGB_EQ(expected, actual) EXPECT_PRED2(equal_rgb, expected, actual)

inline bool equal_rgb_eps(Bifrost::Math::RGB lhs, Bifrost::Math::RGB rhs, float eps) {
    return abs(lhs.r - rhs.r) < eps && abs(lhs.g - rhs.g) < eps && abs(lhs.b - rhs.b) < eps;
}
#define EXPECT_RGB_EQ_EPS(expected, actual, eps) EXPECT_PRED3(equal_rgb_eps, expected, actual, eps)

inline bool equal_rgba(Bifrost::Math::RGBA lhs, Bifrost::Math::RGBA rhs) {
    return Bifrost::Math::almost_equal(lhs.r, rhs.r)
        && Bifrost::Math::almost_equal(lhs.g, rhs.g)
        && Bifrost::Math::almost_equal(lhs.b, rhs.b)
        && Bifrost::Math::almost_equal(lhs.a, rhs.a);
}
#define EXPECT_RGBA_EQ(expected, actual) EXPECT_PRED2(equal_rgba, expected, actual)

// ------------------------------------------------------------------------------------------------
// Vectors
// ------------------------------------------------------------------------------------------------

inline bool equal_normal_eps(Bifrost::Math::Vector3f lhs, Bifrost::Math::Vector3f rhs, double epsilon) {
    Bifrost::Math::Vector3d delta = { double(lhs.x) - double(rhs.x), double(lhs.y) - double(rhs.y), double(lhs.z) - double(rhs.z) };
    double length_squared = delta.x * delta.x + delta.y * delta.y + delta.z * delta.z;
    return length_squared < epsilon * epsilon;
}

#define EXPECT_NORMAL_EQ(expected, actual, epsilon) EXPECT_PRED3(equal_normal_eps, expected, actual, epsilon)

inline bool equal_Vector2f(Bifrost::Math::Vector2f lhs, Bifrost::Math::Vector2f rhs) {
    return Bifrost::Math::almost_equal(lhs.x, rhs.x) && Bifrost::Math::almost_equal(lhs.y, rhs.y);
}
#define EXPECT_VECTOR2F_EQ(expected, actual) EXPECT_PRED2(equal_Vector2f, expected, actual)

template <typename T>
inline bool equal_quaternion(Bifrost::Math::Quaternion<T> expected, Bifrost::Math::Quaternion<T> actual) {
    return Bifrost::Math::almost_equal(expected, actual);
}
#define EXPECT_QUAT_F_EQ(expected, actual) EXPECT_PRED2(equal_quaternion<float>, expected, actual)

// ------------------------------------------------------------------------------------------------
// Misc
// ------------------------------------------------------------------------------------------------

inline bool invalid_AABB(Bifrost::Math::AABB v) {
    return v.maximum.x < v.minimum.x || v.maximum.y < v.minimum.y || v.maximum.z < v.minimum.z;
}
#define EXPECT_INVALID_AABB(val) EXPECT_PRED1(invalid_AABB, val)

#endif // _BIFROST_TESTS_EXPECTS_H_
