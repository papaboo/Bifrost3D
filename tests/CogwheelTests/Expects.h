// Test Cogwheel textures.
// ---------------------------------------------------------------------------
// Copyright (C) 2015-2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#include <Cogwheel/Math/AABB.h>
#include <Cogwheel/Math/Color.h>
#include <Cogwheel/Math/Quaternion.h>

#include <gtest/gtest.h>

#ifndef _COGWHEEL_TESTS_EXPECTS_H_
#define _COGWHEEL_TESTS_EXPECTS_H_

static bool equal_rgb(Cogwheel::Math::RGB lhs, Cogwheel::Math::RGB rhs) {
    return Cogwheel::Math::almost_equal(lhs.r, rhs.r)
        && Cogwheel::Math::almost_equal(lhs.g, rhs.g)
        && Cogwheel::Math::almost_equal(lhs.b, rhs.b);
}
#define EXPECT_RGB_EQ(expected, actual) EXPECT_PRED2(equal_rgb, expected, actual)

static bool equal_rgb_eps(Cogwheel::Math::RGB lhs, Cogwheel::Math::RGB rhs, float eps) {
    return abs(lhs.r - rhs.r) < eps && abs(lhs.g - rhs.g) < eps && abs(lhs.b - rhs.b) < eps;
}
#define EXPECT_RGB_EQ_EPS(expected, actual, eps) EXPECT_PRED3(equal_rgb_eps, expected, actual, eps)

static bool equal_rgba(Cogwheel::Math::RGBA lhs, Cogwheel::Math::RGBA rhs) {
    return Cogwheel::Math::almost_equal(lhs.r, rhs.r)
        && Cogwheel::Math::almost_equal(lhs.g, rhs.g)
        && Cogwheel::Math::almost_equal(lhs.b, rhs.b)
        && Cogwheel::Math::almost_equal(lhs.a, rhs.a);
}
#define EXPECT_RGBA_EQ(expected, actual) EXPECT_PRED2(equal_rgba, expected, actual)

static bool invalid_AABB(Cogwheel::Math::AABB v) {
    return v.maximum.x < v.minimum.x || v.maximum.y < v.minimum.y || v.maximum.z < v.minimum.z;
}
#define EXPECT_INVALID_AABB(val) EXPECT_PRED1(invalid_AABB, val)

static bool equal_normal(Cogwheel::Math::Vector3f lhs, Cogwheel::Math::Vector3f rhs) {
    return dot(lhs, rhs) > 0.99999f;
}
#define EXPECT_NORMAL_EQ(expected, actual) EXPECT_PRED2(equal_normal, expected, actual)

static bool equal_Vector2f(Cogwheel::Math::Vector2f lhs, Cogwheel::Math::Vector2f rhs) {
    return Cogwheel::Math::almost_equal(lhs.x, rhs.x) && Cogwheel::Math::almost_equal(lhs.y, rhs.y);
}
#define EXPECT_VECTOR2F_EQ(expected, actual) EXPECT_PRED2(equal_Vector2f, expected, actual)

static bool float_in_range(float min, float max, float actual) {
    return min <= actual && actual <= max;
}
#define EXPECT_FLOAT_IN_RANGE(min, max, actual) EXPECT_PRED3(float_in_range, min, max, actual)

template <typename T>
static bool equal_quaternion(Cogwheel::Math::Quaternion<T> expected, Cogwheel::Math::Quaternion<T> actual) {
    return Cogwheel::Math::almost_equal(expected, actual);
}
#define EXPECT_QUAT_F_EQ(expected, actual) EXPECT_PRED2(equal_quaternion<float>, expected, actual)

#endif // _COGWHEEL_TESTS_EXPECTS_H_