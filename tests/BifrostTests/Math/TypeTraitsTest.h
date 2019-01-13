// Test Bifrost math type traits.
// ---------------------------------------------------------------------------
// Copyright (C) 2018, Bifrost. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _BIFROST_MATH_TYPE_TRAITS_TEST_H_
#define _BIFROST_MATH_TYPE_TRAITS_TEST_H_

#include <Bifrost/Math/Color.h>
#include <Bifrost/Math/Matrix.h>
#include <Bifrost/Math/Plane.h>
#include <Bifrost/Math/Quaternion.h>
#include <Bifrost/Math/Ray.h>
#include <Bifrost/Math/Rect.h>
#include <Bifrost/Math/Transform.h>
#include <Bifrost/Math/Vector.h>

#include <gtest/gtest.h>

namespace Bifrost {
namespace Math {

GTEST_TEST(Math_TypeTraits, RGB) {
    EXPECT_TRUE(std::is_trivial<RGB>::value);
    EXPECT_TRUE(std::is_standard_layout<RGB>::value);
    EXPECT_TRUE(std::is_pod<RGB>::value);
}

GTEST_TEST(Math_TypeTraits, RGBA) {
    EXPECT_TRUE(std::is_trivial<RGBA>::value);
    EXPECT_TRUE(std::is_standard_layout<RGBA>::value);
    EXPECT_TRUE(std::is_pod<RGBA>::value);
}

GTEST_TEST(Math_TypeTraits, Matrix3x3f) {
    EXPECT_TRUE(std::is_trivial<Matrix3x3f>::value);
    EXPECT_TRUE(std::is_standard_layout<Matrix3x3f>::value);
    EXPECT_TRUE(std::is_pod<Matrix3x3f>::value);
}

GTEST_TEST(Math_TypeTraits, Matrix4x4f) {
    EXPECT_TRUE(std::is_trivial<Matrix4x4f>::value);
    EXPECT_TRUE(std::is_standard_layout<Matrix4x4f>::value);
    EXPECT_TRUE(std::is_pod<Matrix4x4f>::value);
}

GTEST_TEST(Math_TypeTraits, Plane) {
    EXPECT_TRUE(std::is_trivial<Plane>::value);
    EXPECT_TRUE(std::is_standard_layout<Plane>::value);
    EXPECT_TRUE(std::is_pod<Plane>::value);
}

GTEST_TEST(Math_TypeTraits, Quaternionf) {
    EXPECT_TRUE(std::is_trivial<Quaternionf>::value);
    EXPECT_TRUE(std::is_standard_layout<Quaternionf>::value);
    EXPECT_TRUE(std::is_pod<Quaternionf>::value);
}

GTEST_TEST(Math_TypeTraits, Ray) {
    EXPECT_TRUE(std::is_trivial<Ray>::value);
    EXPECT_TRUE(std::is_standard_layout<Ray>::value);
    EXPECT_TRUE(std::is_pod<Ray>::value);
}

GTEST_TEST(Math_TypeTraits, Rectf) {
    EXPECT_TRUE(std::is_trivial<Rectf>::value);
    EXPECT_TRUE(std::is_standard_layout<Rectf>::value);
    EXPECT_TRUE(std::is_pod<Rectf>::value);
}

GTEST_TEST(Math_TypeTraits, Transform) {
    EXPECT_TRUE(std::is_trivial<Transform>::value);
    EXPECT_TRUE(std::is_standard_layout<Transform>::value);
    EXPECT_TRUE(std::is_pod<Transform>::value);
}

GTEST_TEST(Math_TypeTraits, Vector2f) {
    EXPECT_TRUE(std::is_trivial<Vector2f>::value);
    EXPECT_TRUE(std::is_standard_layout<Vector2f>::value);
    EXPECT_TRUE(std::is_pod<Vector2f>::value);
}

GTEST_TEST(Math_TypeTraits, Vector3f) {
    EXPECT_TRUE(std::is_trivial<Vector3f>::value);
    EXPECT_TRUE(std::is_standard_layout<Vector3f>::value);
    EXPECT_TRUE(std::is_pod<Vector3f>::value);
}

GTEST_TEST(Math_TypeTraits, Vector4f) {
    EXPECT_TRUE(std::is_trivial<Vector4f>::value);
    EXPECT_TRUE(std::is_standard_layout<Vector4f>::value);
    EXPECT_TRUE(std::is_pod<Vector4f>::value);
}

} // NS Math
} // NS Bifrost

#endif // _BIFROST_MATH_TYPE_TRAITS_TEST_H_
