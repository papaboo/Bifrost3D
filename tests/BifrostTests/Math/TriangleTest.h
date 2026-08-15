// Test Bifrost triangle.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _BIFROST_MATH_TRIANGLE_TEST_H_
#define _BIFROST_MATH_TRIANGLE_TEST_H_

#include <Bifrost/Math/Triangle.h>

#include <Expects.h>

namespace Bifrost::Math {

GTEST_TEST(Math_Triangle, normal_direction_regression_test) {
    Vector3f v0 = { 1, 1, 0 };
    Vector3f v1 = { -1, 1, 0 };
    Vector3f v2 = { 1, -1, 0 };
    Trianglef triangle = { v0, v1, v2 };

    Vector3f expected_normal = { 0, 0, 1 };
    EXPECT_VECTOR3F_EQ(expected_normal, triangle.get_normal());
}

GTEST_TEST(Math_Triangle, up_direction_is_not_normalized) {
    Vector3f v0 = { 1, 1, 0 };
    Vector3f v1 = { -1, 1, 0 };
    Vector3f v2 = { 1, -1, 0 };
    Trianglef triangle = { v0, v1, v2 };

    Vector3f up = triangle.get_up();
    EXPECT_NE(1.0f, magnitude(up));
}

GTEST_TEST(Math_Triangle, surface_area) {
    Vector3f v0 = { 1, 1, 0 };
    Vector3f v1 = { -1, 1, 0 };
    Vector3f v2 = { 1, -1, 0 };
    Trianglef triangle = { v0, v1, v2 };

    float expected_surface_area = 2;
    EXPECT_FLOAT_EQ(expected_surface_area, triangle.get_surface_area());
}

GTEST_TEST(Math_Triangle, minimum_maximum) {
    // Distribute minimum and maximum coordinates across all vertices
    Vector3f v0 = { 1, 6, 8 };
    Vector3f v1 = { 2, 4, 9 };
    Vector3f v2 = { 3, 5, 7 };
    Trianglef triangle = { v0, v1, v2 };

    Vector3f expected_minimum = { 1, 4, 7 };
    Vector3f expected_maximum = { 3, 6, 9 };
    EXPECT_VECTOR3F_EQ(expected_minimum, triangle.get_min());
    EXPECT_VECTOR3F_EQ(expected_maximum, triangle.get_max());
}

} // NS Bifrost::Math

#endif // _BIFROST_MATH_TRIANGLE_TEST_H_
