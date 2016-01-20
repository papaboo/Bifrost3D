// Test Cogwheel Transform.
// ---------------------------------------------------------------------------
// Copyright (C) 2015, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _COGWHEEL_MATH_TRANSFORM_TEST_H_
#define _COGWHEEL_MATH_TRANSFORM_TEST_H_

#include <Math/Quaternion.h>
#include <Math/Conversions.h>

#include <gtest/gtest.h>

namespace Cogwheel {
namespace Math {

class Math_Transform : public ::testing::Test {
protected:

    // Redefine comparison methods as gtest's EXPECT_PRED and argument overloading doesn't play well with each other.
    static bool compare_transform(Transform lhs, Transform rhs) {
        return almost_equal(lhs.translation, rhs.translation, 10)
            && almost_equal(lhs.rotation, rhs.rotation, 10)
            && almost_equal(lhs.scale, rhs.scale, 10);
    }
    static bool compare_quaternion(Quaternionf lhs, Quaternionf rhs, unsigned short maxUlps) {
        return almost_equal(lhs, rhs, maxUlps);
    }
    static bool compare_vector(Vector3f lhs, Vector3f rhs, unsigned short maxUlps) {
        return almost_equal(lhs, rhs, maxUlps);
    }
};

TEST_F(Math_Transform, applying_translation) {
    Transform t = Transform(Vector3f(3, -4, 1));

    EXPECT_PRED2(compare_transform, t.apply(invert(t)), Transform::identity());

    EXPECT_PRED3(compare_vector, t.apply(Vector3f::zero()), t.translation, 10);

    EXPECT_PRED3(compare_vector, t.apply(-t.translation), Vector3f::zero(), 10);

    Vector3f v = Vector3f(7, 3, -1);
    EXPECT_PRED3(compare_vector, t.apply(v), Vector3f(10, -1, 0), 10);
}

TEST_F(Math_Transform, apply_rotation) {
    Transform t = Transform(Vector3f::zero(), Quaternionf::from_angle_axis(45.0f, Vector3f::up()));

    EXPECT_PRED3(compare_vector, t.apply(Vector3f::zero()), Vector3f::zero(), 10);

    EXPECT_PRED2(compare_transform, t.apply(invert(t)), Transform::identity());

    // Should rotate forward by 45 degress around up.
    EXPECT_PRED3(compare_vector, t.apply(Vector3f::forward()), Vector3f(sqrt(0.5f), 0.0f, sqrt(0.5f)), 10);
}

TEST_F(Math_Transform, apply_scale) {
    Transform t = Transform(Vector3f::zero(), Quaternionf::identity(), 0.5f);

    EXPECT_PRED2(compare_transform, t.apply(invert(t)), Transform::identity());

    EXPECT_PRED3(compare_vector, t.apply(Vector3f::zero()), Vector3f::zero(), 10);

    EXPECT_PRED3(compare_vector, t.apply(Vector3f::one()), Vector3f::one() * 0.5f, 10);
}

TEST_F(Math_Transform, matrix_representation) {
    Transform t = Transform(Vector3f(3, -4, 1), normalize(Quaternionf(4,2,-7, 3)), 0.75f);

    Matrix4x3f m4x3 = to_matrix4x3(t);
    Matrix4x4f m4x4 = to_matrix4x4(t);

    // Compare transformations of points.
    for (Vector3f p : { Vector3f::forward(), Vector3f::right(), Vector3f::up() }) {
        Vector3f transformed_by_quat = t * p;
        
        Vector4f _p = m4x4 * Vector4f(p.x, p.y, p.z, 1);
        Vector3f transformed_by_matrix = Vector3f(_p.x, _p.y, _p.z) / _p.w;
        EXPECT_PRED3(compare_vector, transformed_by_quat, transformed_by_matrix, 10);

        transformed_by_matrix = m4x3 * Vector4f(p.x, p.y, p.z, 1);
        EXPECT_PRED3(compare_vector, transformed_by_quat, transformed_by_matrix, 10);
    }
}

} // NS Math
} // NS Cogwheel

#endif // _COGWHEEL_MATH_TRANSFORM_TEST_H_