// Test Cogwheel Quaternions.
// ---------------------------------------------------------------------------
// Copyright (C) 2015, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _COGWHEEL_MATH_QUATERNION_TEST_H_
#define _COGWHEEL_MATH_QUATERNION_TEST_H_

#include <Math/Quaternion.h>
#include <Math/Conversions.h>

#include <gtest/gtest.h>

namespace Cogwheel {
namespace Math {

class Math_Quaternion : public ::testing::Test {
protected:
    // Redefine comparison methods as gtest's EXPECT_PRED and argument overloading doesn't play well with each other.
    static bool compare_quaternion(Quaternionf lhs, Quaternionf rhs, unsigned short max_ulps) {
        return almost_equal(lhs, rhs, max_ulps);
    }
    static bool compare_vector(Vector3f lhs, Vector3f rhs, unsigned short max_ulps) {
        return almost_equal(lhs, rhs, max_ulps);
    }
};

TEST_F(Math_Quaternion, axis_helpers) {
    Quaternionf quat = Quaternionf::from_angle_axis(25, Vector3f::up());

    unsigned short max_error = 10;
    EXPECT_PRED3(compare_vector, quat.forward(), quat * Vector3f::forward(), max_error);
    EXPECT_PRED3(compare_vector, quat.up(), quat * Vector3f::up(), max_error);
    EXPECT_PRED3(compare_vector, quat.right(), quat * Vector3f::right(), max_error);
}

TEST_F(Math_Quaternion, matrix_representation) {
    // Test N number of quaternions.
    const int N = 10;
    for (int i = 0; i < N; ++i) {
        // Construct a 'random' quaternion.
        float angle = (i * 360.0f) / N;
        Vector3f axis = i % 2 ? Vector3f::up() : Vector3f::zero();
        axis += i % 4 ? Vector3f::forward() : Vector3f::zero();
        axis += i % 8 ? Vector3f::right() : Vector3f::zero();
        
        Quaternionf q0 = Quaternionf::from_angle_axis(angle, normalize(axis));

        Matrix3x3f m = to_matrix3x3(q0);

        Quaternionf q1 = to_quaternion(m);

        // Compare transformations of vectors.
        for (Vector3f v : { Vector3f::forward(), Vector3f::right(), Vector3f::up() }) {
            EXPECT_PRED3(compare_vector, q0 * v, m * v, 20);
            EXPECT_PRED3(compare_vector, q1 * v, m * v, 30); // Apparently toQuaternion has bigger precision issues than toMatrix3x3.
        }
    }
}

} // NS Math
} // NS Cogwheel

#endif // _COGWHEEL_MATH_QUATERNION_TEST_H_