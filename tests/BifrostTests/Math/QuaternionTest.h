// Test Bifrost Quaternions.
// ---------------------------------------------------------------------------
// Copyright (C) 2015, Bifrost. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _BIFROST_MATH_QUATERNION_TEST_H_
#define _BIFROST_MATH_QUATERNION_TEST_H_

#include <Bifrost/Math/Quaternion.h>
#include <Bifrost/Math/Conversions.h>
#include <Bifrost/Math/Utils.h>

#include <gtest/gtest.h>

namespace Bifrost {
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
    Quaternionf quat = Quaternionf::from_angle_axis(degrees_to_radians(25.0f), Vector3f::up());

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
        Vector3f axis = i % 2 ? Vector3f::up() : -Vector3f::up();
        axis += i % 4 ? Vector3f::forward() : Vector3f::zero();
        axis += i % 8 ? Vector3f::right() : Vector3f::zero();
        
        Quaternionf q0 = Quaternionf::from_angle_axis(degrees_to_radians(angle), normalize(axis));

        Matrix3x3f m = to_matrix3x3(q0);

        Quaternionf q1 = to_quaternion(m);

        // Compare transformations of vectors.
        for (Vector3f v : { Vector3f::forward(), Vector3f::right(), Vector3f::up() }) {
            EXPECT_PRED3(compare_vector, q0 * v, m * v, 20);
            EXPECT_PRED3(compare_vector, q1 * v, m * v, 30);
        }
    }
}

TEST_F(Math_Quaternion, look_in) {
    // The forward of a quaternion 'looking along' a direction should equal that direction.
    for (int y = -1; y < 2; ++y) {
        for (int i = 0; i < 8; ++i) {
            float phi = i * 2.0f * PI<float>() / 8.0f;
            Vector3d direction = normalize(Vector3d(cos(phi), float(y), sin(phi)));
            Quaterniond qd = Quaterniond::look_in(direction, Vector3d::up());
            Quaternionf q = Quaterniond(qd);

            // Quaternion generated from look_in should point in the same general direction as the used direction.
            EXPECT_GT(dot(qd.forward(), direction), 0.99999999999);
            // Local 'up' should not point downwards.
            EXPECT_LT(0.0f, q.up().y);
            // Local 'right' should be located in the xz-plane.
            EXPECT_LT(abs(q.right().y), 0.00000005f);
        }
    }
}

} // NS Math
} // NS Bifrost

#endif // _BIFROST_MATH_QUATERNION_TEST_H_
