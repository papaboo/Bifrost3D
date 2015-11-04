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

class Math_TransformTest : public ::testing::Test {
protected:

    // Redefine comparison methods as gtest's EXPECT_PRED and argument overloading doesn't play well with each other.
    static bool compareTransform(Transform lhs, Transform rhs) {
        return almostEqual(lhs.mTranslation, rhs.mTranslation, 10)
            && almostEqual(lhs.mRotation, rhs.mRotation, 10)
            && almostEqual(lhs.mScale, rhs.mScale, 10);
    }
    template <typename Row, typename Column>
    static bool compareMatrix(Matrix<Row, Column> lhs, Matrix<Row, Column> rhs, unsigned short maxUlps) {
        return almostEqual(lhs, rhs, maxUlps);
    }
    static bool compareQuaternion(Quaternionf lhs, Quaternionf rhs, unsigned short maxUlps) {
        return almostEqual(lhs, rhs, maxUlps);
    }
    static bool compareVector(Vector3f lhs, Vector3f rhs, unsigned short maxUlps) {
        return almostEqual(lhs, rhs, maxUlps);
    }
};

TEST_F(Math_TransformTest, ApplyTransform) {
    Transform t = Transform(Vector3f(3, -4, 1));

    EXPECT_PRED2(compareTransform, t.apply(inverse(t)), Transform::identity());

    EXPECT_PRED3(compareVector, t.apply(Vector3f::zero()), t.mTranslation, 10);

    EXPECT_PRED3(compareVector, t.apply(-t.mTranslation), Vector3f::zero(), 10);

    Vector3f v = Vector3f(7, 3, -1);
    EXPECT_PRED3(compareVector, t.apply(v), Vector3f(10, -1, 0), 10);
}

TEST_F(Math_TransformTest, ApplyRotation) {
    Transform t = Transform(Vector3f::zero(), Quaternionf::fromAngleAxis(45.0f, Vector3f::up()));

    EXPECT_PRED3(compareVector, t.apply(Vector3f::zero()), Vector3f::zero(), 10);

    EXPECT_PRED2(compareTransform, t.apply(inverse(t)), Transform::identity());

    // Should rotate forward by 45 degress around up.
    EXPECT_PRED3(compareVector, t.apply(Vector3f::forward()), Vector3f(sqrt(0.5f), 0.0f, sqrt(0.5f)), 10);
}

TEST_F(Math_TransformTest, ApplyScale) {
    Transform t = Transform(Vector3f::zero(), Quaternionf::identity(), 0.5f);

    EXPECT_PRED2(compareTransform, t.apply(inverse(t)), Transform::identity());

    EXPECT_PRED3(compareVector, t.apply(Vector3f::zero()), Vector3f::zero(), 10);

    EXPECT_PRED3(compareVector, t.apply(Vector3f::one()), Vector3f::one() * 0.5f, 10);
}

TEST_F(Math_TransformTest, MatrixRepresentation) {
    Transform t = Transform(Vector3f(3, -4, 1), normalize(Quaternionf(4,2,-7, 3)), 0.75f);

    Matrix4x3f m4x3 = toMatrix4x3(t);
    Matrix4x4f m4x4 = toMatrix4x4(t);

    // Compare transformations of points. TODO Use an RNG to create more random vectors
    for (Vector3f p : { Vector3f::forward(), Vector3f::right(), Vector3f::up() }) {
        Vector3f transformedByQuat = t * p;
        
        Vector4f _p = m4x4 * Vector4f(p.x, p.y, p.z, 1);
        Vector3f transformedByMatrix = Vector3f(_p.x, _p.y, _p.z) / _p.w;
        EXPECT_PRED3(compareVector, transformedByQuat, transformedByMatrix, 10);

        transformedByMatrix = m4x3 * Vector4f(p.x, p.y, p.z, 1);
        EXPECT_PRED3(compareVector, transformedByQuat, transformedByMatrix, 10);
    }
}

} // NS Math
} // NS Cogwheel

#endif // _COGWHEEL_MATH_TRANSFORM_TEST_H_