// Test Bifrost Matrix.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _BIFROST_MATH_MATRIX_TEST_H_
#define _BIFROST_MATH_MATRIX_TEST_H_

#include <Bifrost/Math/Matrix.h>

#include <gtest/gtest.h>

namespace Bifrost {
namespace Math {

class Math_Matrix : public ::testing::Test {
protected:
    // Redefine comparison methods as gtest's EXPECT_PRED and argument overloading doesn't play well with each other.
    static bool compare_matrix2x2f(Matrix2x2f lhs, Matrix2x2f rhs, unsigned short max_ulps) {
        return almost_equal(lhs, rhs, max_ulps);
    }
    static bool compare_matrix3x3f(Matrix3x3f lhs, Matrix3x3f rhs, unsigned short max_ulps) {
        return almost_equal(lhs, rhs, max_ulps);
    }
    static bool compare_matrix4x4f(Matrix4x4f lhs, Matrix4x4f rhs, unsigned short max_ulps) {
        return almost_equal(lhs, rhs, max_ulps);
    }
};

TEST_F(Math_Matrix, invert2x2) {
    // NOTE strangely enough the matrix {1, 4, 5, 8} is experiencing incredible numerical issues and won't even parse a test with a max error of 1000.
    Matrix2x2f mat = { 2, 4,
                       6, 8 };

    Matrix2x2f mat_inverse = invert(mat);

    unsigned short max_error = 10;
    EXPECT_PRED3(compare_matrix2x2f, mat * mat_inverse, Matrix2x2f::identity(), max_error);
    EXPECT_PRED3(compare_matrix2x2f, mat_inverse * mat, Matrix2x2f::identity(), max_error);
}

TEST_F(Math_Matrix, invert3x3) {
    Matrix3x3f mat = { 2, 4, 2,
                       6, 8, 5,
                       4, 8, 5 };

    Matrix3x3f mat_inverse = invert(mat);

    unsigned short max_error = 10;
    EXPECT_PRED3(compare_matrix3x3f, mat * mat_inverse, Matrix3x3f::identity(), max_error);
    EXPECT_PRED3(compare_matrix3x3f, mat_inverse * mat, Matrix3x3f::identity(), max_error);
}

TEST_F(Math_Matrix, invert4x4) {
    Matrix4x4f mat = { 2, 4, 0, 2,
                       0, 0, 1, 0,
                       6, 8, 0, 5,
                       4, 8, 0, 5 };

    Matrix4x4f mat_inverse = invert(mat);

    unsigned short max_error = 10;
    EXPECT_PRED3(compare_matrix4x4f, mat * mat_inverse, Matrix4x4f::identity(), max_error);
    EXPECT_PRED3(compare_matrix4x4f, mat_inverse * mat, Matrix4x4f::identity(), max_error);
}

} // NS Math
} // NS Bifrost

#endif // _BIFROST_MATH_MATRIX_TEST_H_
