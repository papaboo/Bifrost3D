// Test Cogwheel Distribution2D.
// ---------------------------------------------------------------------------
// Copyright (C) 2015, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _COGWHEEL_MATH_DISTRIBUTION2D_TEST_H_
#define _COGWHEEL_MATH_DISTRIBUTION2D_TEST_H_

#include <Cogwheel/Math/Distribution2D.h>

#include <gtest/gtest.h>

namespace Cogwheel {
namespace Math {

GTEST_TEST(Math_Distribution2D, single_value_function) {
    double v = 1.0;

    Distribution2D<float> distribution = Distribution2D<float>(&v, 1, 1);

    EXPECT_EQ(Vector2i::zero(), distribution.samplei(Vector2f(0.0f, 0.0f)));
    EXPECT_EQ(Vector2i::zero(), distribution.samplei(Vector2f(1.0f, 0.0f)));
    EXPECT_EQ(Vector2i::zero(), distribution.samplei(Vector2f(0.0f, 1.0f)));
    EXPECT_EQ(Vector2i::zero(), distribution.samplei(Vector2f(1.0f, 1.0f)));
    EXPECT_EQ(Vector2i::zero(), distribution.samplei(Vector2f(0.5f, 0.5f)));
}

GTEST_TEST(Math_Distribution2D, constant_function) {
    float vs[] = { 1, 1, 1, 1 };

    Distribution2D<float> distribution = Distribution2D<float>(vs, 2, 2);

    // Sample the corners.
    EXPECT_EQ(Vector2i(0, 0), distribution.samplei(Vector2f(0.0f, 0.0f)));
    EXPECT_EQ(Vector2i(1, 0), distribution.samplei(Vector2f(1.0f, 0.0f)));
    EXPECT_EQ(Vector2i(0, 1), distribution.samplei(Vector2f(0.0f, 1.0f)));
    EXPECT_EQ(Vector2i(1, 1), distribution.samplei(Vector2f(1.0f, 1.0f)));

    // Sample indices at centers.
    EXPECT_EQ(Vector2i(0, 0), distribution.samplei(Vector2f(0.25f, 0.25f)));
    EXPECT_EQ(Vector2i(1, 0), distribution.samplei(Vector2f(0.75f, 0.25f)));
    EXPECT_EQ(Vector2i(0, 1), distribution.samplei(Vector2f(0.25f, 0.75f)));
    EXPECT_EQ(Vector2i(1, 1), distribution.samplei(Vector2f(0.75f, 0.75f)));

    // Sample floats at centers.
    EXPECT_EQ(Vector2f(0.5f, 0.5f), distribution.samplef(Vector2f(0.25f, 0.25f)));
    EXPECT_EQ(Vector2f(1.5f, 0.5f), distribution.samplef(Vector2f(0.75f, 0.25f)));
    EXPECT_EQ(Vector2f(0.5f, 1.5f), distribution.samplef(Vector2f(0.25f, 0.75f)));
    EXPECT_EQ(Vector2f(1.5f, 1.5f), distribution.samplef(Vector2f(0.75f, 0.75f)));
}

} // NS Math
} // NS Cogwheel

#endif // _COGWHEEL_MATH_DISTRIBUTION2D_TEST_H_