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

    EXPECT_EQ(Vector2f::zero(), distribution.sample_discretized(Vector2f(0.0f, 0.0f)).index);
    EXPECT_FLOAT_EQ(1.0f, distribution.sample_discretized(Vector2f(0.0f, 0.0f)).PDF);
    EXPECT_EQ(Vector2f::zero(), distribution.sample_discretized(Vector2f(nearly_one, 0.0f)).index);
    EXPECT_FLOAT_EQ(1.0f, distribution.sample_discretized(Vector2f(nearly_one, 0.0f)).PDF);
    EXPECT_EQ(Vector2f::zero(), distribution.sample_discretized(Vector2f(0.0f, nearly_one)).index);
    EXPECT_FLOAT_EQ(1.0f, distribution.sample_discretized(Vector2f(0.0f, nearly_one)).PDF);
    EXPECT_EQ(Vector2f::zero(), distribution.sample_discretized(Vector2f(nearly_one, nearly_one)).index);
    EXPECT_FLOAT_EQ(1.0f, distribution.sample_discretized(Vector2f(nearly_one, nearly_one)).PDF);
    EXPECT_EQ(Vector2f::zero(), distribution.sample_discretized(Vector2f(0.5f, 0.5f)).index);
    EXPECT_FLOAT_EQ(1.0f, distribution.sample_discretized(Vector2f(0.5f, 0.5f)).PDF);
}

GTEST_TEST(Math_Distribution2D, constant_function) {
    float vs[] = { 1, 1, 1, 1 };

    Distribution2D<float> distribution = Distribution2D<float>(vs, 2, 2);

    // Sample the corners.
    EXPECT_EQ(Vector2f(0, 0), distribution.sample_discretized(Vector2f(0.0f, 0.0f)).index);
    EXPECT_FLOAT_EQ(0.25f, distribution.sample_discretized(Vector2f(0.0f, 0.0f)).PDF);
    EXPECT_EQ(Vector2f(1, 0), distribution.sample_discretized(Vector2f(nearly_one, 0.0f)).index);
    EXPECT_FLOAT_EQ(0.25f, distribution.sample_discretized(Vector2f(nearly_one, 0.0f)).PDF);
    EXPECT_EQ(Vector2f(0, 1), distribution.sample_discretized(Vector2f(0.0f, nearly_one)).index);
    EXPECT_FLOAT_EQ(0.25f, distribution.sample_discretized(Vector2f(0.0f, nearly_one)).PDF);
    EXPECT_EQ(Vector2f(1, 1), distribution.sample_discretized(Vector2f(nearly_one, nearly_one)).index);
    EXPECT_FLOAT_EQ(0.25f, distribution.sample_discretized(Vector2f(nearly_one, nearly_one)).PDF);

    // Sample indices at centers.
    EXPECT_EQ(Vector2f(0, 0), distribution.sample_discretized(Vector2f(0.25f, 0.25f)).index);
    EXPECT_FLOAT_EQ(0.25f, distribution.sample_discretized(Vector2f(0.25f, 0.25f)).PDF);
    EXPECT_EQ(Vector2f(1, 0), distribution.sample_discretized(Vector2f(0.75f, 0.25f)).index);
    EXPECT_FLOAT_EQ(0.25f, distribution.sample_discretized(Vector2f(0.75f, 0.25f)).PDF);
    EXPECT_EQ(Vector2f(0, 1), distribution.sample_discretized(Vector2f(0.25f, 0.75f)).index);
    EXPECT_FLOAT_EQ(0.25f, distribution.sample_discretized(Vector2f(0.25f, 0.75f)).PDF);
    EXPECT_EQ(Vector2f(1, 1), distribution.sample_discretized(Vector2f(0.75f, 0.75f)).index);
    EXPECT_FLOAT_EQ(0.25f, distribution.sample_discretized(Vector2f(0.75f, 0.75f)).PDF);

    // Sample floats at centers.
    EXPECT_EQ(Vector2f(0.5f, 0.5f), distribution.sample_continuous(Vector2f(0.25f, 0.25f)).index);
    EXPECT_FLOAT_EQ(0.25f, distribution.sample_discretized(Vector2f(0.25f, 0.25f)).PDF);
    EXPECT_EQ(Vector2f(1.5f, 0.5f), distribution.sample_continuous(Vector2f(0.75f, 0.25f)).index);
    EXPECT_FLOAT_EQ(0.25f, distribution.sample_discretized(Vector2f(0.75f, 0.25f)).PDF);
    EXPECT_EQ(Vector2f(0.5f, 1.5f), distribution.sample_continuous(Vector2f(0.25f, 0.75f)).index);
    EXPECT_FLOAT_EQ(0.25f, distribution.sample_discretized(Vector2f(0.25f, 0.75f)).PDF);
    EXPECT_EQ(Vector2f(1.5f, 1.5f), distribution.sample_continuous(Vector2f(0.75f, 0.75f)).index);
    EXPECT_FLOAT_EQ(0.25f, distribution.sample_discretized(Vector2f(0.75f, 0.75f)).PDF);
}

GTEST_TEST(Math_Distribution2D, interesting_function) {
    float vs[] = { 0, 5, 0,   // Weight: 0.2
                   2, 3, 0,   // Weight: 0.2 
                   4, 5, 6 }; // Weight: 0.6

    Distribution2D<float> distribution = Distribution2D<float>(vs, 3, 3);

    // Sample top left of first row.
    Vector2f random = Vector2f(0.0f, 0.0f);
    EXPECT_VECTOR2F_EQ(Vector2f(1.0f, 0.0f), distribution.sample_continuous(random).index);
    EXPECT_FLOAT_EQ(0.2f, distribution.sample_continuous(random).PDF);
    
    // Sample center of first row.
    random = Vector2f(0.5f, 0.1f);
    EXPECT_VECTOR2F_EQ(Vector2f(1.5f, 0.5f), distribution.sample_continuous(random).index);
    EXPECT_FLOAT_EQ(0.2f, distribution.sample_continuous(random).PDF);
    
    // Sample bottom right of first row.
    random = Vector2f(nearly_one, previous_float(0.2f));
    EXPECT_VECTOR2F_EQ(Vector2f(2.0f, 1.0f), distribution.sample_continuous(random).index);
    EXPECT_FLOAT_EQ(0.2f, distribution.sample_continuous(random).PDF);

    // Sample center value.
    random = Vector2f(0.5f, 0.3f);
    EXPECT_VECTOR2F_EQ(Vector2f(1.166666666667f, 1.5f), distribution.sample_continuous(random).index);
    EXPECT_FLOAT_EQ(0.12f, distribution.sample_continuous(random).PDF);

    // Sample lowest right corner.
    random = Vector2f(nearly_one, nearly_one);
    EXPECT_VECTOR2F_EQ(Vector2f(3.0f, 3.0f), distribution.sample_continuous(random).index);
    EXPECT_FLOAT_EQ(0.24f, distribution.sample_continuous(random).PDF);
}

} // NS Math
} // NS Cogwheel

#endif // _COGWHEEL_MATH_DISTRIBUTION2D_TEST_H_