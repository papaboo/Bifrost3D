// Test Cogwheel Statistics.
// ---------------------------------------------------------------------------
// Copyright (C) 2015, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _COGWHEEL_MATH_STATISTICS_TEST_H_
#define _COGWHEEL_MATH_STATISTICS_TEST_H_

#include <Cogwheel/Math/Statistics.h>

#include <gtest/gtest.h>

namespace Cogwheel {
namespace Math {

GTEST_TEST(Math_Statistics, accumulate) {
    int vs[] = { 1, 2, 3, 4, 5 };
    Statistics<float> stats = Statistics<float>(vs, vs + 5);
    EXPECT_EQ(5, stats.sample_count);
    EXPECT_FLOAT_EQ(1.0f, stats.minimum);
    EXPECT_FLOAT_EQ(5.0f, stats.maximum);
    EXPECT_FLOAT_EQ(3.0f, stats.mean);
    EXPECT_FLOAT_EQ(11.0f, stats.m2);
    EXPECT_FLOAT_EQ(2.0f, stats.variance());
}

GTEST_TEST(Math_Statistics, adding) {
    int vs1[] = { 1, 2, 3 };
    Statistics<double> stats = Statistics<double>(vs1, vs1 + 3);
    stats.add(4);
    stats.add(5);
    EXPECT_EQ(5, stats.sample_count);
    EXPECT_FLOAT_EQ(1.0f, float(stats.minimum));
    EXPECT_FLOAT_EQ(5.0f, float(stats.maximum));
    EXPECT_FLOAT_EQ(3.0f, float(stats.mean));
    EXPECT_FLOAT_EQ(11.0f, float(stats.m2));
    EXPECT_FLOAT_EQ(2.0f, float(stats.variance()));
}

GTEST_TEST(Math_Statistics, merging) {
    int vs1[] = { 1, 2 };
    int vs2[] = { 3, 4, 5 };
    Statistics<double> stats = Statistics<double>(vs1, vs1 + 2);
    stats.merge_with(Statistics<double>(vs2, vs2 + 3));
    EXPECT_EQ(5, stats.sample_count);
    EXPECT_FLOAT_EQ(1.0f, float(stats.minimum));
    EXPECT_FLOAT_EQ(5.0f, float(stats.maximum));
    EXPECT_FLOAT_EQ(3.0f, float(stats.mean));
    EXPECT_FLOAT_EQ(11.0f, float(stats.m2));
    EXPECT_FLOAT_EQ(2.0f, float(stats.variance()));
}

} // NS Math
} // NS Cogwheel

#endif // _COGWHEEL_MATH_STATISTICS_TEST_H_