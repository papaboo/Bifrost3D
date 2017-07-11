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
    Statistics stats = Statistics(vs, vs + 5);
    EXPECT_EQ(5, stats.sample_count);
    EXPECT_FLOAT_EQ(1.0f, stats.minimum);
    EXPECT_FLOAT_EQ(5.0f, stats.maximum);
    EXPECT_FLOAT_EQ(3.0f, stats.mean);
    EXPECT_FLOAT_EQ(11.0f, stats.m2);
    EXPECT_FLOAT_EQ(2.0f, stats.variance());
}

GTEST_TEST(Math_Statistics, merging) {
    int vs1[] = { 1, 2 };
    int vs2[] = { 3, 4, 5 };
    Statistics stats = Statistics(vs1, vs1 + 2);
    stats.merge_with(Statistics(vs2, vs2 + 3));
    EXPECT_EQ(5, stats.sample_count);
    EXPECT_FLOAT_EQ(1.0f, stats.minimum);
    EXPECT_FLOAT_EQ(5.0f, stats.maximum);
    EXPECT_FLOAT_EQ(3.0f, stats.mean);
    EXPECT_FLOAT_EQ(11.0f, stats.m2);
    EXPECT_FLOAT_EQ(2.0f, stats.variance());
}

} // NS Math
} // NS Cogwheel

#endif // _COGWHEEL_MATH_STATISTICS_TEST_H_