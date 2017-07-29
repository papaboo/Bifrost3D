// Test Cogwheel Utilities.
// ------------------------------------------------------------------------------------------------
// Copyright (C) 2015, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _COGWHEEL_MATH_UTILS_TEST_H_
#define _COGWHEEL_MATH_UTILS_TEST_H_

#include <Cogwheel/Math/Utils.h>

#include <gtest/gtest.h>

namespace Cogwheel {
namespace Math {

GTEST_TEST(Math_Utils, previous_float) {
    for (float v : {-1.0f, -0.0f, 0.0f, 1.0f}) {
        float previous_v = previous_float(v);
        EXPECT_LT(previous_v, v);
        EXPECT_EQ(1, compute_ulps(v, previous_v));
    }
}

GTEST_TEST(Math_Utils, next_float) {
    for (float v : {-2.0f, -1.0f, -0.0f, 0.0f, 1.0f, 2.0f}) {
        float next_v = next_float(v);
        EXPECT_LT(v, next_v);
        EXPECT_EQ(1, compute_ulps(v, next_v));
    }
}

GTEST_TEST(Math_Utils, compute_ulps) {
    std::vector<float> vs = { -2.0f, -1.0f, -0.0f, 0.0f, 1.0f, 2.0f };
    for (int i = 0; i < vs.size(); ++i) {
        float v = vs[i];
        float next_v = v, previous_v = v;
        for (int j = 0; j < i; ++j) {
            next_v = next_float(next_v);
            previous_v = previous_float(previous_v);
        }

        int expected_ulps = i;
        EXPECT_EQ(expected_ulps, compute_ulps(v, next_v));
        EXPECT_EQ(expected_ulps, compute_ulps(next_v, v));
        EXPECT_EQ(expected_ulps, compute_ulps(v, previous_v));
        EXPECT_EQ(expected_ulps, compute_ulps(previous_v, v));
    }
}

} // NS Math
} // NS Cogwheel

#endif // _COGWHEEL_MATH_UTILS_TEST_H_