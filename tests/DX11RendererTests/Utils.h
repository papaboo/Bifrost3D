// DX11Renderer testing utils.
// -------------------------------------------------------------------------------------------------
// Copyright (C) 2015-2017, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// -------------------------------------------------------------------------------------------------

#ifndef _DX11RENDERERTEST_UTILS_H_
#define _DX11RENDERERTEST_UTILS_H_

// -------------------------------------------------------------------------------------------------
// Comparison helpers.
// -------------------------------------------------------------------------------------------------

inline bool almost_equal_eps(float lhs, float rhs, float eps) {
    return lhs < rhs + eps && lhs + eps > rhs;
}

#define EXPECT_FLOAT_EQ_EPS(expected, actual, epsilon) EXPECT_PRED3(almost_equal_eps, expected, actual, epsilon)

inline bool almost_equal_percentage(float lhs, float rhs, float percentage) {
    float eps = lhs * percentage;
    return almost_equal_eps(lhs, rhs, eps);
}

#define EXPECT_FLOAT_EQ_PCT(expected, actual, percentage) EXPECT_PRED3(almost_equal_percentage, expected, actual, percentage)

#endif // _DX11RENDERERTEST_UTILS_H_