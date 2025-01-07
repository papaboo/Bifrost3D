// Test Bifrost lines.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _BIFROST_MATH_LINE_TEST_H_
#define _BIFROST_MATH_LINE_TEST_H_

#include <Bifrost/Math/Line.h>

#include <gtest/gtest.h>

namespace Bifrost::Math {

GTEST_TEST(Math_Line, construct_from_points) {
    Vector2f point0 = Vector2f(2, 2);
    Vector2f point1 = Vector2f(4, 3);
    
    Linef line0 = Line(point0, point1);
    EXPECT_FLOAT_EQ(0.5f, line0.slope);
    EXPECT_FLOAT_EQ(1.0f, line0.intercept);

    Linef line1 = Line(point0, point1);
    EXPECT_FLOAT_EQ(0.5f, line1.slope);
    EXPECT_FLOAT_EQ(1.0f, line1.intercept);
}

GTEST_TEST(Math_Line, fit_yields_correct_mean_squared_error) {
    Vector2d points[4] = { Vector2d(0, 0), Vector2d(1, 0), Vector2d(2, 2), Vector2d(3, 2), };

    auto fit = LineFitterd::fit(points, points + 4);
    float actual_mse = (float)fit.mse;
    float expected_mse = (float)fit.line.mse(points, points + 4);
    EXPECT_FLOAT_EQ(expected_mse, actual_mse);
}

GTEST_TEST(Math_Line, fit_finds_the_minima) {
    Vector2f points[4] = { Vector2f(0, 0), Vector2f(1, 0), Vector2f(2, 2), Vector2f(3, 2), };

    auto fit = LineFitterf::fit(points, points + 4);

    Linef line_near_minima[] = {
        Linef(fit.line.slope + 0.001f, fit.line.intercept),
        Linef(fit.line.slope - 0.001f, fit.line.intercept),
        Linef(fit.line.slope, fit.line.intercept + 0.001f),
        Linef(fit.line.slope, fit.line.intercept - 0.001f),
    };

    for (Linef tested_line : line_near_minima) {
        float tested_mse = tested_line.mse(points, points + 4);
        EXPECT_LT(fit.mse, tested_mse);
    }
}

} // NS Bifrost::Math

#endif // _BIFROST_MATH_LINE_TEST_H_
