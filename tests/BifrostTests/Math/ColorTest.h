// Test Bifrost colors.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _BIFROST_MATH_COLOR_TEST_H_
#define _BIFROST_MATH_COLOR_TEST_H_

#include <Bifrost/Math/Color.h>

#include <gtest/gtest.h>

namespace Bifrost::Math {

inline bool equal_hsv_eps(Bifrost::Math::HSV lhs, Bifrost::Math::HSV rhs, float eps) {
    // Account for hue wrapping around so hue 0 == 360
    float abs_hue_diff = abs(lhs.h - rhs.h);
    return (abs_hue_diff < eps || (abs_hue_diff - 360) < eps) && abs(lhs.s - rhs.s) < eps && abs(lhs.v - rhs.v) < eps;
}
#define EXPECT_HSV_EQ_EPS(expected, actual, eps) EXPECT_PRED3(equal_hsv_eps, expected, actual, eps)

GTEST_TEST(Math_Color_HSV, color_conversion) {
    RGB white = RGB(1);
    RGB grey = RGB(0.5f);
    RGB black = RGB(0);
    RGB red = RGB(1, 0, 0);
    RGB dark_green = RGB(0, 0.5f, 0);
    RGB blue = RGB(0, 0, 1);
    RGB dark_purple = RGB(0.5f, 0, 0.5f);
    RGB yellow = RGB(1, 1, 0);
    RGB teal = RGB(0, 0.5f, 0.5f);

    HSV actual_hsv_white = HSV(white);
    HSV actual_hsv_grey = HSV(grey);
    HSV actual_hsv_black = HSV(black);
    HSV actual_hsv_red = HSV(red);
    HSV actual_hsv_dark_green = HSV(dark_green);
    HSV actual_hsv_blue = HSV(blue);
    HSV actual_hsv_dark_purple = HSV(dark_purple);
    HSV actual_hsv_yellow = HSV(yellow);
    HSV actual_hsv_teal = HSV(teal);

    HSV expected_hsv_white = HSV(0, 0, 1);
    HSV expected_hsv_grey = HSV(0, 0, 0.5f);
    HSV expected_hsv_black = HSV(0, 0, 0);
    HSV expected_hsv_red = HSV(0, 1, 1);
    HSV expected_hsv_dark_green = HSV(120, 1, 0.5f);
    HSV expected_hsv_blue = HSV(240, 1, 1);
    HSV expected_hsv_dark_purple = HSV(300, 1, 0.5f);
    HSV expected_hsv_yellow = HSV(60, 1, 1);
    HSV expected_hsv_teal = HSV(180, 1, 0.5f);

    EXPECT_HSV_EQ_EPS(expected_hsv_white, actual_hsv_white, 0.0001f);
    EXPECT_HSV_EQ_EPS(expected_hsv_grey, actual_hsv_grey, 0.0001f);
    EXPECT_HSV_EQ_EPS(expected_hsv_black, actual_hsv_black, 0.0001f);
}

GTEST_TEST(Math_Color_HSV, RGB_to_HSV_to_RGB_conversion_consistency) {
    for (float r = 0; r <= 2.0f; r += 0.5f)
        for (float g = 0; g <= 2.0f; g += 0.5f)
            for (float b = 0; b <= 2.0f; b += 0.5f) {
                RGB expected_rgb = RGB(r, g, b);
                RGB actual_rgb = (RGB)HSV(expected_rgb);
                EXPECT_RGB_EQ_EPS(expected_rgb, actual_rgb, 0.00001f);
            }
}

GTEST_TEST(Math_Color_HSV, lerp_towards_closest_hue) {
    HSV low_hue = HSV(60, 0, 0);
    HSV middle_hue = HSV(180, 1, 0);
    HSV high_hue = HSV(300, 0, 1);
    float t = 0.25f;

    { // Lerp between low and middle
        HSV low_to_middle = lerp(low_hue, middle_hue, t);
        EXPECT_FLOAT_EQ(low_to_middle.h, 90.0f);
        EXPECT_FLOAT_EQ(low_to_middle.s, t);
        EXPECT_FLOAT_EQ(low_to_middle.v, 0.0f);

        HSV middle_to_low = lerp(middle_hue, low_hue, t);
        EXPECT_FLOAT_EQ(middle_to_low.h, 150.0f);
        EXPECT_FLOAT_EQ(middle_to_low.s, 1 - t);
        EXPECT_FLOAT_EQ(middle_to_low.v, 0.0f);
    }

    { // Lerp between middle and high
        HSV middle_to_high = lerp(middle_hue, high_hue, t);
        EXPECT_FLOAT_EQ(middle_to_high.h, 210.0f);
        EXPECT_FLOAT_EQ(middle_to_high.s, 1 - t);
        EXPECT_FLOAT_EQ(middle_to_high.v, t);

        HSV high_to_middle = lerp(high_hue, middle_hue, t);
        EXPECT_FLOAT_EQ(high_to_middle.h, 270.0f);
        EXPECT_FLOAT_EQ(high_to_middle.s, t);
        EXPECT_FLOAT_EQ(high_to_middle.v, 1 - t);
    }

    { // Lerp between low and high
        HSV low_to_high = lerp(low_hue, high_hue, t);
        EXPECT_FLOAT_EQ(low_to_high.h, 30.0f);
        EXPECT_FLOAT_EQ(low_to_high.s, 0);
        EXPECT_FLOAT_EQ(low_to_high.v, t);

        HSV high_to_low = lerp(high_hue, low_hue, t);
        EXPECT_FLOAT_EQ(high_to_low.h, 330.0f);
        EXPECT_FLOAT_EQ(high_to_low.s, 0);
        EXPECT_FLOAT_EQ(high_to_low.v, 1 - t);
    }
}

GTEST_TEST(Math_Color_sRGB, sRGB_to_linear_to_sRGB_identity_conversions) {
    for (int v = 0; v < 256; v++) {
        UNorm8 expected = byte(v);
        UNorm8 actual = linear_to_sRGB(sRGB_to_linear(expected));
        EXPECT_FLOAT_EQ(expected.raw, actual.raw);
    }
}

GTEST_TEST(Math_Color_sRGB, linear_to_sRGB_to_linear_identity_conversions) {
    for (float v = 0.0f; v <= 1.0f; v += 0.125f)
        EXPECT_FLOAT_EQ(v, sRGB_to_linear(linear_to_sRGB(v)));
}

} // NS Bifrost::Math

#endif // _BIFROST_MATH_COLOR_TEST_H_
