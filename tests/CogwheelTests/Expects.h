// Test Cogwheel textures.
// ---------------------------------------------------------------------------
// Copyright (C) 2015-2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#include <Cogwheel/Math/Color.h>

#include <gtest/gtest.h>

#ifndef _COGWHEEL_TESTS_EXPECTS_H_
#define _COGWHEEL_TESTS_EXPECTS_H_

static bool equal_rgb(Cogwheel::Math::RGB lhs, Cogwheel::Math::RGB rhs) {
    return Cogwheel::Math::almost_equal(lhs.r, rhs.r)
        && Cogwheel::Math::almost_equal(lhs.g, rhs.g)
        && Cogwheel::Math::almost_equal(lhs.b, rhs.b);
}

#define EXPECT_RGB_EQ(expected, actual) EXPECT_PRED2(equal_rgb, expected, actual)

static bool equal_rgba(Cogwheel::Math::RGBA lhs, Cogwheel::Math::RGBA rhs) {
    return Cogwheel::Math::almost_equal(lhs.r, rhs.r)
        && Cogwheel::Math::almost_equal(lhs.g, rhs.g)
        && Cogwheel::Math::almost_equal(lhs.b, rhs.b)
        && Cogwheel::Math::almost_equal(lhs.a, rhs.a);
}

#define EXPECT_RGBA_EQ(expected, actual) EXPECT_PRED2(equal_rgba, expected, actual)

#endif // _COGWHEEL_TESTS_EXPECTS_H_