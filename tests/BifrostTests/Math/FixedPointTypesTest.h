// Test Bifrost fixed point types.
// ------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _BIFROST_MATH_FIXED_POINT_TYPES_TEST_H_
#define _BIFROST_MATH_FIXED_POINT_TYPES_TEST_H_

#include <Bifrost/Math/FixedPointTypes.h>

#include <gtest/gtest.h>

namespace Bifrost {
namespace Math {

GTEST_TEST(Math_FixedPointTypes, UNorm8) {
    // Test numerical constants
    EXPECT_EQ(0, UNorm8::zero().raw);
    EXPECT_EQ(0.0f, UNorm8::zero().value());
    EXPECT_EQ(255, UNorm8::one().raw);
    EXPECT_EQ(1.0f, UNorm8::one().value());

    // The 1/2 value is going to be exactly halfway between two UNorm8 representations and should be max precision away from it's input value.
    UNorm8 half = UNorm8(0.5f);
    EXPECT_EQ(128, half.raw);
    EXPECT_EQ(0.5f + UNorm8::max_precision(), half.value());

    // Test assignment operator overload
    UNorm8 quater = (unsigned char)64;
    EXPECT_EQ(64, quater.raw);

    // Test overflow handling. Integer part is discarded and fraction is kept.
    UNorm8 half2 = 1.5f;
    EXPECT_EQ(half.raw, half2.raw);
}

} // NS Math
} // NS Bifrost

#endif // _BIFROST_MATH_FIXED_POINT_TYPES_TEST_H_
