// Test Bifrost random number generation and hashing.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _BIFROST_MATH_RNG_TEST_H_
#define _BIFROST_MATH_RNG_TEST_H_

#include <Bifrost/Math/RNG.h>

#include <gtest/gtest.h>

namespace Bifrost::Math {

GTEST_TEST(Math_RNG, PMJB_non_power_of_two_sample_count) {
    const int sample_count = 16;
    Vector2f samples[sample_count];

    // Fill array with invalid sentinel value.
    Vector2f sentinel = { 1e10f, 1e20f };
    std::fill_n(samples, sample_count, sentinel);

    int filled_sample_count = 16;
    RNG::fill_progressive_multijittered_bluenoise_samples(samples, samples + filled_sample_count);

    // Test that valid samples are in [0, 1[ range.
    for (int i = 0; i < filled_sample_count; ++i) {
        EXPECT_GE(samples[i].x, 0.0f);
        EXPECT_GE(samples[i].y, 0.0f);
        EXPECT_LT(samples[i].x, 1.0f);
        EXPECT_LT(samples[i].y, 1.0f);
    }

    // Test that samples outside the filled range maintain the sentinel value.
    for (int i = filled_sample_count; i < sample_count; ++i)
        EXPECT_EQ(samples[i], sentinel);
}

} // NS Bifrost::Math

#endif // _BIFROST_MATH_RNG_TEST_H_
