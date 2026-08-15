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

namespace MonteCarlo {

GTEST_TEST(Assets_Math_MonteCarlo, PDF) {
    { // PDF with valid value contains valid value.
        float simple_value = 0.5f;
        PDF simple_PDF = simple_value;
        EXPECT_EQ(simple_value, simple_PDF.value());
        EXPECT_TRUE(simple_PDF.is_valid());
        EXPECT_TRUE(simple_PDF.use_for_MIS());
        EXPECT_FALSE(simple_PDF.is_delta_dirac());

        // Disabling valid PDF for MIS flags MIS as disabled.
        simple_PDF.disable_MIS();
        EXPECT_EQ(simple_value, simple_PDF.value());
        EXPECT_TRUE(simple_PDF.is_valid());
        EXPECT_FALSE(simple_PDF.use_for_MIS());
        EXPECT_TRUE(simple_PDF.is_delta_dirac()); // We flag PDFs as invalid for MIS by flagging them as delta dirac, so this is unfortunately true.
    }

    {// PDF validity check ignores tiny values as they are not robustly handled by floating point math.
        float invalid_value = MIN_VALID_PDF * 0.5f;
        PDF invalid_PDF = invalid_value;
        EXPECT_EQ(invalid_value, invalid_PDF.value());
        EXPECT_FALSE(invalid_PDF.is_valid());
        EXPECT_FALSE(invalid_PDF.use_for_MIS());
        EXPECT_FALSE(invalid_PDF.is_delta_dirac());

        // Disable MIS on already invalid PDF changes nothing
        invalid_PDF.disable_MIS();
        EXPECT_EQ(invalid_value, invalid_PDF.value());
        EXPECT_FALSE(invalid_PDF.is_valid());
        EXPECT_FALSE(invalid_PDF.use_for_MIS());
        EXPECT_TRUE(invalid_PDF.is_delta_dirac()); // We flag PDFs as invalid for MIS by flagging them as delta dirac, so this is unfortunately true.
    }

    { // Delta dirac PDFs are invalid and cannot be used for MIS
        PDF delta_PDF = PDF::delta_dirac(1);
        EXPECT_TRUE(delta_PDF.is_valid());
        EXPECT_FALSE(delta_PDF.use_for_MIS());
        EXPECT_TRUE(delta_PDF.is_delta_dirac());

        // Disable MIS on a delta dirac PDF changes nothing as MIS isn't applicable to delta functions.
        delta_PDF.disable_MIS();
        EXPECT_TRUE(delta_PDF.is_valid());
        EXPECT_FALSE(delta_PDF.use_for_MIS());
        EXPECT_TRUE(delta_PDF.is_delta_dirac());
    }

    { // Invalid PDF
        PDF invalid_PDF = PDF::invalid();
        EXPECT_FALSE(invalid_PDF.is_valid());
        EXPECT_FALSE(invalid_PDF.use_for_MIS());
        EXPECT_TRUE(invalid_PDF.is_delta_dirac());

        // Disable MIS on already invalid PDF changes nothing
        invalid_PDF.disable_MIS();
        EXPECT_FALSE(invalid_PDF.is_valid());
        EXPECT_FALSE(invalid_PDF.use_for_MIS());
        EXPECT_TRUE(invalid_PDF.is_delta_dirac());
    }
}

GTEST_TEST(Assets_Math_MonteCarlo, Balance_heuristic_invariants) {
    // Sanity checks.
    EXPECT_FLOAT_EQ(0.5f, balance_heuristic(1.0f, 1.0f));
    EXPECT_FLOAT_EQ(0.25f, balance_heuristic(1.0f, 3.0f));

    // The balance heuristic should return 1 if the second pdf is NAN, as then the first sample trivially wins.
    EXPECT_EQ(1.0f, balance_heuristic(1.0f, NAN));

    float almost_inf = std::numeric_limits<float>::max();
    EXPECT_TRUE(isinf(almost_inf + almost_inf));

    // The balance heuristic should handle values close to infinity.
    EXPECT_FLOAT_EQ(1.0f / almost_inf, balance_heuristic(1.0f, almost_inf));
    EXPECT_FLOAT_EQ(1.0f, balance_heuristic(almost_inf, 1.0f));
    EXPECT_FLOAT_EQ(0.0f, balance_heuristic(0.5f * almost_inf, almost_inf));
    EXPECT_FLOAT_EQ(1.0f, balance_heuristic(almost_inf, 0.5f * almost_inf));

    // The balance heuristic should handle infinity.
    EXPECT_FLOAT_EQ(0.0f, balance_heuristic(1.0f, std::numeric_limits<float>::infinity()));
    EXPECT_FLOAT_EQ(1.0f, balance_heuristic(std::numeric_limits<float>::infinity(), 1.0f));

    // Zero should be a valid first parameter and always return zero.
    EXPECT_FLOAT_EQ(0.0f, balance_heuristic(0.0f, 0.0f));
    EXPECT_FLOAT_EQ(0.0f, balance_heuristic(0.0f, 1.0f));
    EXPECT_FLOAT_EQ(0.0f, balance_heuristic(0.0f, almost_inf));
}

GTEST_TEST(Assets_Math_MonteCarlo, Power_heuristic_invariants) {
    // Sanity checks.
    EXPECT_FLOAT_EQ(0.5f, power_heuristic(1.0f, 1.0f));
    EXPECT_FLOAT_EQ(0.1f, power_heuristic(1.0f, 3.0f));

    // The power heuristic should return 1 if the second pdf is NAN, as then the first sample trivially wins.
    EXPECT_EQ(1.0f, power_heuristic(1.0f, NAN));

    float almost_inf = std::numeric_limits<float>::max();
    EXPECT_TRUE(isinf(almost_inf * almost_inf));

    // The power heuristic should handle values that squared become infinity.
    EXPECT_FLOAT_EQ(0.0f, power_heuristic(1.0f, almost_inf));
    EXPECT_FLOAT_EQ(1.0f, power_heuristic(almost_inf, 1.0f));

    // Zero should be a valid first parameter and always return zero.
    EXPECT_FLOAT_EQ(0.0f, power_heuristic(0.0f, 0.0f));
    EXPECT_FLOAT_EQ(0.0f, power_heuristic(0.0f, 1.0f));
    EXPECT_FLOAT_EQ(0.0f, power_heuristic(0.0f, almost_inf));

    // Hacking the power heuristic by giving it pdf's that'll force the divisor to become infinite.
    EXPECT_FLOAT_EQ(0.0f, power_heuristic(0.9f * sqrt(almost_inf), sqrt(almost_inf)));
    EXPECT_FLOAT_EQ(1.0f, power_heuristic(sqrt(almost_inf), 0.9f * sqrt(almost_inf)));
}

} // NS MonteCarlo
} // NS Bifrost::Math

#endif // _BIFROST_MATH_RNG_TEST_H_
