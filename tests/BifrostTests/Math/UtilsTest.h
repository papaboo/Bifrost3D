// Test Bifrost Utilities.
// ------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _BIFROST_MATH_UTILS_TEST_H_
#define _BIFROST_MATH_UTILS_TEST_H_

#include <Bifrost/Math/Utils.h>

#include <gtest/gtest.h>

namespace Bifrost {
namespace Math {

GTEST_TEST(Math_Utils, compute_ulps_handles_sign) {
    int zero_sign_difference = compute_ulps(0.0f, -0.0f);
    EXPECT_EQ(0, zero_sign_difference);
}

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

GTEST_TEST(Math_Utils, most_significant_bit) {
    EXPECT_EQ(0, most_significant_bit(0b1));
    EXPECT_EQ(1, most_significant_bit(0b10));
    EXPECT_EQ(1, most_significant_bit(0b11));
    EXPECT_EQ(3, most_significant_bit(0b1010));
    EXPECT_EQ(9, most_significant_bit(0b1000101100));
}

// ------------------------------------------------------------------------------------------------
// Gaussian bilinear samples test.
// ------------------------------------------------------------------------------------------------
GTEST_TEST(Math_Utils, bilinear_gaussian_samples) {
    static auto gaussian_filter = [](float* value, float std_dev, int support) -> float {
        float double_variance = 2.0f * std_dev * std_dev;

        float sum = 0.0f;
        float total_weight = 0.0f;
        for (int i = -support; i <= support; ++i) {
            float weight = exp(-(i * i) / double_variance);

            sum += *(value + i) * weight;
            total_weight += weight;
        }
        return sum / total_weight;
    };

    static auto sampled_filter = [](float* value, Tap* guassian_samples, int sample_count) -> float {
        float sum = 0.0f;
        for (int s = sample_count-1; s >= 0; --s) {
            int index = int(guassian_samples[s].offset);
            float frac = guassian_samples[s].offset - index;

            // left side
            float lower_value = lerp(value[-index], value[-index - 1], frac);
            float upper_value = lerp(value[index], value[index + 1], frac);
            sum += (lower_value + upper_value) * guassian_samples[s].weight;
        }

        return sum;
    };

    float values[21] = { 0, 0, 0, 0, 0, 0, 0, 
                         1, 1, 1, 1, 1, 1, 1, 
                         0, 0, 0, 0, 0, 0, 0 };

    const int support = 4;
    const int sample_count = (support + 1) / 2;
    Tap gaussian_samples[sample_count];
    for (float std_dev : {0.1f, 0.5f, 1.0f}) {

        fill_bilinear_gaussian_samples(std_dev, gaussian_samples, gaussian_samples + sample_count);

        // Test that the samples' weight sum to 0.5, as they cover the one half of the bell curve.
        float total_weight = 0.0;
        for (int s = sample_count - 1; s >= 0; --s)
            total_weight += gaussian_samples[s].weight;
        EXPECT_FLOAT_EQ_PCT(0.5f, total_weight, 0.0000005f);

        // Test that they filter similarly to a gaussian filter.
        for (int i : {5, 7, 10}) {
            float gaussian_filtered_value = gaussian_filter(values + i, std_dev, support);
            float sampled_filtered_value = sampled_filter(values + i, gaussian_samples, sample_count);
            EXPECT_FLOAT_EQ_PCT(gaussian_filtered_value, sampled_filtered_value, 0.0025f);
        }
    }
}

} // NS Math
} // NS Bifrost

#endif // _BIFROST_MATH_UTILS_TEST_H_
