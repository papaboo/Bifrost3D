// Test Cogwheel Distribution1D.
// ---------------------------------------------------------------------------
// Copyright (C) 2015, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _COGWHEEL_MATH_DISTRIBUTION1D_TEST_H_
#define _COGWHEEL_MATH_DISTRIBUTION1D_TEST_H_

#include <Cogwheel/Math/Distribution1D.h>
#include <Cogwheel/Math/RNG.h>

#include <gtest/gtest.h>

namespace Cogwheel {
namespace Math {

GTEST_TEST(Math_Distribution1D, single_value_function) {
    double f = 1.0;
    const Distribution1D<float> distribution = Distribution1D<float>(&f, 1);

    EXPECT_FLOAT_EQ(0, distribution.sample_continuous(0.0f).index);
    EXPECT_FLOAT_EQ(1.0f, distribution.sample_continuous(0.0f).PDF);
    EXPECT_FLOAT_EQ(0.5f, distribution.sample_continuous(0.5f).index);
    EXPECT_FLOAT_EQ(1.0f, distribution.sample_continuous(0.5f).PDF);
    EXPECT_FLOAT_EQ(nearly_one, distribution.sample_continuous(nearly_one).index);
    EXPECT_FLOAT_EQ(1.0f, distribution.sample_continuous(nearly_one).PDF);
}

GTEST_TEST(Math_Distribution1D, constant_function) {
    float f[] = { 2, 2, 2 };
    int element_count = 3;

    const Distribution1D<float> distribution = Distribution1D<float>(f, element_count);

    for (int i = 0; i < 5; ++i) {
        float random_sample = i / 5.0f;
        { // Evaluate the function.
            EXPECT_FLOAT_EQ(f[int(random_sample * element_count)], distribution.evaluate(random_sample));
        }
        
        { // Sampling.
            EXPECT_FLOAT_EQ(random_sample, distribution.sample_continuous(random_sample).index);
            EXPECT_FLOAT_EQ(1.0f, distribution.sample_continuous(random_sample).PDF);
        }
    }
}

GTEST_TEST(Math_Distribution1D, non_constant_function) {
    float f[] = { 0, 5, 0, 2, 3 };
    int element_count = 5;

    const Distribution1D<float> distribution = Distribution1D<float>(f, element_count);
    float integral = distribution.get_integral();

    // Evaluate the function.
    for (int i = 0; i < 4; ++i) {
        float random_sample = i / 4.0f;
        EXPECT_FLOAT_EQ(f[int(random_sample * element_count)], distribution.evaluate(random_sample));
    }

    { // Sample center of cell 1.
        auto sample = distribution.sample_continuous(0.25f);
        EXPECT_EQ(1, int(sample.index * element_count));
        EXPECT_FLOAT_EQ(f[1], distribution.evaluate(sample.index));
        EXPECT_FLOAT_EQ(f[1] / integral, sample.PDF);
    }
    
    { // Sample right boundary of cell 1.
        auto sample = distribution.sample_continuous(previous_float(previous_float(0.5f)));
        EXPECT_EQ(1, int(sample.index * element_count));
        EXPECT_FLOAT_EQ(f[1], distribution.evaluate(sample.index));
        EXPECT_FLOAT_EQ(f[1] / integral, sample.PDF);
    }

    { // Sample left boundary of cell 3.
        auto sample = distribution.sample_continuous(0.5f);
        EXPECT_EQ(3, int(sample.index * element_count));
        EXPECT_FLOAT_EQ(f[3], distribution.evaluate(sample.index));
        EXPECT_FLOAT_EQ(f[3] / integral, sample.PDF);
    }

    { // Sample center of cell 4.
        auto sample = distribution.sample_continuous(0.85f);
        EXPECT_FLOAT_EQ(4.5f, sample.index * element_count);
        EXPECT_FLOAT_EQ(f[4], distribution.evaluate(sample.index));
        EXPECT_FLOAT_EQ(f[4] / integral, sample.PDF);
    }
}

GTEST_TEST(Math_Distribution1D, consistent_PDF) {
    float f[] = { 0, 5, 0, 8, 3 };
    const Distribution1D<float> distribution = Distribution1D<float>(f, 5);

    for (int i = 0; i < 32; ++i) {
        auto samplef = distribution.sample_continuous(RNG::van_der_corput(i, 59205));
        EXPECT_FLOAT_EQ(samplef.PDF, distribution.PDF_continuous(samplef.index));

        auto samplei = distribution.sample_discrete(RNG::van_der_corput(i, 59205));
        EXPECT_FLOAT_EQ(samplei.PDF, distribution.PDF_discrete(samplei.index));
    }
}

GTEST_TEST(Math_Distribution1D, reconstruct_continuous_function) {
    float f[] = { 0, 5, 0, 8, 3 };
    int element_count = 5;

    const Distribution1D<float> distribution = Distribution1D<float>(f, element_count);
    float integral = distribution.get_integral();

    const int ITERATION_COUNT = 8192;
    float f_sampled[] = { 0, 0, 0, 0, 0 };
    for (int i = 0; i < ITERATION_COUNT; ++i) {
        auto sample = distribution.sample_continuous(RNG::van_der_corput(i, 0u));
        float f = distribution.evaluate(sample.index);
        int index = int(sample.index * element_count);
        f_sampled[index] += f / sample.PDF * element_count;
    }

    for (int e = 0; e < element_count; ++e) {
        f_sampled[e] /= ITERATION_COUNT;
        EXPECT_FLOAT_EQ(f[e], f_sampled[e]);
    }
}

GTEST_TEST(Math_Distribution1D, reconstruct_discrete_function) {
    float f[] = { 0, 5, 0, 8, 3 };
    int element_count = 5;

    const Distribution1D<float> distribution = Distribution1D<float>(f, element_count);
    float integral = distribution.get_integral();

    const int ITERATION_COUNT = 8192;
    float f_sampled[] = { 0, 0, 0, 0, 0 };
    for (int i = 0; i < ITERATION_COUNT; ++i) {
        auto sample = distribution.sample_discrete(RNG::van_der_corput(i, 0u));
        float f = distribution.evaluate(sample.index);
        f_sampled[sample.index] += f / sample.PDF;
    }

    for (int e = 0; e < element_count; ++e) {
        f_sampled[e] /= ITERATION_COUNT;
        EXPECT_FLOAT_EQ(f[e], f_sampled[e]);
    }
}

} // NS Math
} // NS Cogwheel

#endif // _COGWHEEL_MATH_DISTRIBUTION1D_TEST_H_