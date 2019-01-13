// Test Bifrost Distribution2D.
// ---------------------------------------------------------------------------
// Copyright (C) 2015, Bifrost. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _BIFROST_MATH_DISTRIBUTION2D_TEST_H_
#define _BIFROST_MATH_DISTRIBUTION2D_TEST_H_

#include <Bifrost/Math/Distribution1D.h>
#include <Bifrost/Math/Distribution2D.h>
#include <Bifrost/Math/RNG.h>

#include <gtest/gtest.h>

namespace Bifrost {
namespace Math {

GTEST_TEST(Math_Distribution2D, single_value_function) {
    double f = 1.0;

    const Distribution2D<float> distribution = Distribution2D<float>(&f, 1, 1);

    EXPECT_EQ(Vector2f::zero(), distribution.sample_discrete(Vector2f(0.0f, 0.0f)).index);
    EXPECT_FLOAT_EQ(1.0f, distribution.sample_discrete(Vector2f(0.0f, 0.0f)).PDF);
    EXPECT_EQ(Vector2f::zero(), distribution.sample_discrete(Vector2f(nearly_one, 0.0f)).index);
    EXPECT_FLOAT_EQ(1.0f, distribution.sample_discrete(Vector2f(nearly_one, 0.0f)).PDF);
    EXPECT_EQ(Vector2f::zero(), distribution.sample_discrete(Vector2f(0.0f, nearly_one)).index);
    EXPECT_FLOAT_EQ(1.0f, distribution.sample_discrete(Vector2f(0.0f, nearly_one)).PDF);
    EXPECT_EQ(Vector2f::zero(), distribution.sample_discrete(Vector2f(nearly_one, nearly_one)).index);
    EXPECT_FLOAT_EQ(1.0f, distribution.sample_discrete(Vector2f(nearly_one, nearly_one)).PDF);
    EXPECT_EQ(Vector2f::zero(), distribution.sample_discrete(Vector2f(0.5f, 0.5f)).index);
    EXPECT_FLOAT_EQ(1.0f, distribution.sample_discrete(Vector2f(0.5f, 0.5f)).PDF);
}

GTEST_TEST(Math_Distribution2D, constant_function) {
    float f[] = { 3, 3, 3, 3 };

    const Distribution2D<float> distribution = Distribution2D<float>(f, 2, 2);
    float integral = distribution.get_integral();

    for (int y = 0; y < 5; ++y)
        for (int x = 0; x < 5; ++x) {
            Vector2f random_sample = { x / 5.0f, y / 5.0f };
            Vector2i index = { int(random_sample.x * distribution.get_width()), 
                               int(random_sample.y * distribution.get_height()) };
            int f_index = index.x + index.y * distribution.get_width();
            
            { // Evaluate the function.
                EXPECT_FLOAT_EQ(f[f_index], distribution.evaluate(random_sample));
            }

            { // Sampling.
                EXPECT_VECTOR2F_EQ(random_sample, distribution.sample_continuous(random_sample).index);
                EXPECT_FLOAT_EQ(1.0f, distribution.sample_continuous(random_sample).PDF);
            }
        }
}

GTEST_TEST(Math_Distribution2D, non_constant_function) {
    float f[] = { 0, 5, 0, 3,
                  2, 1, 1, 4 };

    const Distribution2D<float> distribution = Distribution2D<float>(f, 4, 2);
    float integral = distribution.get_integral();

    // Evaluate the function.
    for (int y = 0; y < 2; ++y)
        for (int x = 0; x < 4; ++x) {
            Vector2i index = { x, y };
            int f_index = index.x + index.y * distribution.get_width();
            EXPECT_FLOAT_EQ(f[f_index], distribution.evaluate(index));
        }

    // Sample top left of [1, 0]
    Vector2f random = Vector2f(0.0f, 0.0f);
    EXPECT_VECTOR2F_EQ(Vector2f(0.25f, 0.0f), distribution.sample_continuous(random).index);
    EXPECT_FLOAT_EQ(f[1] / integral, distribution.sample_continuous(random).PDF);
    
    // Sample center of [1, 0]
    random = Vector2f(0.3125f, 0.25f);
    EXPECT_VECTOR2F_EQ(Vector2f(0.375f, 0.25f), distribution.sample_continuous(random).index);
    EXPECT_FLOAT_EQ(f[1] / integral, distribution.sample_continuous(random).PDF);
    
    // Sample bottom right of first row.
    random = Vector2f(nearly_one, previous_float(0.5f));
    EXPECT_VECTOR2F_EQ(Vector2f(1.0f, 0.5f), distribution.sample_continuous(random).index);
    EXPECT_FLOAT_EQ(f[3] / integral, distribution.sample_continuous(random).PDF);
}

GTEST_TEST(Math_Distribution2D, consistent_PDF) {
    float f[] = { 0, 5, 0, 3,
                  2, 1, 1, 4 };
    const Distribution2D<float> distribution = Distribution2D<float>(f, 4, 2);

    for (int i = 0; i < 32; ++i) {
        auto samplef = distribution.sample_continuous(RNG::sample02(i));
        EXPECT_FLOAT_EQ(samplef.PDF, distribution.PDF_continuous(samplef.index));

        auto samplei = distribution.sample_discrete(RNG::sample02(i));
        EXPECT_FLOAT_EQ(samplei.PDF, distribution.PDF_discrete(samplei.index));
    }
}

GTEST_TEST(Math_Distribution2D, reconstruct_continuous_function) {
    float f[] = { 0, 5, 0, 3,
                  2, 1, 1, 4 };
    int width = 4, height = 2, element_count = width * height;

    const Distribution2D<float> distribution = Distribution2D<float>(f, width, height);
    float integral = distribution.get_integral();

    const int ITERATION_COUNT = 8192;
    float f_sampled[] = { 0, 0, 0, 0, 
                          0, 0, 0, 0 };
    for (int i = 0; i < ITERATION_COUNT; ++i) {
        auto sample = distribution.sample_continuous(RNG::sample02(i, Vector2ui::zero()));
        float f = distribution.evaluate(sample.index);
        Vector2i index = { int(sample.index.x * width), int(sample.index.y * height) };
        int f_index = index.x + index.y * width;
        EXPECT_LT(f_index, element_count);
        f_sampled[f_index] += f / sample.PDF * element_count;
    }

    for (int e = 0; e < element_count; ++e) {
        f_sampled[e] /= ITERATION_COUNT;
        EXPECT_FLOAT_EQ(f[e], f_sampled[e]);
    }
}

GTEST_TEST(Math_Distribution2D, reconstruct_discrete_function) {
    float f[] = { 0, 5, 0, 3,
                  2, 1, 1, 4 };
    int width = 4, height = 2, element_count = width * height;

    const Distribution2D<float> distribution = Distribution2D<float>(f, width, height);
    float integral = distribution.get_integral();

    const int ITERATION_COUNT = 8192;
    float f_sampled[] = { 0, 0, 0, 0,
                          0, 0, 0, 0 };
    for (int i = 0; i < ITERATION_COUNT; ++i) {
        auto sample = distribution.sample_discrete(RNG::sample02(i, Vector2ui::zero()));
        float f = distribution.evaluate(sample.index);
        int f_index = sample.index.x + sample.index.y * width;
        EXPECT_LT(f_index, element_count);
        f_sampled[f_index] += f / sample.PDF;
    }

    for (int e = 0; e < element_count; ++e) {
        f_sampled[e] /= ITERATION_COUNT;
        EXPECT_FLOAT_EQ(f[e], f_sampled[e]);
    }
}

GTEST_TEST(Math_Distribution2D, distribution1D_equality) {
    float f[] = { 0, 5, 0, 8, 3 };
    int element_count = 5;

    const Distribution1D<float> distribution_1D = Distribution1D<float>(f, element_count);
    float integral_1D = distribution_1D.get_integral();

    const Distribution2D<float> vertical_distribution = Distribution2D<float>(f, 1, element_count);
    const Distribution2D<float> horizontal_distribution = Distribution2D<float>(f, element_count, 1);
    EXPECT_FLOAT_EQ(integral_1D, vertical_distribution.get_integral());
    EXPECT_FLOAT_EQ(integral_1D, horizontal_distribution.get_integral());

    for (int i = 0; i < 32; ++i) {
        float rn = RNG::van_der_corput(i, 59305u);

        { // Discrete samples
            auto sample_1D = distribution_1D.sample_discrete(rn);
            auto vertical_sample = vertical_distribution.sample_discrete(Vector2f(0.5f, rn));
            auto horizontal_sample = horizontal_distribution.sample_discrete(Vector2f(rn, 0.5f));

            EXPECT_EQ(sample_1D.index, vertical_sample.index.y);
            EXPECT_EQ(sample_1D.index, horizontal_sample.index.x);

            EXPECT_EQ(sample_1D.PDF, vertical_sample.PDF);
            EXPECT_EQ(sample_1D.PDF, horizontal_sample.PDF);
            
            float f_1D = distribution_1D.evaluate(sample_1D.index);
            EXPECT_EQ(f_1D, vertical_distribution.evaluate(vertical_sample.index));
            EXPECT_EQ(f_1D, horizontal_distribution.evaluate(horizontal_sample.index));
        }

        { // Continuous samples
            auto sample_1D = distribution_1D.sample_continuous(rn);
            auto vertical_sample = vertical_distribution.sample_continuous(Vector2f(0.5f, rn));
            auto horizontal_sample = horizontal_distribution.sample_continuous(Vector2f(rn, 0.5f));

            EXPECT_EQ(sample_1D.index, vertical_sample.index.y);
            EXPECT_EQ(sample_1D.index, horizontal_sample.index.x);

            EXPECT_EQ(sample_1D.PDF, vertical_sample.PDF);
            EXPECT_EQ(sample_1D.PDF, horizontal_sample.PDF);

            float f_1D = distribution_1D.evaluate(sample_1D.index);
            EXPECT_EQ(f_1D, vertical_distribution.evaluate(vertical_sample.index));
            EXPECT_EQ(f_1D, horizontal_distribution.evaluate(horizontal_sample.index));
        }
    }
}

} // NS Math
} // NS Bifrost

#endif // _BIFROST_MATH_DISTRIBUTION2D_TEST_H_
