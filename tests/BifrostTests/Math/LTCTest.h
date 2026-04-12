// Test Bifrost linearly transformed cosines.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _BIFROST_MATH_LTC_TEST_H_
#define _BIFROST_MATH_LTC_TEST_H_

#include <Bifrost/Math/Distributions.h>
#include <Bifrost/Math/LTC.h>
#include <Bifrost/Math/RNG.h>

#include <gtest/gtest.h>

namespace Bifrost::Math {

GTEST_TEST(Math_LTC, identity_LTC_has_identity_matrices) {
    IsotropicLTC ltc = IsotropicLTC::identity();

    Matrix3x3f m = ltc.get_M();
    Matrix3x3f inverse_m = ltc.get_inverse_M();

    EXPECT_MATRIX3X3F_EQ(Matrix3x3f::identity(), m);
    EXPECT_MATRIX3X3F_EQ(Matrix3x3f::identity(), inverse_m);
}

GTEST_TEST(Math_LTC, inverse_M_is_inverse_of_M) {

    RNG::XorShift32 rng = RNG::XorShift32(3);
    // Test a few randomly initialized LTCs.
    for (int i = 0; i < 5; ++i) {
        IsotropicLTC ltc = { rng.sample1f(), rng.sample1f(), rng.sample1f(), rng.sample1f(), rng.sample1f() };

        Matrix3x3f expected_m = ltc.get_M();
        Matrix3x3f actual_m = invert(ltc.get_inverse_M());

        EXPECT_MATRIX3X3F_EQ(expected_m, actual_m);
    }
}

GTEST_TEST(Math_LTC, identity_LTC_is_diffuse_distribution) {
    IsotropicLTC ltc = IsotropicLTC::identity();

    RNG::XorShift32 rng = RNG::XorShift32(3);
    for (int s = 0; s < 8; ++s) {
        Vector2f random_sample = rng.sample2f();

        auto ltc_sample = ltc.sample(random_sample);
        auto cosine_sample = Distributions::Cosine::sample(random_sample);

        EXPECT_FLOAT_EQ_EPS(cosine_sample.PDF, ltc_sample.PDF, 1e-6f);
        EXPECT_NORMAL_EQ(cosine_sample.direction, ltc_sample.direction, 1e-5f);
    }
}

GTEST_TEST(Math_LTC, perfectly_sampled) {
    // Test a few randomly initialized LTCs.
    RNG::XorShift32 rng = RNG::XorShift32(3);
    for (int i = 0; i < 5; ++i) {
        IsotropicLTC ltc = { rng.sample1f(), rng.sample1f(), rng.sample1f(), rng.sample1f(), rng.sample1f() };

        for (float cos_theta_i : { 0.0f, 0.4f, 0.7f, 1.0f}) {
            Vector3f wi = { sqrt(1 - pow2(cos_theta_i)), 0.0f, cos_theta_i };
            float evaluation = ltc.evaluate(wi);
            float PDF = ltc.PDF(wi);
            EXPECT_FLOAT_EQ(PDF, evaluation);
        }

    }
}

GTEST_TEST(Math_LTC, integrates_to_one) {
    const unsigned int sample_count = 512u;
    Bifrost::Math::RNG::PmjbRNG ltc_rng(sample_count);

    // Test a few randomly initialized LTCs.
    RNG::XorShift32 rng = RNG::XorShift32(3);
    for (int i = 0; i < 5; ++i) {
        IsotropicLTC ltc = { rng.sample1f(), rng.sample1f(), rng.sample1f(), rng.sample1f(), rng.sample1f() };

        double sum = 0.0f;
        for (unsigned int i = 0; i < sample_count; ++i) {
            auto ltc_sample = ltc.sample(ltc_rng.sample2f(i));
            sum += ltc.evaluate(ltc_sample.direction) / ltc_sample.PDF;
        }
        float integral = float(sum / sample_count);

        EXPECT_FLOAT_EQ(1.0f, integral);
    }
}

} // NS Bifrost::Math

#endif // _BIFROST_MATH_LTC_TEST_H_
