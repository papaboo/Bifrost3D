// Test OptiXRenderer's OrenNayar BRDF.
// ---------------------------------------------------------------------------
// Copyright (C) 2015-2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _OPTIXRENDERER_BSDFS_OREN_NAYAR_TEST_H_
#define _OPTIXRENDERER_BSDFS_OREN_NAYAR_TEST_H_

#include <Utils.h>

#include <Cogwheel/Math/Utils.h>

#include <OptiXRenderer/RNG.h>
#include <OptiXRenderer/Shading/BSDFs/OrenNayar.h>
#include <OptiXRenderer/Utils.h>

#include <gtest/gtest.h>

namespace OptiXRenderer {

GTEST_TEST(OrenNayar, power_conservation) {
    using namespace optix;

    const unsigned int MAX_SAMPLES = 1024u;
    const float3 tint = make_float3(1.0f, 1.0f, 1.0f);
    const float3 wo = normalize(make_float3(1.0f, 1.0f, 1.0f));
    
    for (int a = 0; a < 10; ++a) {
        const float alpha = a / 10.0f;
        float ws[MAX_SAMPLES];
        for (unsigned int i = 0u; i < MAX_SAMPLES; ++i) {
            BSDFSample sample = Shading::BSDFs::OrenNayar::sample(tint, alpha, wo, RNG::sample02(i));

            if (is_PDF_valid(sample.PDF))
                ws[i] = sample.weight.x * sample.direction.z / sample.PDF; // f * ||cos_theta|| / pdf
            else
                ws[i] = 0.0f;
        }

        float average_w = Cogwheel::Math::sort_and_pairwise_summation(ws, ws + MAX_SAMPLES) / float(MAX_SAMPLES);
        EXPECT_LE(average_w, 1.0f);
    }
}

GTEST_TEST(OrenNayar, Helmholtz_reciprocity) {
    using namespace optix;

    const unsigned int MAX_SAMPLES = 128u;
    const float3 tint = make_float3(1.0f, 1.0f, 1.0f);
    const float3 wo = normalize(make_float3(1.0f, 1.0f, 1.0f));

    for (int a = 0; a < 11; ++a) {
        const float alpha = lerp(0.2f, 1.0f, a / 10.0f);
        for (unsigned int i = 0u; i < MAX_SAMPLES; ++i) {
            BSDFSample sample = Shading::BSDFs::OrenNayar::sample(tint, alpha, wo, RNG::sample02(i));

            if (is_PDF_valid(sample.PDF)) {
                float3 f = Shading::BSDFs::OrenNayar::evaluate(tint, alpha, sample.direction, wo);

                EXPECT_COLOR_EQ_EPS(sample.weight, f, make_float3(0.0001f));
            }
        }
    }
}

GTEST_TEST(OrenNayar, consistent_PDF) {
    using namespace optix;

    const unsigned int MAX_SAMPLES = 128u;
    const float3 tint = make_float3(1.0f, 1.0f, 1.0f);
    const float3 wo = normalize(make_float3(1.0f, 1.0f, 1.0f));

    for (int a = 0; a < 11; ++a) {
        const float alpha = lerp(0.2f, 1.0f, a / 10.0f);
        for (unsigned int i = 0u; i < MAX_SAMPLES; ++i) {
            BSDFSample sample = Shading::BSDFs::OrenNayar::sample(tint, alpha, wo, RNG::sample02(i));
            if (is_PDF_valid(sample.PDF)) {
                float PDF = Shading::BSDFs::OrenNayar::PDF(alpha, wo, sample.direction);
                EXPECT_FLOAT_EQ_EPS(sample.PDF, PDF, 0.0001f);
            }
        }
    }
}

GTEST_TEST(OrenNayar, evaluate_with_PDF) {
    using namespace optix;

    const unsigned int MAX_SAMPLES = 128u;
    const float3 tint = make_float3(1.0f, 1.0f, 1.0f);
    const float3 wo = normalize(make_float3(1.0f, 1.0f, 1.0f));

    for (int a = 0; a < 11; ++a) {
        const float alpha = lerp(0.2f, 1.0f, a / 10.0f);
        for (unsigned int i = 0u; i < MAX_SAMPLES; ++i) {
            BSDFSample sample = Shading::BSDFs::OrenNayar::sample(tint, alpha, wo, RNG::sample02(i));

            if (is_PDF_valid(sample.PDF)) {
                BSDFResponse response = Shading::BSDFs::OrenNayar::evaluate_with_PDF(tint, alpha, wo, sample.direction);
                EXPECT_COLOR_EQ_EPS(Shading::BSDFs::OrenNayar::evaluate(tint, alpha, wo, sample.direction), response.weight, make_float3(0.000000001f));
                EXPECT_FLOAT_EQ(Shading::BSDFs::OrenNayar::PDF(alpha, wo, sample.direction), response.PDF);
            }
        }
    }
}

} // NS OptiXRenderer

#endif // _OPTIXRENDERER_BSDFS_OREN_NAYAR_TEST_H_