// Test OptiXRenderer's Burley BRDF.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _OPTIXRENDERER_BSDFS_BURLEY_TEST_H_
#define _OPTIXRENDERER_BSDFS_BURLEY_TEST_H_

#include <Utils.h>

#include <Bifrost/Math/Utils.h>

#include <OptiXRenderer/RNG.h>
#include <OptiXRenderer/Shading/BSDFs/Burley.h>
#include <OptiXRenderer/Utils.h>

#include <gtest/gtest.h>

namespace OptiXRenderer {

GTEST_TEST(Burley, power_conservation) {
    using namespace optix;

    const unsigned int MAX_SAMPLES = 1024u;
    const float3 tint = make_float3(1.0f, 1.0f, 1.0f);
    const float3 wo = normalize(make_float3(1.0f, 1.0f, 1.0f));
    
    for (float roughness : {0.0f, 0.2f, 0.4f, 0.6f, 0.8f, 1.0f}) {
        float ws[MAX_SAMPLES];
        for (unsigned int i = 0u; i < MAX_SAMPLES; ++i) {
            BSDFSample sample = Shading::BSDFs::Burley::sample(tint, roughness, wo, RNG::sample02(i));

            if (is_PDF_valid(sample.PDF))
                ws[i] = sample.reflectance.x * sample.direction.z / sample.PDF; // f * ||cos_theta|| / pdf
            else
                ws[i] = 0.0f;
        }

        float average_w = Bifrost::Math::sort_and_pairwise_summation(ws, ws + MAX_SAMPLES) / float(MAX_SAMPLES);
        EXPECT_LE(average_w, 1.0f);
    }
}

GTEST_TEST(Burley, Helmholtz_reciprocity) {
    using namespace optix;

    const unsigned int MAX_SAMPLES = 128u;
    const float3 tint = make_float3(1.0f, 1.0f, 1.0f);
    const float3 wo = normalize(make_float3(1.0f, 1.0f, 1.0f));

    for (int r = 0; r < 11; ++r) {
        const float roughness = lerp(0.2f, 1.0f, r / 10.0f);
        for (unsigned int i = 0u; i < MAX_SAMPLES; ++i) {
            BSDFSample sample = Shading::BSDFs::Burley::sample(tint, roughness, wo, RNG::sample02(i));

            if (is_PDF_valid(sample.PDF)) {
                float3 f = Shading::BSDFs::Burley::evaluate(tint, roughness, sample.direction, wo);
                EXPECT_COLOR_EQ_EPS(sample.reflectance, f, make_float3(0.0001f));
            }
        }
    }
}

GTEST_TEST(Burley, consistent_PDF) {
    using namespace optix;

    const unsigned int MAX_SAMPLES = 128u;
    const float3 tint = make_float3(1.0f, 1.0f, 1.0f);
    const float3 wo = normalize(make_float3(1.0f, 1.0f, 1.0f));

    for (float roughness : {0.2f, 0.4f, 0.6f, 0.8f, 1.0f}) {
        for (unsigned int i = 0u; i < MAX_SAMPLES; ++i) {
            BSDFSample sample = Shading::BSDFs::Burley::sample(tint, roughness, wo, RNG::sample02(i));
            if (is_PDF_valid(sample.PDF)) {
                float PDF = Shading::BSDFs::Burley::PDF(roughness, wo, sample.direction);
                EXPECT_FLOAT_EQ_EPS(sample.PDF, PDF, 0.0001f);
            }
        }
    }
}

GTEST_TEST(Burley, evaluate_with_PDF) {
    using namespace optix;

    const unsigned int MAX_SAMPLES = 128u;
    const float3 tint = make_float3(1.0f, 1.0f, 1.0f);
    const float3 wo = normalize(make_float3(1.0f, 1.0f, 1.0f));

    for (float roughness : {0.2f, 0.4f, 0.6f, 0.8f, 1.0f}) {
        for (unsigned int i = 0u; i < MAX_SAMPLES; ++i) {
            BSDFSample sample = Shading::BSDFs::Burley::sample(tint, roughness, wo, RNG::sample02(i));

            if (is_PDF_valid(sample.PDF)) {
                BSDFResponse response = Shading::BSDFs::Burley::evaluate_with_PDF(tint, roughness, wo, sample.direction);
                EXPECT_COLOR_EQ_EPS(Shading::BSDFs::Burley::evaluate(tint, roughness, wo, sample.direction), response.reflectance, make_float3(0.000000001f));
                EXPECT_FLOAT_EQ(Shading::BSDFs::Burley::PDF(roughness, wo, sample.direction), response.PDF);
            }
        }
    }
}

} // NS OptiXRenderer

#endif // _OPTIXRENDERER_BSDFS_BURLEY_TEST_H_