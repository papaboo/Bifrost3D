// Test OptiXRenderer's GGX distribution and BSDF.
// ---------------------------------------------------------------------------
// Copyright (C) 2015-2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _OPTIXRENDERER_BSDFS_GGX_TEST_H_
#define _OPTIXRENDERER_BSDFS_GGX_TEST_H_

#include <Utils.h>

#include <Cogwheel/Math/Statistics.h>
#include <Cogwheel/Math/Utils.h>

#include <OptiXRenderer/RNG.h>
#include <OptiXRenderer/Shading/BSDFs/GGX.h>
#include <OptiXRenderer/Utils.h>

#include <gtest/gtest.h>

namespace OptiXRenderer {

GTEST_TEST(GGX, power_conservation) {
    using namespace optix;

    const unsigned int MAX_SAMPLES = 1024u;
    const float3 tint = make_float3(1.0f, 1.0f, 1.0f);

    for (int i = 0; i < 10; ++i) {
        const float3 wo = normalize(make_float3(float(i), 0.0f, 1.001f - float(i) * 0.1f));
        for (int a = 0; a < 10; ++a) {
            const float alpha = a / 10.0f;
            float ws[MAX_SAMPLES];
            for (unsigned int i = 0u; i < MAX_SAMPLES; ++i) {
                BSDFSample sample = Shading::BSDFs::GGX::sample(tint, alpha, wo, RNG::sample02(i));

                if (is_PDF_valid(sample.PDF))
                    ws[i] = sample.weight.x * sample.direction.z / sample.PDF; // f * ||cos_theta|| / pdf
                else
                    ws[i] = 0.0f;
            }

            float average_w = Cogwheel::Math::sort_and_pairwise_summation(ws, ws + MAX_SAMPLES) / float(MAX_SAMPLES);
            EXPECT_LE(average_w, 1.0f);
        }
    }
}

GTEST_TEST(GGX, Helmholtz_reciprocity) {
    using namespace optix;

    const unsigned int MAX_SAMPLES = 128u;
    const float3 tint = make_float3(1.0f, 1.0f, 1.0f);
    const float3 wo = normalize(make_float3(1.0f, 1.0f, 1.0f));

    for (int a = 0; a < 11; ++a) {
        const float alpha = lerp(0.2f, 1.0f, a / 10.0f);
        for (unsigned int i = 0u; i < MAX_SAMPLES; ++i) {
            BSDFSample sample = Shading::BSDFs::GGX::sample(tint, alpha, wo, RNG::sample02(i));

            if (is_PDF_valid(sample.PDF)) {
                float3 f = Shading::BSDFs::GGX::evaluate(tint, alpha, sample.direction, wo, normalize(sample.direction + wo));

                EXPECT_COLOR_EQ_EPS(sample.weight, f, make_float3(0.0001f));
            }
        }
    }
}

GTEST_TEST(GGX, consistent_PDF) {
    using namespace optix;

    const unsigned int MAX_SAMPLES = 128u;
    const float3 tint = make_float3(1.0f, 1.0f, 1.0f);
    const float3 wo = normalize(make_float3(1.0f, 1.0f, 1.0f));

    for (int a = 0; a < 11; ++a) {
        const float alpha = lerp(0.2f, 1.0f, a / 10.0f);
        for (unsigned int i = 0u; i < MAX_SAMPLES; ++i) {
            BSDFSample sample = Shading::BSDFs::GGX::sample(tint, alpha, wo, RNG::sample02(i));
            if (is_PDF_valid(sample.PDF)) {
                float3 wi = sample.direction;
                float PDF = Shading::BSDFs::GGX::PDF(alpha, wo, wi, normalize(wo + wi));
                EXPECT_FLOAT_EQ_EPS(sample.PDF, PDF, 0.0001f);
            }
        }
    }
}

GTEST_TEST(GGX, evaluate_with_PDF) {
    using namespace optix;

    const unsigned int MAX_SAMPLES = 128u;
    const float3 tint = make_float3(1.0f, 1.0f, 1.0f);
    const float3 wo = normalize(make_float3(1.0f, 1.0f, 1.0f));

    for (int a = 0; a < 11; ++a) {
        const float alpha = lerp(0.2f, 1.0f, a / 10.0f);
        for (unsigned int i = 0u; i < MAX_SAMPLES; ++i) {
            BSDFSample sample = Shading::BSDFs::GGX::sample(tint, alpha, wo, RNG::sample02(i));

            if (is_PDF_valid(sample.PDF)) {
                float3 wi = sample.direction;
                float3 halfway = normalize(wo + wi);
                BSDFResponse response = Shading::BSDFs::GGX::evaluate_with_PDF(tint, alpha, wo, wi, halfway);
                EXPECT_COLOR_EQ_EPS(Shading::BSDFs::GGX::evaluate(tint, alpha, wo, wi, halfway), response.weight, make_float3(0.000000001f));
                EXPECT_FLOAT_EQ(Shading::BSDFs::GGX::PDF(alpha, wo, wi, halfway), response.PDF);
            }
        }
    }
}

GTEST_TEST(GGX, minimal_alpha) {
    using namespace optix;
        
    const float3 tint = make_float3(1.0f, 1.0f, 1.0f);
    const float alpha = 0.0f;

    const float3 incident_w = make_float3(0.0f, 0.0f, 1.0f);
    const float3 grazing_w = normalize(make_float3(0.0f, 1.0f, 0.001f));

    float3 f = Shading::BSDFs::GGX::evaluate(tint, 0.00000000001f, incident_w, incident_w, normalize(incident_w + incident_w));
    EXPECT_FALSE(isnan(f.x));

    f = Shading::BSDFs::GGX::evaluate(tint, alpha, grazing_w, incident_w, normalize(grazing_w + incident_w));
    EXPECT_FALSE(isnan(f.x));

    f = Shading::BSDFs::GGX::evaluate(tint, alpha, grazing_w, grazing_w, normalize(grazing_w + grazing_w));
    EXPECT_FALSE(isnan(f.x));

    const float3 grazing_wi = make_float3(grazing_w.x, -grazing_w.y, grazing_w.z);
    f = Shading::BSDFs::GGX::evaluate(tint, 0.00000000001f, grazing_w, grazing_wi, normalize(grazing_w + grazing_wi));
    EXPECT_FALSE(isnan(f.x));
}

GTEST_TEST(GGX, sampling_variance) {
    using namespace Shading::BSDFs;
    using namespace optix;

    // Test that sampling GGX with the visible normal distribution has lower variance than Walter's sampling.

    const int MAX_SAMPLES = 8196;
    const float3 wo = normalize(make_float3(1.0f, 0.0f, 1.0f));

    const float3 tint = make_float3(1.0f, 1.0f, 1.0f);
    const float ggx_alpha = 0.1f;

    for (float ggx_alpha = 0.1f; ggx_alpha < 1.0f; ggx_alpha += 0.2f) {
        // Original GGX sample strategy.
        auto original_GGX = Cogwheel::Math::Statistics(0, MAX_SAMPLES, [&](int i) -> float {
            BSDFSample sample = GGX::sample(tint, ggx_alpha, wo, RNG::sample02(i));
            if (is_PDF_valid(sample.PDF))
                return sample.weight.x * abs(sample.direction.z) / sample.PDF; // f * ||cos_theta|| / pdf
            else
                return 0.0f;
        });

        // Heitz et al GGX sampling strategy.
        auto GGX_with_VNDF = Cogwheel::Math::Statistics(0, MAX_SAMPLES, [&](int i) -> float {
            BSDFSample sample = GGXWithVNDF::sample(tint, ggx_alpha, wo, RNG::sample02(i));
            if (is_PDF_valid(sample.PDF)) {
                float w = sample.weight.x * abs(sample.direction.z) / sample.PDF; // f * ||cos_theta|| / pdf
                EXPECT_LT(w, 1.0f);
                return w;
            } else
                return 0.0f;
        });

        EXPECT_TRUE(almost_equal_eps(float(original_GGX.mean), float(GGX_with_VNDF.mean), 0.001f));
        EXPECT_LT(GGX_with_VNDF.variance, original_GGX.variance);
    }
}

} // NS OptiXRenderer

#endif // _OPTIXRENDERER_BSDFS_GGX_TEST_H_