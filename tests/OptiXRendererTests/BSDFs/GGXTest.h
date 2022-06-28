// Test OptiXRenderer's GGX distribution and BSDF.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _OPTIXRENDERER_BSDFS_GGX_TEST_H_
#define _OPTIXRENDERER_BSDFS_GGX_TEST_H_

#include <Utils.h>

#include <Bifrost/Assets/Shading/Fittings.h>
#include <Bifrost/Math/RNG.h>
#include <Bifrost/Math/Statistics.h>
#include <Bifrost/Math/Utils.h>

#include <OptiXRenderer/RNG.h>
#include <OptiXRenderer/Shading/BSDFs/GGX.h>
#include <OptiXRenderer/Utils.h>

#include <gtest/gtest.h>

namespace OptiXRenderer {

GTEST_TEST(GGX_R, power_conservation) {
    using namespace optix;

    const unsigned int MAX_SAMPLES = 1024u;
    const float full_specularity = 1.0f;

    for (float cos_theta : { 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f }) {
        const float3 wo = { sqrt(1 - pow2(cos_theta)), 0.0f, cos_theta };
        for (float alpha : { 0.0f, 0.0675f, 0.125f, 0.25f, 0.5f, 1.0f }) {
            float ws[MAX_SAMPLES];
            for (unsigned int i = 0u; i < MAX_SAMPLES; ++i) {
                BSDFSample sample = Shading::BSDFs::GGX_R::sample(alpha, full_specularity, wo, RNG::sample02(i));

                if (is_PDF_valid(sample.PDF))
                    ws[i] = sample.reflectance.x * sample.direction.z / sample.PDF; // f * ||cos_theta|| / pdf
                else
                    ws[i] = 0.0f;
            }

            float average_w = Bifrost::Math::sort_and_pairwise_summation(ws, ws + MAX_SAMPLES) / float(MAX_SAMPLES);
            EXPECT_LE(average_w, 1.0f);
        }
    }
}

GTEST_TEST(GGX_R, Helmholtz_reciprocity) {
    using namespace optix;

    const unsigned int MAX_SAMPLES = 128u;
    const float3 wo = normalize(make_float3(1.0f, 1.0f, 1.0f));
    const float full_specularity = 1.0f;

    for (float alpha : { 0.0675f, 0.125f, 0.25f, 0.5f, 1.0f }) {
        for (unsigned int i = 0u; i < MAX_SAMPLES; ++i) {
            BSDFSample sample = Shading::BSDFs::GGX_R::sample(alpha, full_specularity, wo, RNG::sample02(i));

            if (is_PDF_valid(sample.PDF)) {
                float3 f = Shading::BSDFs::GGX_R::evaluate(alpha, make_float3(full_specularity), sample.direction, wo);
                EXPECT_COLOR_EQ_EPS(sample.reflectance, f, make_float3(0.0001f));
            }
        }
    }
}

GTEST_TEST(GGX_R, consistent_PDF) {
    using namespace optix;

    const unsigned int MAX_SAMPLES = 128u;
    const float3 wo = normalize(make_float3(1.0f, 1.0f, 1.0f));
    const float full_specularity = 1.0f;

    for (float alpha : { 0.0675f, 0.125f, 0.25f, 0.5f, 1.0f }) {
        for (unsigned int i = 0u; i < MAX_SAMPLES; ++i) {
            BSDFSample sample = Shading::BSDFs::GGX_R::sample(alpha, full_specularity, wo, RNG::sample02(i));
            if (is_PDF_valid(sample.PDF)) {
                float3 wi = sample.direction;
                float PDF = Shading::BSDFs::GGX_R::PDF(alpha, wo, normalize(wo + wi));
                EXPECT_FLOAT_EQ_EPS(sample.PDF, PDF, 0.0001f);
            }
        }
    }
}

GTEST_TEST(GGX_R, evaluate_with_PDF) {
    using namespace optix;

    const unsigned int MAX_SAMPLES = 128u;
    const float3 wo = normalize(make_float3(1.0f, 1.0f, 1.0f));
    const float3 full_specularity = { 1, 1, 1 };

    for (float alpha : { 0.0675f, 0.125f, 0.25f, 0.5f, 1.0f }) {
        for (unsigned int i = 0u; i < MAX_SAMPLES; ++i) {
            BSDFSample sample = Shading::BSDFs::GGX_R::sample(alpha, full_specularity, wo, RNG::sample02(i));

            if (is_PDF_valid(sample.PDF)) {
                float3 wi = sample.direction;
                float3 halfway = normalize(wo + wi);
                BSDFResponse response = Shading::BSDFs::GGX_R::evaluate_with_PDF(alpha, full_specularity, wo, wi, halfway);
                EXPECT_FLOAT_EQ(Shading::BSDFs::GGX_R::evaluate(alpha, full_specularity, wo, wi, halfway).x, response.reflectance.x);
                EXPECT_FLOAT_EQ(Shading::BSDFs::GGX_R::PDF(alpha, wo, halfway), response.PDF);
            }
        }
    }
}

GTEST_TEST(GGX_R, minimal_alpha) {
    using namespace optix;
        
    const float min_alpha = 0.00000000001f;
    const float full_specularity = 1.0f;

    const float3 incident_w = make_float3(0.0f, 0.0f, 1.0f);
    const float3 grazing_w = normalize(make_float3(0.0f, 1.0f, 0.001f));

    float f = Shading::BSDFs::GGX_R::evaluate(min_alpha, full_specularity, incident_w, incident_w);
    EXPECT_FALSE(isnan(f));

    f = Shading::BSDFs::GGX_R::evaluate(min_alpha, full_specularity, grazing_w, incident_w);
    EXPECT_FALSE(isnan(f));

    f = Shading::BSDFs::GGX_R::evaluate(min_alpha, full_specularity, grazing_w, grazing_w);
    EXPECT_FALSE(isnan(f));

    const float3 grazing_wi = make_float3(grazing_w.x, -grazing_w.y, grazing_w.z);
    f = Shading::BSDFs::GGX_R::evaluate(min_alpha, full_specularity, grazing_w, grazing_wi);
    EXPECT_FALSE(isnan(f));
}

GTEST_TEST(GGX_R, estimate_alpha_from_max_PDF) {
    using namespace Bifrost;
    using namespace Bifrost::Assets::Shading::EstimateGGXAlpha;
    using namespace Shading::BSDFs;

    const int sample_count = 16;
    const float max_alpha_error = 1.0f / max_PDF_sample_count;
    Math::Vector2f samples[sample_count];
    Math::RNG::fill_progressive_multijittered_bluenoise_samples(samples, samples + sample_count, 4);

    for (auto sample : samples) {
        float cos_theta = sample.x;
        float max_PDF = decode_PDF(sample.y);
        float estimated_alpha = estimate_alpha(cos_theta, max_PDF);

        optix::float3 wo = { sqrt(1 - pow2(cos_theta)), 0.0f, cos_theta };
        optix::float3 halfway = { 0, 0, 1 };

        float estimated_PDF = GGX_R::PDF(estimated_alpha, wo, halfway);

        // Shift alpha towards the correct PDF by the max_alpha_error.
        // If the estimated PDF is lower than the max PDF, then the alpha needs to be reduced (the peak increased),
        // otherwise the alpha should be increased (blurrier reflection).
        float alpha_step_size = max_alpha_error * (estimated_PDF < max_PDF ? -1 : 1);
        float shifted_alpha = estimated_alpha + alpha_step_size;
        shifted_alpha = Math::clamp(shifted_alpha, 0.0f, 1.0f);
        float shifted_PDF = GGX_R::PDF(shifted_alpha, wo, halfway);

        // Wether the max PDF is found somewhere between the estimated PDF and the shifted PDF,
        // i.e. the correct alpha is between the estimated alpha and the shifted alpha.
        bool passed_correct_alpha = (estimated_PDF <= max_PDF && max_PDF <= shifted_PDF) ||
                                    (shifted_PDF <= max_PDF && max_PDF <= estimated_PDF);
        // Not all max PDFs are possible when alpha is limited to the range [0, 1]. Discard those invalid samples.
        bool invalid_max_PDF = (shifted_alpha == 0.0f && shifted_PDF < max_PDF) ||
                               (shifted_alpha == 1.0f && max_PDF < shifted_PDF);

        EXPECT_TRUE(passed_correct_alpha || invalid_max_PDF);
    }
}

} // NS OptiXRenderer

#endif // _OPTIXRENDERER_BSDFS_GGX_TEST_H_