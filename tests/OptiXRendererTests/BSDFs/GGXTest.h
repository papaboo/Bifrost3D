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

#include <Bifrost/Math/Statistics.h>
#include <Bifrost/Math/Utils.h>

#include <OptiXRenderer/RNG.h>
#include <OptiXRenderer/Shading/BSDFs/GGX.h>
#include <OptiXRenderer/Utils.h>

#include <gtest/gtest.h>

namespace OptiXRenderer {

GTEST_TEST(GGX, power_conservation) {
    using namespace optix;

    const unsigned int MAX_SAMPLES = 1024u;
    const float full_specularity = 1.0f;

    for (float cos_theta : { 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f }) {
        const float3 wo = { sqrt(1 - pow2(cos_theta)), 0.0f, cos_theta };
        for (float alpha : { 0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f }) {
            float ws[MAX_SAMPLES];
            for (unsigned int i = 0u; i < MAX_SAMPLES; ++i) {
                BSDFSample sample = Shading::BSDFs::GGX::sample(alpha, full_specularity, wo, RNG::sample02(i));

                if (is_PDF_valid(sample.PDF))
                    ws[i] = sample.weight.x * sample.direction.z / sample.PDF; // f * ||cos_theta|| / pdf
                else
                    ws[i] = 0.0f;
            }

            float average_w = Bifrost::Math::sort_and_pairwise_summation(ws, ws + MAX_SAMPLES) / float(MAX_SAMPLES);
            EXPECT_LE(average_w, 1.0f);
        }
    }
}

GTEST_TEST(GGX, Helmholtz_reciprocity) {
    using namespace optix;

    const unsigned int MAX_SAMPLES = 128u;
    const float3 wo = normalize(make_float3(1.0f, 1.0f, 1.0f));
    const float full_specularity = 1.0f;

    for (float alpha : { 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f }) {
        for (unsigned int i = 0u; i < MAX_SAMPLES; ++i) {
            BSDFSample sample = Shading::BSDFs::GGX::sample(alpha, full_specularity, wo, RNG::sample02(i));

            if (is_PDF_valid(sample.PDF)) {
                float3 f = Shading::BSDFs::GGX::evaluate(alpha, make_float3(full_specularity), sample.direction, wo);
                EXPECT_COLOR_EQ_EPS(sample.weight, f, make_float3(0.0001f));
            }
        }
    }
}

GTEST_TEST(GGX, consistent_PDF) {
    using namespace optix;

    const unsigned int MAX_SAMPLES = 128u;
    const float3 wo = normalize(make_float3(1.0f, 1.0f, 1.0f));
    const float full_specularity = 1.0f;

    for (float alpha : { 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f }) {
        for (unsigned int i = 0u; i < MAX_SAMPLES; ++i) {
            BSDFSample sample = Shading::BSDFs::GGX::sample(alpha, full_specularity, wo, RNG::sample02(i));
            if (is_PDF_valid(sample.PDF)) {
                float3 wi = sample.direction;
                float PDF = Shading::BSDFs::GGX::PDF(alpha, wo, normalize(wo + wi));
                EXPECT_FLOAT_EQ_EPS(sample.PDF, PDF, 0.0001f);
            }
        }
    }
}

GTEST_TEST(GGX, evaluate_with_PDF) {
    using namespace optix;

    const unsigned int MAX_SAMPLES = 128u;
    const float3 wo = normalize(make_float3(1.0f, 1.0f, 1.0f));
    const float full_specularity = 1.0f;

    for (float alpha : { 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f }) {
        for (unsigned int i = 0u; i < MAX_SAMPLES; ++i) {
            BSDFSample sample = Shading::BSDFs::GGX::sample(alpha, full_specularity, wo, RNG::sample02(i));

            if (is_PDF_valid(sample.PDF)) {
                float3 wi = sample.direction;
                float3 halfway = normalize(wo + wi);
                BSDFResponse response = Shading::BSDFs::GGX::evaluate_with_PDF(alpha, full_specularity, wo, wi);
                const float3 f = make_float3(Shading::BSDFs::GGX::evaluate(alpha, full_specularity, wo, wi));
                EXPECT_COLOR_EQ_EPS(f, response.weight, make_float3(0.000001f));
                EXPECT_FLOAT_EQ(Shading::BSDFs::GGX::PDF(alpha, wo, halfway), response.PDF);
            }
        }
    }
}

GTEST_TEST(GGX, minimal_alpha) {
    using namespace optix;
        
    const float alpha = 0.0f;
    const float full_specularity = 1.0f;

    const float3 incident_w = make_float3(0.0f, 0.0f, 1.0f);
    const float3 grazing_w = normalize(make_float3(0.0f, 1.0f, 0.001f));

    float f = Shading::BSDFs::GGX::evaluate(0.00000000001f, full_specularity, incident_w, incident_w);
    EXPECT_FALSE(isnan(f));

    f = Shading::BSDFs::GGX::evaluate(alpha, full_specularity, grazing_w, incident_w);
    EXPECT_FALSE(isnan(f));

    f = Shading::BSDFs::GGX::evaluate(alpha, full_specularity, grazing_w, grazing_w);
    EXPECT_FALSE(isnan(f));

    const float3 grazing_wi = make_float3(grazing_w.x, -grazing_w.y, grazing_w.z);
    f = Shading::BSDFs::GGX::evaluate(0.00000000001f, full_specularity, grazing_w, grazing_wi);
    EXPECT_FALSE(isnan(f));
}

// Test that sampling from the distribution of visible normals has lower variance than the original GGX sampling and they converge to the same result.
GTEST_TEST(GGX, sampling_variance) {
    using namespace optix;

    static auto sample_GGX_vanilla = [](float alpha, const float3& specularity, const float3& wo, float2 random_sample) -> BSDFSample {
        BSDFSample bsdf_sample;

        Distributions::DirectionalSample ggx_sample = Distributions::GGX::sample(alpha, random_sample);
        bsdf_sample.direction = reflect(-wo, ggx_sample.direction);

        bsdf_sample.PDF = ggx_sample.PDF / (4.0f * dot(wo, ggx_sample.direction));
        bsdf_sample.weight = Shading::BSDFs::GGX::evaluate(alpha, specularity, wo, bsdf_sample.direction, ggx_sample.direction);

        bool discardSample = bsdf_sample.PDF < 0.00001f || bsdf_sample.direction.z < 0.00001f; // Discard samples if the pdf is too low (precision issues) or if the new direction points into the surface (energy loss).
        return discardSample ? BSDFSample::none() : bsdf_sample;
    };

    typedef BSDFSample(*SampleFunction)(float alpha, const float3& specularity, const float3& wo, float2 random_sample);
    auto sampling_mean_and_variance = [&](SampleFunction function, float alpha, float3 wo, double& mean, double& variance) {
        const float3 specularity = { 1, 1, 1 };

        const unsigned int MAX_SAMPLES = 4098;
        auto ws = std::vector<double>(MAX_SAMPLES);
        auto ws_squared = std::vector<double>(MAX_SAMPLES);

        for (unsigned int i = 0u; i < MAX_SAMPLES; ++i) {
            BSDFSample sample = function(alpha, specularity, wo, RNG::sample02(i));
            if (is_PDF_valid(sample.PDF)) {
                ws[i] = sample.weight.x * abs(sample.direction.z) / sample.PDF; // f * ||cos_theta|| / pdf
                ws_squared[i] = ws[i] * ws[i];
            }
            else
                ws_squared[i] = ws[i] = 0.0f;
        }

        mean = Bifrost::Math::sort_and_pairwise_summation(ws.begin(), ws.end()) / MAX_SAMPLES;
        double mean_squared = Bifrost::Math::sort_and_pairwise_summation(ws_squared.begin(), ws_squared.end()) / MAX_SAMPLES;
        variance = mean_squared - mean * mean;
    };

    const float3 wo = normalize(make_float3(1.0f, 0.0f, 1.0f));
    for (float alpha : { 0.125f, 0.25f, 0.5f, 1.0f }) {
        double vanilla_GGX_mean, vanilla_GGX_variance;
        sampling_mean_and_variance(sample_GGX_vanilla, alpha, wo, vanilla_GGX_mean, vanilla_GGX_variance);

        double GGX_VNDF_mean, GGX_VNDF_variance;
        sampling_mean_and_variance(Shading::BSDFs::GGX::sample, alpha, wo, GGX_VNDF_mean, GGX_VNDF_variance);

        EXPECT_TRUE(almost_equal_eps(float(vanilla_GGX_mean), float(GGX_VNDF_mean), 0.001f));
        EXPECT_LT(GGX_VNDF_variance, vanilla_GGX_variance);
    }
}

} // NS OptiXRenderer

#endif // _OPTIXRENDERER_BSDFS_GGX_TEST_H_