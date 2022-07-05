// Test OptiXRenderer's GGX distribution and BSDF.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _OPTIXRENDERER_BSDFS_GGX_TEST_H_
#define _OPTIXRENDERER_BSDFS_GGX_TEST_H_

#include <BSDFTestUtils.h>

#include <Bifrost/Assets/Shading/Fittings.h>

#include <OptiXRenderer/Shading/BSDFs/GGX.h>
#include <OptiXRenderer/RNG.h>

#include <gtest/gtest.h>

namespace OptiXRenderer {

// ---------------------------------------------------------------------------
// GGX reflection tests. Specularity is always set to 1 to have full reflection.
// ---------------------------------------------------------------------------

class GGXReflectionWrapper {
public:
    float m_alpha;
    float m_specularity;

    GGXReflectionWrapper(float alpha)
        : m_alpha(alpha), m_specularity(1.0f) {}

    optix::float3 evaluate(optix::float3 wo, optix::float3 wi) const {
        return optix::make_float3(1) * Shading::BSDFs::GGX_R::evaluate(m_alpha, m_specularity, wo, wi);
    }

    float PDF(optix::float3 wo, optix::float3 wi) const {
        optix::float3 halfway = optix::normalize(wo + wi);
        return Shading::BSDFs::GGX_R::PDF(m_alpha, wo, halfway);
    }

    BSDFResponse evaluate_with_PDF(optix::float3 wo, optix::float3 wi) const {
        return Shading::BSDFs::GGX_R::evaluate_with_PDF(m_alpha, m_specularity, wo, wi);
    }

    BSDFSample sample(optix::float3 wo, optix::float3 random_sample) const {
        return Shading::BSDFs::GGX_R::sample(m_alpha, m_specularity, wo, optix::make_float2(random_sample));
    }
};

GTEST_TEST(GGX_R, power_conservation) {
    for (float cos_theta : { 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f }) {
        optix::float3 wo = { sqrt(1 - pow2(cos_theta)), 0.0f, cos_theta };
        for (float alpha : { 0.0f, 0.0675f, 0.125f, 0.25f, 0.5f, 1.0f }) {
            auto ggx = GGXReflectionWrapper(alpha);
            auto res = BSDFTestUtils::directional_hemispherical_reflectance_function(ggx, wo, 1024u);
            EXPECT_LE(res.reflectance, 1.0f);
        }
    }
}

GTEST_TEST(GGX_R, Helmholtz_reciprocity) {
    optix::float3 wo = optix::normalize(optix::make_float3(1.0f, 1.0f, 1.0f));
    for (float alpha : { 0.0675f, 0.125f, 0.25f, 0.5f, 1.0f }) {
        auto ggx = GGXReflectionWrapper(alpha);
        BSDFTestUtils::helmholtz_reciprocity(ggx, wo, 16u);
    }
}

GTEST_TEST(GGX_R, function_consistency) {
    optix::float3 wo = optix::normalize(optix::make_float3(1.0f, 1.0f, 1.0f));
    for (float alpha : { 0.0675f, 0.125f, 0.25f, 0.5f, 1.0f }) {
        auto ggx = GGXReflectionWrapper(alpha);
        BSDFTestUtils::BSDF_consistency_test(ggx, wo, 16u);
    }
}

GTEST_TEST(GGX_R, minimal_alpha) {
    using namespace optix;

    const float min_alpha = Shading::BSDFs::GGX::alpha_from_roughness(0.0f);
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
    using namespace Bifrost::Assets::Shading::EstimateGGXAlpha;
    using namespace Shading::BSDFs;

    const int sample_count = 16;
    const float max_alpha_error = 1.0f / max_PDF_sample_count;
    
    for (int i = 0; i < sample_count; i++) {
        optix::float2 sample = RNG::sample02(i);
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
        shifted_alpha = optix::clamp(shifted_alpha, 0.0f, 1.0f);
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