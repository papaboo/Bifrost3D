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
        return Shading::BSDFs::GGX_R::PDF(m_alpha, wo, wi);
    }

    BSDFResponse evaluate_with_PDF(optix::float3 wo, optix::float3 wi) const {
        return Shading::BSDFs::GGX_R::evaluate_with_PDF(m_alpha, m_specularity, wo, wi);
    }

    BSDFSample sample(optix::float3 wo, optix::float3 random_sample) const {
        return Shading::BSDFs::GGX_R::sample(m_alpha, m_specularity, wo, optix::make_float2(random_sample));
    }

    std::string to_string() const {
        std::ostringstream out;
        out << "GGX reflection: alpha: " << m_alpha << ", specularity: " << m_specularity;
        return out.str();
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

GTEST_TEST(GGX_R, sampling_standard_deviation) {
    float expected_rho_std_dev = 0.685f;
    float alpha = 0.75f;
    auto ggx = GGXReflectionWrapper(alpha); 
    BSDFTestUtils::BSDF_sampling_variance_test(ggx, 1024, expected_rho_std_dev);
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
        optix::float3 reflected_wi = { -wo.x, -wo.y, wo.z };

        float estimated_PDF = GGX_R::PDF(estimated_alpha, wo, reflected_wi);

        // Shift alpha towards the correct PDF by the max_alpha_error.
        // If the estimated PDF is lower than the max PDF, then the alpha needs to be reduced (the peak increased),
        // otherwise the alpha should be increased (blurrier reflection).
        float alpha_step_size = max_alpha_error * (estimated_PDF < max_PDF ? -1 : 1);
        float shifted_alpha = estimated_alpha + alpha_step_size;
        shifted_alpha = optix::clamp(shifted_alpha, 0.0f, 1.0f);
        float shifted_PDF = GGX_R::PDF(shifted_alpha, wo, reflected_wi);

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

// ---------------------------------------------------------------------------
// GGX transmission tests.
// ---------------------------------------------------------------------------

class GGXTransmissionWrapper {
public:
    float m_alpha;
    float m_ior_i_over_o;

    GGXTransmissionWrapper(float alpha, float ior_i_over_o)
        : m_alpha(alpha), m_ior_i_over_o(ior_i_over_o) {}

    optix::float3 evaluate(optix::float3 wo, optix::float3 wi) const {
        return optix::make_float3(1) * Shading::BSDFs::GGX_T::evaluate(m_alpha, m_ior_i_over_o, wo, wi);
    }

    float PDF(optix::float3 wo, optix::float3 wi) const {
        return Shading::BSDFs::GGX_T::PDF(m_alpha, m_ior_i_over_o, wo, wi);
    }

    BSDFResponse evaluate_with_PDF(optix::float3 wo, optix::float3 wi) const {
        return Shading::BSDFs::GGX_T::evaluate_with_PDF(m_alpha, m_ior_i_over_o, wo, wi);
    }

    BSDFSample sample(optix::float3 wo, optix::float3 random_sample) const {
        return Shading::BSDFs::GGX_T::sample(m_alpha, m_ior_i_over_o, wo, optix::make_float2(random_sample));
    }

    std::string to_string() const {
        std::ostringstream out;
        out << "GGX transmission: alpha: " << m_alpha << ", ior_i / ior_o: " << m_ior_i_over_o;
        return out.str();
    }
};

GTEST_TEST(GGX_T, power_conservation) {
    for (float ior_i_over_o : { 0.5f, 0.9f, 1.1f, 1.5f })
        for (float cos_theta : { -1.0f, -0.7f, -0.4f, -0.1f, 0.1f, 0.4f, 0.7f, 1.0f }) {
            optix::float3 wo = { sqrt(1 - pow2(cos_theta)), 0.0f, cos_theta };
            for (float alpha : { 0.0f, 0.0675f, 0.125f, 0.25f, 0.5f, 1.0f }) {
                auto ggx = GGXTransmissionWrapper(alpha, ior_i_over_o);
                auto res = BSDFTestUtils::directional_hemispherical_reflectance_function(ggx, wo, 1024u);
                EXPECT_LE(res.reflectance, 1.0f);
            }
        }
}

// GTEST_TEST(GGX_T, Helmholtz_reciprocity) {
//     for (float ior_i_over_o : { 0.5f, 0.9f, 1.1f, 1.5f })
//         for (float cos_theta : { -1.0f, -0.4f, -0.1f, 0.1f, 0.4f, 1.0f }) {
//             optix::float3 wo = { sqrt(1 - pow2(cos_theta)), 0.0f, cos_theta };
//             for (float alpha : { 0.0675f, 0.125f, 0.25f, 0.5f, 1.0f }) {
//                 auto ggx = GGXTransmissionWrapper(alpha, 0.0f, ior_i_over_o);
//                 BSDFTestUtils::helmholtz_reciprocity(ggx, wo, 16u);
//             }
//         }
// }

GTEST_TEST(GGX_T, function_consistency) {
    for (float ior_i_over_o : { 0.5f, 0.9f, 1.1f, 1.5f })
        for (float cos_theta : { -1.0f, -0.4f, -0.1f, 0.1f, 0.4f, 1.0f }) {
            optix::float3 wo = { sqrt(1 - pow2(cos_theta)), 0.0f, cos_theta };
            for (float alpha : { 0.0675f, 0.125f, 0.25f, 0.5f, 1.0f }) {
                auto ggx = GGXTransmissionWrapper(alpha, ior_i_over_o);
                BSDFTestUtils::BSDF_consistency_test(ggx, wo, 16u);
            }
        }
}

GTEST_TEST(GGX_T, sampling_standard_deviation) {
    float alpha = 0.75f;
    float ior_i_over_os[] = { 0.5f, 0.9f, 1.1f, 1.5f };
    float expected_rho_std_devs[] = { 2.05f, 0.53f, 0.05f, 0.08f };
    for (int i = 0; i < 4; i++) {
        auto ggx = GGXTransmissionWrapper(alpha, ior_i_over_os[i]);
        BSDFTestUtils::BSDF_sampling_variance_test(ggx, 1024, expected_rho_std_devs[i]);
    }
}

GTEST_TEST(GGX_T, consistent_sampling_across_hemispheres) {
    optix::float3 random_sample = { 0.5f, 0.5f, 0.5f };
    for (float cos_theta : { -1.0f, -0.4f, -0.1f, 0.1f, 0.4f, 1.0f }) {
        optix::float3 positive_wo = BSDFTestUtils::wo_from_cos_theta(cos_theta);
        optix::float3 negative_wo = { positive_wo.x, positive_wo.y, -positive_wo.z };
        for (float alpha : { 0.0675f, 0.125f, 0.25f, 0.5f, 1.0f }) {
            for (float ior_i_over_o : { 0.5f, 0.9f, 1.1f, 1.5f }) {
                auto ggx = GGXTransmissionWrapper(alpha, ior_i_over_o);
                auto positive_sample = ggx.sample(positive_wo, random_sample);
                auto negative_sample = ggx.sample(negative_wo, random_sample);
                EXPECT_EQ(positive_sample.PDF, negative_sample.PDF);
                EXPECT_FLOAT3_EQ(positive_sample.reflectance, negative_sample.reflectance);

                optix::float3 flipped_negative_direction = negative_sample.direction;
                flipped_negative_direction.z = -flipped_negative_direction.z;
                EXPECT_FLOAT3_EQ(positive_sample.direction, flipped_negative_direction);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Full GGX with reflection and transmission tests.
// ---------------------------------------------------------------------------

class GGXWrapper {
public:
    float m_alpha;
    float m_specularity;
    float m_ior_i_over_o;
    optix::float3 m_tint;

    GGXWrapper(float alpha, float specularity, float ior_i_over_o, optix::float3 tint = optix::make_float3(1))
        : m_alpha(alpha), m_specularity(specularity), m_ior_i_over_o(ior_i_over_o), m_tint(tint) {}

    optix::float3 evaluate(optix::float3 wo, optix::float3 wi) const {
        return Shading::BSDFs::GGX::evaluate(m_tint, m_alpha, m_specularity, m_ior_i_over_o, wo, wi);
    }

    float PDF(optix::float3 wo, optix::float3 wi) const {
        return Shading::BSDFs::GGX::PDF(m_alpha, m_specularity, m_ior_i_over_o, wo, wi);
    }

    BSDFResponse evaluate_with_PDF(optix::float3 wo, optix::float3 wi) const {
        return Shading::BSDFs::GGX::evaluate_with_PDF(m_tint, m_alpha, m_specularity, m_ior_i_over_o, wo, wi);
    }

    BSDFSample sample(optix::float3 wo, optix::float3 random_sample) const {
        return Shading::BSDFs::GGX::sample(m_tint, m_alpha, m_specularity, m_ior_i_over_o, wo, random_sample);
    }

    std::string to_string() const {
        std::ostringstream out;
        out << "GGX: alpha: " << m_alpha << ", specularity: " << m_specularity << ", ior_i / ior_o: " << m_ior_i_over_o;
        return out.str();
    }
};

GTEST_TEST(GGX, power_conservation) {
    for (float ior_i_over_o : { 0.5f, 0.9f, 1.1f, 1.5f })
        for (float cos_theta : { -1.0f, -0.7f, -0.4f, -0.1f, 0.1f, 0.4f, 0.7f, 1.0f }) {
            optix::float3 wo = { sqrt(1 - pow2(cos_theta)), 0.0f, cos_theta };
            for (float alpha : { 0.0f, 0.0675f, 0.125f, 0.25f, 0.5f, 1.0f }) {
                auto ggx = GGXWrapper(alpha, 0.04f, ior_i_over_o);
                auto res = BSDFTestUtils::directional_hemispherical_reflectance_function(ggx, wo, 1024u);
                EXPECT_LE(res.reflectance, 1.0f);
            }
        }
}

GTEST_TEST(GGX, function_consistency) {
    for (float ior_i_over_o : { 0.5f, 0.9f, 1.1f, 1.5f })
        for (float cos_theta : { -1.0f, -0.4f, -0.1f, 0.1f, 0.4f, 1.0f }) {
            optix::float3 wo = { sqrt(1 - pow2(cos_theta)), 0.0f, cos_theta };
            for (float alpha : { 0.0675f, 0.25f, 1.0f }) {
                auto ggx = GGXWrapper(alpha, 0.04f, ior_i_over_o);
                BSDFTestUtils::BSDF_consistency_test(ggx, wo, 16u);
            }
        }
}

GTEST_TEST(GGX, reflection_response_equals_GGX_R) {
    using namespace optix;

    int max_sample_count = 16;
    float fully_specular = 1.0f; // Disable transmission
    float ior_i_over_o = 1.5f;

    for (float cos_theta : { 0.2f, 1.0f }) {
        const float3 wo = { sqrt(1 - pow2(cos_theta)), 0.0f, cos_theta };
        for (float alpha : { 0.0675f, 0.25f, 1.0f }) {
            for (int i = 0; i < max_sample_count; ++i) {
                float3 rng_sample = make_float3(RNG::sample02(i), (i + 0.5f) / max_sample_count);
                BSDFSample ggx_sample = Shading::BSDFs::GGX::sample(alpha, fully_specular, ior_i_over_o, wo, rng_sample);
                if (is_PDF_valid(ggx_sample.PDF)) {
                    BSDFResponse ggx_r_response = Shading::BSDFs::GGX_R::evaluate_with_PDF(alpha, fully_specular, wo, ggx_sample.direction);
                    EXPECT_COLOR_EQ_PCT(ggx_sample.reflectance, ggx_r_response.reflectance, make_float3(0.00002f));
                    EXPECT_FLOAT_EQ_PCT(ggx_sample.PDF, ggx_r_response.PDF, 0.00002f);
                }
            }
        }
    }
}

GTEST_TEST(GGX, transmission_response_equals_GGX_T) {
    using namespace optix;

    int max_sample_count = 16;
    float specularity = 0.0f;
    float ior_i_over_o = 1.5f;

    RNG::LinearCongruential rng; rng.seed(73856093);
    for (float ior_i_over_o : { 0.5f, 1.5f }) {
        for (float cos_theta : { 0.4f, 1.0f }) {
            const float3 wo = { sqrt(1 - pow2(cos_theta)), 0.0f, cos_theta };
            for (float alpha : { 0.0675f, 0.25f, 1.0f }) {
                int i = 0;
                while (i < max_sample_count) {
                    BSDFSample ggx_sample = Shading::BSDFs::GGX::sample(alpha, specularity, ior_i_over_o, wo, rng.sample3f());
                    bool is_transmission = sign(ggx_sample.direction.z) != sign(cos_theta);
                    if (is_transmission && is_PDF_valid(ggx_sample.PDF)) {
                        BSDFResponse ggx_t_response = Shading::BSDFs::GGX_T::evaluate_with_PDF(alpha, ior_i_over_o, wo, ggx_sample.direction);

                        // GGX_T ignores Fresnel, so we need to add it explicitly.
                        // We assume that GGX uses Schlick's Fresnel approximation.
                        float3 halfway = Shading::BSDFs::GGX_T::compute_halfway_vector(ior_i_over_o, wo, ggx_sample.direction);
                        float transmission_propability = 1.0f - schlick_fresnel(specularity, dot(wo, halfway));
                        ggx_t_response.reflectance *= transmission_propability;
                        ggx_t_response.PDF *= transmission_propability;

                        EXPECT_COLOR_EQ_PCT(ggx_sample.reflectance, ggx_t_response.reflectance, make_float3(0.00002f));
                        EXPECT_FLOAT_EQ_PCT(ggx_sample.PDF, ggx_t_response.PDF, 0.00002f);
                        ++i;
                    }
                }
            }
        }
    }
}

GTEST_TEST(GGX, sample_according_to_specularity) {
    using namespace optix;

    // A black tint zeroes out the transmission reflectance, while the reflection is still white.
    // The reflectance should therefore equal the reflection to transmission ratio.
    float3 black = { 0, 0, 0 };
    float roughness = 0.0f;
    float alpha = Shading::BSDFs::GGX::alpha_from_roughness(roughness);

    for (float cos_theta : { -1.0f, 1.0f }) {
        float3 wo = { 0, 0, cos_theta };
        for (float specularity : { 0.0f, 0.5f, 1.0f }) {
            for (float ior_i_over_o : { 0.5f, 1.5f }) {
                auto ggx = GGXWrapper(alpha, specularity, ior_i_over_o, black);
                auto res = BSDFTestUtils::directional_hemispherical_reflectance_function(ggx, wo, 1024u);
                EXPECT_FLOAT_EQ_EPS(specularity, res.reflectance, 0.00001f) << "alpha: " << alpha << ", cos_theta: " << cos_theta << ", specularity: " << specularity;
            }
        }
    }
}

GTEST_TEST(GGX, sampling_standard_deviation) {
    float alpha = 0.75f;
    float specularity = 0.5f;
    float ior_i_over_os[] = { 0.5f, 0.9f, 1.1f, 1.5f };
    float expected_rho_std_devs[] = { 1.07f, 0.61f, 0.46f, 0.46f };
    for (int i = 0; i < 4; i++) {
        auto ggx = GGXWrapper(alpha, specularity, ior_i_over_os[i]);
        BSDFTestUtils::BSDF_sampling_variance_test(ggx, 1024, expected_rho_std_devs[i]);
    }
}

} // NS OptiXRenderer

#endif // _OPTIXRENDERER_BSDFS_GGX_TEST_H_