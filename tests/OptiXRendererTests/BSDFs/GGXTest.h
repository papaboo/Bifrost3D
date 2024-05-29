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

    GGXReflectionWrapper(float alpha, float specularity = 1.0f)
        : m_alpha(alpha), m_specularity(specularity) {}

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
            EXPECT_FLOAT3_LE(res.reflectance, 1.0f);
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
    float expected_rho_std_dev = 0.36f;
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

GTEST_TEST(GGX_R, fully_grazing_evaluates_to_black) {
    using namespace optix;

    const float3 incident_w = make_float3(0.0f, 0.0f, 1.0f);
    const float3 grazing_w = make_float3(0.0f, 1.0f, 0.0f);

    for (float alpha : { 0.0f, 0.5f, 1.0f }) {
        auto ggx = GGXReflectionWrapper(alpha);

        float grazing_wo_f = ggx.evaluate(grazing_w, incident_w).x;
        EXPECT_FLOAT_EQ(grazing_wo_f, 0.0f);

        float grazing_wi_f = ggx.evaluate(incident_w, grazing_w).x;
        EXPECT_FLOAT_EQ(grazing_wi_f, 0.0f);

        float both_grazing_f = ggx.evaluate(grazing_w, grazing_w).x;
        EXPECT_FLOAT_EQ(both_grazing_f, 0.0f) << ggx.to_string();
    }
}

// Validate that a few select samples in the GGX and GGX with Fresnel precomputed Rho tables have the correct values and can be looked up correct.
// We test the corner and middle samples and make sure that the sample coordinates match with the original precomputed sample coords.
GTEST_TEST(GGX_R, validate_ggx_rho_precomputations) {
    using namespace Bifrost::Assets::Shading;

    int sample_count = 4096;

    float middle_cos_theta = (Rho::GGX_angle_sample_count / 2) / (Rho::GGX_angle_sample_count - 1.0f);
    float middle_roughness = (Rho::GGX_roughness_sample_count / 2) / (Rho::GGX_roughness_sample_count - 1.0f);

    for (float cos_theta : { 0.000001f, middle_cos_theta, 1.0f }) {
        optix::float3 wo = BSDFTestUtils::wo_from_cos_theta(cos_theta);
        for (float roughness : { 0.0f, middle_roughness, 1.0f }) {
            float alpha = Shading::BSDFs::GGX::alpha_from_roughness(roughness);

            auto no_specularity_ggx = GGXReflectionWrapper(alpha, 0.0f);
            float expected_no_specularity_rho = BSDFTestUtils::directional_hemispherical_reflectance_function(no_specularity_ggx, wo, sample_count).reflectance.x;
            float actual_no_specularity_rho = Rho::sample_GGX_with_fresnel(cos_theta, roughness);
            EXPECT_FLOAT_EQ_EPS(expected_no_specularity_rho, actual_no_specularity_rho, 0.0001f) << "for cos_theta: " << cos_theta << " and roughness: " << roughness;

            auto full_specularity_ggx = GGXReflectionWrapper(alpha, 1.0f);
            float expected_full_specularity_rho = BSDFTestUtils::directional_hemispherical_reflectance_function(full_specularity_ggx, wo, sample_count).reflectance.x;
            float actual_full_specularity_rho = Rho::sample_GGX(cos_theta, roughness);
            EXPECT_FLOAT_EQ_EPS(expected_full_specularity_rho, actual_full_specularity_rho, 0.0001f) << "for cos_theta: " << cos_theta << " and roughness: " << roughness;
        }
    }
}

GTEST_TEST(GGX_R, estimate_bounded_VNDF_alpha_from_max_PDF) {
    using namespace Bifrost::Assets::Shading;
    using namespace Shading::BSDFs;

    const int sample_count = 16;
    const float max_alpha_error = 1.0f / Estimate_GGX_bounded_VNDF_alpha::max_PDF_sample_count;

    for (int i = 0; i < sample_count; i++) {
        optix::float2 sample = RNG::sample02(i);
        float cos_theta = sample.x;
        float max_PDF = Estimate_GGX_bounded_VNDF_alpha::decode_PDF(sample.y);
        float estimated_alpha = Estimate_GGX_bounded_VNDF_alpha::estimate_alpha(cos_theta, max_PDF);

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
    float m_specularity;

    GGXTransmissionWrapper(float alpha, float ior_i_over_o, float specularity = nanf(""))
        : m_alpha(alpha), m_ior_i_over_o(ior_i_over_o), m_specularity(specularity) {}

    optix::float3 evaluate(optix::float3 wo, optix::float3 wi) const {
        float reflectance = Shading::BSDFs::GGX_T::evaluate(m_alpha, m_ior_i_over_o, wo, wi);
        reflectance *= fresnel(wo, wi);
        return optix::make_float3(reflectance);
    }

    float PDF(optix::float3 wo, optix::float3 wi) const {
        return Shading::BSDFs::GGX_T::PDF(m_alpha, m_ior_i_over_o, wo, wi);
    }

    BSDFResponse evaluate_with_PDF(optix::float3 wo, optix::float3 wi) const {
        BSDFResponse response = Shading::BSDFs::GGX_T::evaluate_with_PDF(m_alpha, m_ior_i_over_o, wo, wi);
        float f = fresnel(wo, wi);
        response.reflectance *= f;
        return response;
    }

    BSDFSample sample(optix::float3 wo, optix::float3 random_sample) const {
        BSDFSample sample = Shading::BSDFs::GGX_T::sample(m_alpha, m_ior_i_over_o, wo, optix::make_float2(random_sample));
        float f = fresnel(wo, sample.direction);
        sample.reflectance *= f;
        return sample;
    }

    std::string to_string() const {
        std::ostringstream out;
        out << "GGX transmission: alpha: " << m_alpha << ", ior_i / ior_o: " << m_ior_i_over_o;
        return out.str();
    }

private:
    // GGX_T ignores Fresnel, so we need to add it explicitly.
    float fresnel(optix::float3 wo, optix::float3 wi) const {
        if (isnan(m_specularity))
            return 1.0f;

        optix::float3 halfway = Shading::BSDFs::GGX_T::compute_halfway_vector(m_ior_i_over_o, wo, wi);
        return 1.0f - schlick_fresnel(m_specularity, dot(wo, halfway));
    }
};

GTEST_TEST(GGX_T, power_conservation) {
    for (float ior_i_over_o : { 0.5f, 0.9f, 1.1f, 1.5f })
        for (float cos_theta : { -1.0f, -0.7f, -0.4f, -0.1f, 0.1f, 0.4f, 0.7f, 1.0f }) {
            optix::float3 wo = { sqrt(1 - pow2(cos_theta)), 0.0f, cos_theta };
            for (float alpha : { 0.0f, 0.0675f, 0.125f, 0.25f, 0.5f, 1.0f }) {
                auto ggx = GGXTransmissionWrapper(alpha, ior_i_over_o);
                auto res = BSDFTestUtils::directional_hemispherical_reflectance_function(ggx, wo, 1024u);
                EXPECT_LE(res.reflectance.x, 1.0f);
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

GTEST_TEST(GGX_T, fully_grazing_evaluates_to_black) {
    using namespace optix;

    const float3 incident_w = make_float3(0.0f, 0.0f, -1.0f);
    const float3 grazing_w = make_float3(0.0f, 1.0f, 0.0f);

    for (float alpha : { 0.0f, 0.5f, 1.0f }) {
        for (float ior_i_over_o : { 0.5f, 0.9f, 1.1f, 1.5f }) {
            auto ggx = GGXTransmissionWrapper(alpha, ior_i_over_o);

            float grazing_wo_f = ggx.evaluate(grazing_w, incident_w).x;
            EXPECT_FLOAT_EQ(grazing_wo_f, 0.0f) << ggx.to_string();

            float grazing_wi_f = ggx.evaluate(incident_w, grazing_w).x;
            EXPECT_FLOAT_EQ(grazing_wi_f, 0.0f) << ggx.to_string();

            float both_grazing_f = ggx.evaluate(grazing_w, grazing_w).x;
            EXPECT_FLOAT_EQ(both_grazing_f, 0.0f) << ggx.to_string();
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
    bool m_disable_reflection;

    GGXWrapper(float alpha, float specularity, float ior_i_over_o, optix::float3 tint = optix::make_float3(1), bool disable_reflection = false)
        : m_alpha(alpha), m_specularity(specularity), m_ior_i_over_o(ior_i_over_o), m_tint(tint), m_disable_reflection(disable_reflection){}

    optix::float3 evaluate(optix::float3 wo, optix::float3 wi) const {
        if (m_disable_reflection && same_hemisphere(wo, wi))
            return optix::make_float3(0.0f);
        return Shading::BSDFs::GGX::evaluate(m_tint, m_alpha, m_specularity, m_ior_i_over_o, wo, wi);
    }

    float PDF(optix::float3 wo, optix::float3 wi) const {
        if (m_disable_reflection && same_hemisphere(wo, wi))
            return 0.0f;
        return Shading::BSDFs::GGX::PDF(m_alpha, m_specularity, m_ior_i_over_o, wo, wi);
    }

    BSDFResponse evaluate_with_PDF(optix::float3 wo, optix::float3 wi) const {
        if (m_disable_reflection && same_hemisphere(wo, wi))
            return BSDFResponse::none();
        return Shading::BSDFs::GGX::evaluate_with_PDF(m_tint, m_alpha, m_specularity, m_ior_i_over_o, wo, wi);
    }

    BSDFSample sample(optix::float3 wo, optix::float3 random_sample) const {
        auto sample = Shading::BSDFs::GGX::sample(m_tint, m_alpha, m_specularity, m_ior_i_over_o, wo, random_sample);
        bool reject_sample = m_disable_reflection && same_hemisphere(wo, sample.direction);
        return reject_sample ? BSDFSample::none() : sample;
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
                EXPECT_FLOAT3_LE(res.reflectance, 1.0f);
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

GTEST_TEST(GGX, reflection_reflectance_equals_GGX_R) {
    using namespace optix;

    float fully_specular = 1.0f; // Disable transmission
    float ior_i_over_o = 1.5f;

    for (float cos_theta : { 0.2f, 1.0f }) {
        float3 wo = BSDFTestUtils::wo_from_cos_theta(cos_theta);
        for (float alpha : { 0.0675f, 0.25f, 1.0f }) {
            auto ggx = GGXWrapper(alpha, fully_specular, ior_i_over_o);
            auto ggx_r = GGXReflectionWrapper(alpha);
            auto ggx_result = BSDFTestUtils::directional_hemispherical_reflectance_function(ggx, wo, 4096);
            auto ggx_r_result = BSDFTestUtils::directional_hemispherical_reflectance_function(ggx_r, wo, 2048);

            float cos_theta_direction = dot(ggx_result.mean_direction, ggx_r_result.mean_direction);
            EXPECT_FLOAT3_EQ_EPS(ggx_result.reflectance, ggx_r_result.reflectance, 0.001f) << ggx.to_string();
            EXPECT_FLOAT_EQ_EPS(1.0f, cos_theta_direction, 0.002f) << ggx.to_string();
        }
    }
}

GTEST_TEST(GGX, transmission_reflectance_equals_GGX_T) {
    using namespace optix;

    float specularity = 0.0f;
    float ior_i_over_o = 1.5f;
    float3 transmissive_tint = { 1.0f, 1.0f, 1.0f };
    bool disable_reflection = true;

    for (float ior_i_over_o : { 0.5f, 1.5f }) {
        for (float cos_theta : { 0.4f, 1.0f }) {
            float3 wo = BSDFTestUtils::wo_from_cos_theta(cos_theta);
            for (float alpha : { 0.0675f, 0.25f, 1.0f }) {
                auto ggx = GGXWrapper(alpha, specularity, ior_i_over_o, transmissive_tint, disable_reflection);
                auto ggx_t = GGXTransmissionWrapper(alpha, ior_i_over_o, specularity);
                auto ggx_result = BSDFTestUtils::directional_hemispherical_reflectance_function(ggx, wo, 2 * 4096);
                auto ggx_t_result = BSDFTestUtils::directional_hemispherical_reflectance_function(ggx_t, wo, 2 * 2048);

                float cos_theta_direction = dot(ggx_result.mean_direction, ggx_t_result.mean_direction);
                EXPECT_FLOAT3_EQ_EPS(ggx_result.reflectance, ggx_t_result.reflectance, 0.0015f) << ggx.to_string();
                EXPECT_FLOAT_EQ_EPS(1.0f, cos_theta_direction, 0.002f) << ggx.to_string();
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
                EXPECT_FLOAT_EQ_EPS(specularity, res.reflectance.x, 0.00001f) << "alpha: " << alpha << ", cos_theta: " << cos_theta << ", specularity: " << specularity;
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

GTEST_TEST(GGX, fully_grazing_evaluates_to_black) {
    using namespace optix;

    const float specularity = 1.0f;
    const float3 grazing_wo = make_float3(0.0f, 1.0f, 0.0f);
    const float3 grazing_wi = make_float3(0.0f, -1.0f, 0.0f);

    for (float alpha : { 0.0f, 0.5f, 1.0f }) {
        for (float ior_i_over_o : { 0.5f, 0.9f, 1.1f, 1.5f }) {
            auto ggx = GGXWrapper(alpha, specularity, ior_i_over_o);

            for (float z_offset : { -0.1f, 0.0f, 0.1f }) {
                float3 w_offset = { 0, 0, z_offset };

                float grazing_wo_f = ggx.evaluate(grazing_wo, normalize(grazing_wi + w_offset)).x;
                EXPECT_FLOAT_EQ(grazing_wo_f, 0.0f) << ggx.to_string();

                float grazing_wi_f = ggx.evaluate(normalize(grazing_wo + w_offset), grazing_wi).x;
                EXPECT_FLOAT_EQ(grazing_wi_f, 0.0f) << ggx.to_string();
            }
        }
    }
}

} // NS OptiXRenderer

#endif // _OPTIXRENDERER_BSDFS_GGX_TEST_H_