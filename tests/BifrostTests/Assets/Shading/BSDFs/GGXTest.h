// Test Bifrost's GGX distribution and BSDF.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _BIFROST_ASSETS_SHADING_BSDFS_GGX_TEST_H_
#define _BIFROST_ASSETS_SHADING_BSDFS_GGX_TEST_H_

#include <Assets/Shading/BSDFTestUtils.h>
#include <Expects.h>

#include <Bifrost/Assets/Shading/BSDFs/GGX.h>
#include <Bifrost/Assets/Shading/Fittings.h>

#include <gtest/gtest.h>

namespace Bifrost::Assets::Shading::BSDFs {

// ---------------------------------------------------------------------------
// GGX reflection tests. Specularity is always set to 1 to have full reflection.
// ---------------------------------------------------------------------------

class GGXReflectionWrapper {
public:
    float m_alpha;
    float m_specularity;
    bool m_normalize_rho;

    GGXReflectionWrapper(float alpha, float specularity = 1.0f)
        : m_alpha(alpha), m_specularity(specularity), m_normalize_rho(false) {}

    Math::RGB evaluate(Math::Vector3f wo, Math::Vector3f wi) const {
        float scale = m_normalize_rho ? rho_normalizer(abs(wo.z)) : 1;
        return Math::RGB(Shading::BSDFs::GGX_R::evaluate(m_alpha, m_specularity, wo, wi) * scale);
    }

    void normalized_rho(bool normalize_rho) { m_normalize_rho = normalize_rho; }

    Math::MonteCarlo::PDF pdf(Math::Vector3f wo, Math::Vector3f wi) const {
        return Shading::BSDFs::GGX_R::pdf(m_alpha, wo, wi);
    }

    BSDFResponse evaluate_with_PDF(Math::Vector3f wo, Math::Vector3f wi) const {
        BSDFResponse response = Shading::BSDFs::GGX_R::evaluate_with_PDF(m_alpha, m_specularity, wo, wi);
        if (m_normalize_rho)
            response.reflectance *= rho_normalizer(abs(wo.z));
        return response;
    }

    BSDFSample sample(Math::Vector3f wo, Math::Vector3f random_sample) const {
        BSDFSample sample = Shading::BSDFs::GGX_R::sample(m_alpha, m_specularity, wo, Math::Vector2f(random_sample.x , random_sample.y));
        if (m_normalize_rho)
            sample.reflectance *= rho_normalizer(abs(wo.z));
        return sample;
    }

    std::string to_string() const {
        std::ostringstream out;
        out << "GGX reflection: alpha: " << m_alpha << ", specularity: " << m_specularity;
        return out.str();
    }

private:
    float rho_normalizer(float abs_cos_theta_o) const {
        float roughness = BSDFs::GGX::roughness_from_alpha(m_alpha);
        float no_specularity_rho = Rho::sample_GGX_with_fresnel(abs_cos_theta_o, roughness);
        float full_specularity_rho = Rho::sample_GGX(abs_cos_theta_o, roughness);
        float rho = Math::lerp(no_specularity_rho, full_specularity_rho, m_specularity);
        return 1.0f / rho;
    }
};

GTEST_TEST(Assets_Shading_BSDFs_GGX_R, power_conservation) {
    for (float cos_theta_o : { 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f }) {
        Math::Vector3f wo = BSDFTestUtils::w_from_cos_theta(cos_theta_o);
        for (float alpha : { 0.0f, 0.0675f, 0.125f, 0.25f, 0.5f, 1.0f }) {
            auto ggx = GGXReflectionWrapper(alpha);
            auto res = BSDFTestUtils::directional_hemispherical_reflectance_function(ggx, wo, 1024u);
            EXPECT_RGB_LE(res.reflectance, 1.0f);
        }
    }
}

GTEST_TEST(Assets_Shading_BSDFs_GGX_R, Helmholtz_reciprocity) {
    Math::Vector3f wo = Math::normalize(Math::Vector3f(1.0f, 1.0f, 1.0f));
    for (float alpha : { 0.0675f, 0.125f, 0.25f, 0.5f, 1.0f }) {
        auto ggx = GGXReflectionWrapper(alpha);
        BSDFTestUtils::helmholtz_reciprocity(ggx, wo, 16u);
    }
}

GTEST_TEST(Assets_Shading_BSDFs_GGX_R, function_consistency) {
    Math::Vector3f wo = Math::normalize(Math::Vector3f(1.0f, 1.0f, 1.0f));
    for (float alpha : { 0.0675f, 0.125f, 0.25f, 0.5f, 1.0f }) {
        auto ggx = GGXReflectionWrapper(alpha);
        BSDFTestUtils::BSDF_consistency_test(ggx, wo, 16u);
    }
}

GTEST_TEST(Assets_Shading_BSDFs_GGX_R, PDF_positivity) {
    for (float cos_theta_o : {-0.8f, -0.4f, 0.1f, 0.5f, 0.9f}) {
        Math::Vector3f wo = BSDFTestUtils::w_from_cos_theta(cos_theta_o);
        for (float alpha : { 0.2f, 0.6f, 1.0f }) {
            auto ggx = GGXReflectionWrapper(alpha);
            BSDFTestUtils::PDF_positivity_test(ggx, wo, 128);
        }
    }
}

GTEST_TEST(Assets_Shading_BSDFs_GGX_R, sampling_standard_deviation) {
    float expected_rho_std_dev = 0.36f;
    float alpha = 0.75f;
    auto ggx = GGXReflectionWrapper(alpha); 
    BSDFTestUtils::BSDF_sampling_variance_test(ggx, 1024, expected_rho_std_dev);
}

GTEST_TEST(Assets_Shading_BSDFs_GGX_R, minimal_alpha) {
    using namespace Bifrost::Math;

    const float min_alpha = Shading::BSDFs::GGX::alpha_from_roughness(0.0f);
    const float full_specularity = 1.0f;

    const Vector3f incident_w = Vector3f(0.0f, 0.0f, 1.0f);
    const Vector3f grazing_w = normalize(Vector3f(0.0f, 1.0f, 0.001f));

    float f = Shading::BSDFs::GGX_R::evaluate(min_alpha, full_specularity, incident_w, incident_w);
    EXPECT_FALSE(isnan(f));

    f = Shading::BSDFs::GGX_R::evaluate(min_alpha, full_specularity, grazing_w, incident_w);
    EXPECT_FALSE(isnan(f));

    f = Shading::BSDFs::GGX_R::evaluate(min_alpha, full_specularity, grazing_w, grazing_w);
    EXPECT_FALSE(isnan(f));

    const Vector3f grazing_wi = Vector3f(grazing_w.x, -grazing_w.y, grazing_w.z);
    f = Shading::BSDFs::GGX_R::evaluate(min_alpha, full_specularity, grazing_w, grazing_wi);
    EXPECT_FALSE(isnan(f));
}

GTEST_TEST(Assets_Shading_BSDFs_GGX_R, fully_grazing_evaluates_to_black) {
    Math::Vector3f incident_w = { 0.0f, 0.0f, 1.0f };
    Math::Vector3f grazing_w = { 0.0f, 1.0f, 0.0f };

    for (float alpha : { 0.0f, 0.5f, 1.0f }) {
        auto ggx = GGXReflectionWrapper(alpha);

        float grazing_wo_f = ggx.evaluate(grazing_w, incident_w).r;
        EXPECT_FLOAT_EQ(grazing_wo_f, 0.0f);

        float grazing_wi_f = ggx.evaluate(incident_w, grazing_w).r;
        EXPECT_FLOAT_EQ(grazing_wi_f, 0.0f);

        float both_grazing_f = ggx.evaluate(grazing_w, grazing_w).r;
        EXPECT_FLOAT_EQ(both_grazing_f, 0.0f) << ggx.to_string();
    }
}

GTEST_TEST(Assets_Shading_BSDFs_GGX_R, estimate_bounded_VNDF_alpha_from_max_PDF) {
    const int sample_count = 16;
    const float max_alpha_error = 1.0f / Estimate_GGX_bounded_VNDF_alpha::max_PDF_sample_count;

    for (int i = 0; i < sample_count; i++) {
        Math::Vector2f sample = BSDFTestUtils::bsdf_rng_sample2f(i);
        float cos_theta_o = sample.x;
        float max_PDF = sample.y / (1 - sample.y); // Non-linear mapping from [0, 1] to [0, inf[
        float estimated_alpha = Estimate_GGX_bounded_VNDF_alpha::estimate_alpha(cos_theta_o, max_PDF);

        Math::Vector3f wo = BSDFTestUtils::w_from_cos_theta(cos_theta_o);
        Math::Vector3f reflected_wi = { -wo.x, -wo.y, wo.z };

        float estimated_PDF = GGX_R::pdf(estimated_alpha, wo, reflected_wi).value();

        // Shift alpha towards the correct PDF by the max_alpha_error.
        // If the estimated PDF is lower than the max PDF, then the alpha needs to be reduced (the peak increased),
        // otherwise the alpha should be increased (blurrier reflection).
        float alpha_step_size = max_alpha_error * (estimated_PDF < max_PDF ? -1 : 1);
        float shifted_alpha = estimated_alpha + alpha_step_size;
        shifted_alpha = Math::clamp(shifted_alpha, 0.0f, 1.0f);
        float shifted_PDF = GGX_R::pdf(shifted_alpha, wo, reflected_wi).value();

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

    Math::RGB evaluate(Math::Vector3f wo, Math::Vector3f wi) const {
        float reflectance = Shading::BSDFs::GGX_T::evaluate(m_alpha, m_ior_i_over_o, wo, wi);
        reflectance *= fresnel(wo, wi);
        return Math::RGB(reflectance);
    }

    Math::MonteCarlo::PDF pdf(Math::Vector3f wo, Math::Vector3f wi) const {
        return Shading::BSDFs::GGX_T::pdf(m_alpha, m_ior_i_over_o, wo, wi);
    }

    BSDFResponse evaluate_with_PDF(Math::Vector3f wo, Math::Vector3f wi) const {
        BSDFResponse response = Shading::BSDFs::GGX_T::evaluate_with_PDF(m_alpha, m_ior_i_over_o, wo, wi);
        float f = fresnel(wo, wi);
        response.reflectance *= f;
        return response;
    }

    BSDFSample sample(Math::Vector3f wo, Math::Vector3f random_sample) const {
        BSDFSample sample = Shading::BSDFs::GGX_T::sample(m_alpha, m_ior_i_over_o, wo, Math::Vector2f(random_sample.x, random_sample.y));
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
    float fresnel(Math::Vector3f wo, Math::Vector3f wi) const {
        if (isnan(m_specularity))
            return 1.0f;

        Math::Vector3f halfway = Shading::BSDFs::GGX_T::compute_halfway_vector(m_ior_i_over_o, wo, wi);
        return 1.0f - schlick_fresnel(m_specularity, dot(wo, halfway));
    }
};

GTEST_TEST(Assets_Shading_BSDFs_GGX_T, power_conservation) {
    for (float ior_i_over_o : { 0.5f, 0.9f, 1.1f, 1.5f })
        for (float cos_theta_o : { -1.0f, -0.7f, -0.4f, -0.1f, 0.1f, 0.4f, 0.7f, 1.0f }) {
            Math::Vector3f wo = BSDFTestUtils::w_from_cos_theta(cos_theta_o);
            for (float alpha : { 0.0f, 0.0675f, 0.125f, 0.25f, 0.5f, 1.0f }) {
                auto ggx = GGXTransmissionWrapper(alpha, ior_i_over_o);
                auto res = BSDFTestUtils::directional_hemispherical_reflectance_function(ggx, wo, 1024u);
                EXPECT_LE(res.reflectance.r, 1.0f);
            }
        }
}

// GTEST_TEST(Assets_Shading_BSDFs_GGX_T, Helmholtz_reciprocity) {
//     for (float ior_i_over_o : { 0.5f, 0.9f, 1.1f, 1.5f })
//         for (float cos_theta_o : { -1.0f, -0.4f, -0.1f, 0.1f, 0.4f, 1.0f }) {
//             Math::Vector3f wo = BSDFTestUtils::w_from_cos_theta(cos_theta_o);
//             for (float alpha : { 0.0675f, 0.125f, 0.25f, 0.5f, 1.0f }) {
//                 auto ggx = GGXTransmissionWrapper(alpha, 0.0f, ior_i_over_o);
//                 BSDFTestUtils::helmholtz_reciprocity(ggx, wo, 16u);
//             }
//         }
// }

GTEST_TEST(Assets_Shading_BSDFs_GGX_T, function_consistency) {
    for (float ior_i_over_o : { 0.5f, 0.9f, 1.1f, 1.5f })
        for (float cos_theta_o : { -1.0f, -0.4f, -0.1f, 0.1f, 0.4f, 1.0f }) {
            Math::Vector3f wo = BSDFTestUtils::w_from_cos_theta(cos_theta_o);
            for (float alpha : { 0.0675f, 0.125f, 0.25f, 0.5f, 1.0f }) {
                auto ggx = GGXTransmissionWrapper(alpha, ior_i_over_o);
                BSDFTestUtils::BSDF_consistency_test(ggx, wo, 16u);
            }
        }
}

GTEST_TEST(Assets_Shading_BSDFs_GGX_T, PDF_positivity) {
    for (float cos_theta_o : {-0.8f, -0.4f, 0.1f, 0.5f, 0.9f}) {
        Math::Vector3f wo = BSDFTestUtils::w_from_cos_theta(cos_theta_o);
        for (float alpha : { 0.2f, 0.6f, 1.0f }) {
            auto ggx = GGXTransmissionWrapper(alpha, 0.5f);
            BSDFTestUtils::PDF_positivity_test(ggx, wo, 128);
        }
    }
}

GTEST_TEST(Assets_Shading_BSDFs_GGX_T, sampling_standard_deviation) {
    float alpha = 0.75f;
    float ior_i_over_os[] = { 0.5f, 0.9f, 1.1f, 1.5f };
    float expected_rho_std_devs[] = { 2.05f, 0.53f, 0.05f, 0.08f };
    for (int i = 0; i < 4; i++) {
        auto ggx = GGXTransmissionWrapper(alpha, ior_i_over_os[i]);
        BSDFTestUtils::BSDF_sampling_variance_test(ggx, 1024, expected_rho_std_devs[i]);
    }
}

GTEST_TEST(Assets_Shading_BSDFs_GGX_T, consistent_sampling_across_hemispheres) {
    Math::Vector3f random_sample = { 0.5f, 0.5f, 0.5f };
    for (float cos_theta_o : { -1.0f, -0.4f, -0.1f, 0.1f, 0.4f, 1.0f }) {
        Math::Vector3f positive_wo = BSDFTestUtils::w_from_cos_theta(cos_theta_o);
        Math::Vector3f negative_wo = { positive_wo.x, positive_wo.y, -positive_wo.z };
        for (float alpha : { 0.0675f, 0.125f, 0.25f, 0.5f, 1.0f }) {
            for (float ior_i_over_o : { 0.5f, 0.9f, 1.1f, 1.5f }) {
                auto ggx = GGXTransmissionWrapper(alpha, ior_i_over_o);
                auto positive_sample = ggx.sample(positive_wo, random_sample);
                auto negative_sample = ggx.sample(negative_wo, random_sample);
                EXPECT_EQ(positive_sample.PDF.value(), negative_sample.PDF.value());
                EXPECT_RGB_EQ(positive_sample.reflectance, negative_sample.reflectance);

                Math::Vector3f flipped_negative_direction = negative_sample.direction;
                flipped_negative_direction.z = -flipped_negative_direction.z;
                EXPECT_VECTOR3F_EQ(positive_sample.direction, flipped_negative_direction);
            }
        }
    }
}

GTEST_TEST(Assets_Shading_BSDFs_GGX_T, fully_grazing_evaluates_to_black) {
    Math::Vector3f incident_w = { 0.0f, 0.0f, -1.0f };
    Math::Vector3f grazing_w = { 0.0f, 1.0f, 0.0f };

    for (float alpha : { 0.0f, 0.5f, 1.0f }) {
        for (float ior_i_over_o : { 0.5f, 0.9f, 1.1f, 1.5f }) {
            auto ggx = GGXTransmissionWrapper(alpha, ior_i_over_o);

            float grazing_wo_f = ggx.evaluate(grazing_w, incident_w).r;
            EXPECT_FLOAT_EQ(grazing_wo_f, 0.0f) << ggx.to_string();

            float grazing_wi_f = ggx.evaluate(incident_w, grazing_w).r;
            EXPECT_FLOAT_EQ(grazing_wi_f, 0.0f) << ggx.to_string();

            float both_grazing_f = ggx.evaluate(grazing_w, grazing_w).r;
            EXPECT_FLOAT_EQ(both_grazing_f, 0.0f) << ggx.to_string();
        }
    }
}

GTEST_TEST(Assets_Shading_BSDFs_GGX_T, snells_law) {
    float ior_o = 1;
    float ior_i = 2;
    float ior_i_over_o = ior_i / ior_o;
    float alpha = 0.0f; // Use a smooth surface to test snells law to only allow a single output direction.

    for (float cos_theta_o : { 0.2f, 0.5f, 0.9f }) {
        Math::Vector3f wo = BSDFTestUtils::w_from_cos_theta(cos_theta_o);
        Math::Vector3f wi = GGX_T::sample(alpha, ior_i_over_o, wo, Math::Vector2f(0.5f)).direction;

        float sin_theta_o = sin_theta(wo);
        float sin_theta_i = sin_theta(wi);

        EXPECT_FLOAT_EQ_EPS(ior_o * sin_theta_o, ior_i * sin_theta_i, 1e-6f);
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
    Math::RGB m_tint;
    bool m_disable_reflection;

    GGXWrapper(float alpha, float ior_i_over_o, Math::RGB tint = Math::RGB::white(), bool disable_reflection = false)
        : m_alpha(alpha), m_specularity(dielectric_specularity(air_ior, ior_i_over_o)), m_ior_i_over_o(ior_i_over_o),
          m_tint(tint), m_disable_reflection(disable_reflection) { }

    void overwrite_specularity(float specularity) { m_specularity = specularity; }

    Math::RGB evaluate(Math::Vector3f wo, Math::Vector3f wi) const {
        if (m_disable_reflection && same_hemisphere(wo, wi))
            return Math::RGB::black();
        return Shading::BSDFs::GGX::evaluate(m_tint, m_alpha, m_specularity, m_ior_i_over_o, wo, wi);
    }

    Math::MonteCarlo::PDF pdf(Math::Vector3f wo, Math::Vector3f wi) const {
        if (m_disable_reflection && same_hemisphere(wo, wi))
            return 0.0f;
        return Shading::BSDFs::GGX::pdf(m_tint, m_alpha, m_specularity, m_ior_i_over_o, wo, wi);
    }

    BSDFResponse evaluate_with_PDF(Math::Vector3f wo, Math::Vector3f wi) const {
        if (m_disable_reflection && same_hemisphere(wo, wi))
            return BSDFResponse::none();
        return Shading::BSDFs::GGX::evaluate_with_PDF(m_tint, m_alpha, m_specularity, m_ior_i_over_o, wo, wi);
    }

    BSDFSample sample(Math::Vector3f wo, Math::Vector3f random_sample) const {
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

GTEST_TEST(Assets_Shading_BSDFs_GGX, zero_roughness_converted_to_effectively_smooth_alpha) {
    float smooth_roughness = 0.0f;
    float smooth_alpha = Shading::BSDFs::GGX::alpha_from_roughness(smooth_roughness);
    EXPECT_TRUE(Shading::BSDFs::GGX::effectively_smooth(smooth_alpha));
}

GTEST_TEST(Assets_Shading_BSDFs_GGX, power_conservation) {
    for (float ior_i_over_o : { 0.5f, 0.9f, 1.1f, 1.5f })
        for (float cos_theta_o : { -1.0f, -0.7f, -0.4f, -0.1f, 0.1f, 0.4f, 0.7f, 1.0f }) {
            Math::Vector3f wo = BSDFTestUtils::w_from_cos_theta(cos_theta_o);
            for (float alpha : { 0.0f, 0.0675f, 0.125f, 0.25f, 0.5f, 1.0f }) {
                auto ggx = GGXWrapper(alpha, ior_i_over_o);
                auto res = BSDFTestUtils::directional_hemispherical_reflectance_function(ggx, wo, 1024u);
                EXPECT_RGB_LE(res.reflectance, 1.0f + 1e-5f) << ggx.to_string() << ", cos_theta: " << cos_theta_o;
            }
        }
}

GTEST_TEST(Assets_Shading_BSDFs_GGX, function_consistency) {
    for (float ior_i_over_o : { 0.5f, 0.9f, 1.1f, 1.5f })
        for (float cos_theta_o : { -1.0f, -0.4f, -0.1f, 0.1f, 0.4f, 1.0f }) {
            Math::Vector3f wo = BSDFTestUtils::w_from_cos_theta(cos_theta_o);
            for (float alpha : { 0.0675f, 0.25f, 1.0f }) {
                for (float transmission_tint : { 0.5f, 1.0f }) {
                    auto ggx = GGXWrapper(alpha, ior_i_over_o, Math::RGB(transmission_tint));
                    BSDFTestUtils::BSDF_consistency_test(ggx, wo, 16u);
                }
            }
        }
}

GTEST_TEST(Assets_Shading_BSDFs_GGX, PDF_positivity) {
    float medium_IOR = 1.5f;

    for (float cos_theta_o : {-0.8f, -0.4f, 0.1f, 0.5f, 0.9f}) {
        Math::Vector3f wo = BSDFTestUtils::w_from_cos_theta(cos_theta_o);
        for (float alpha : { 0.2f, 0.6f, 1.0f }) {
            auto ggx = GGXWrapper(alpha, medium_IOR);
            BSDFTestUtils::PDF_positivity_test(ggx, wo, 128);
        }
    }
}

GTEST_TEST(Assets_Shading_BSDFs_GGX, reflection_reflectance_equals_GGX_R) {
    float ior_i_over_o = 1.5f;

    for (float cos_theta_o : { 0.2f, 1.0f }) {
        Math::Vector3f wo = BSDFTestUtils::w_from_cos_theta(cos_theta_o);
        for (float alpha : { 0.0675f, 0.25f, 1.0f }) {
            auto ggx = GGXWrapper(alpha, ior_i_over_o);
            ggx.overwrite_specularity(1.0f); // Disable transmission
            auto ggx_r = GGXReflectionWrapper(alpha);
            auto ggx_result = BSDFTestUtils::directional_hemispherical_reflectance_function(ggx, wo, 4096);
            auto ggx_r_result = BSDFTestUtils::directional_hemispherical_reflectance_function(ggx_r, wo, 2048);

            float cos_theta_direction = dot(ggx_result.mean_direction, ggx_r_result.mean_direction);
            EXPECT_RGB_EQ_EPS(ggx_result.reflectance, ggx_r_result.reflectance, 0.001f) << ggx.to_string();
            EXPECT_FLOAT_EQ_EPS(1.0f, cos_theta_direction, 0.002f) << ggx.to_string();
        }
    }
}

GTEST_TEST(Assets_Shading_BSDFs_GGX, transmission_reflectance_equals_GGX_T) {
    float specularity = 0.0f;
    Math::RGB transmissive_tint = Math::RGB::white();
    bool disable_reflection = true;

    for (float ior_i_over_o : { 0.5f, 1.5f }) {
        for (float cos_theta_o : { 0.4f, 1.0f }) {
            Math::Vector3f wo = BSDFTestUtils::w_from_cos_theta(cos_theta_o);
            for (float alpha : { 0.0675f, 0.25f, 1.0f }) {
                auto ggx = GGXWrapper(alpha, ior_i_over_o, transmissive_tint, disable_reflection);
                ggx.overwrite_specularity(specularity);
                auto ggx_t = GGXTransmissionWrapper(alpha, ior_i_over_o, specularity);
                auto ggx_result = BSDFTestUtils::directional_hemispherical_reflectance_function(ggx, wo, 4096);
                auto ggx_t_result = BSDFTestUtils::directional_hemispherical_reflectance_function(ggx_t, wo, 4096);

                float cos_theta_direction = dot(ggx_result.mean_direction, ggx_t_result.mean_direction);
                EXPECT_RGB_EQ_EPS(ggx_result.reflectance, ggx_t_result.reflectance, 0.0015f) << ggx.to_string();
                EXPECT_FLOAT_EQ_EPS(1.0f, cos_theta_direction, 0.002f) << ggx.to_string();
            }
        }
    }
}

GTEST_TEST(Assets_Shading_BSDFs_GGX, sample_according_to_specularity) {
    // A black tint zeroes out the transmission reflectance, while the reflection is still white.
    // The reflectance should therefore equal the reflection to transmission ratio.
    Math::RGB black = { 0, 0, 0 };
    float roughness = 0.0f;
    float alpha = Shading::BSDFs::GGX::alpha_from_roughness(roughness);

    for (float cos_theta : { -1.0f, 1.0f }) {
        Math::Vector3f wo = { 0, 0, cos_theta };
        for (float ior_i_over_o : { 0.5f, 1.5f }) {
            for (float specularity : { 0.0f, 0.5f, 1.0f }) {
                auto ggx = GGXWrapper(alpha, ior_i_over_o, black);
                ggx.overwrite_specularity(specularity);
                auto res = BSDFTestUtils::directional_hemispherical_reflectance_function(ggx, wo, 1024u);
                EXPECT_FLOAT_EQ_EPS(specularity, res.reflectance.r, 0.00001f) << "alpha: " << alpha << ", cos_theta: " << cos_theta << ", specularity: " << specularity;
            }
        }
    }
}

GTEST_TEST(Assets_Shading_BSDFs_GGX, sampling_standard_deviation) {
    float alpha = 0.75f;
    float ior_i_over_os[] = { 0.5f, 0.9f, 1.1f, 1.5f };
    float expected_rho_std_devs[] = { 0.70f, 0.57f, 0.46f, 0.46f };
    for (int i = 0; i < 4; i++) {
        auto ggx = GGXWrapper(alpha, ior_i_over_os[i]);
        ggx.overwrite_specularity(0.5f); // Somewhat equal distribution of samples between reflection and transmission.
        BSDFTestUtils::BSDF_sampling_variance_test(ggx, 1024, expected_rho_std_devs[i]);
    }
}

GTEST_TEST(Assets_Shading_BSDFs_GGX, black_transmission_never_sampled) {
    float medium_IOR = 1.5f;
    Math::RGB black_transmission = Math::RGB::black();

    for (float alpha : { 0.2f, 0.6f, 1.0f }) {
        auto ggx = GGXWrapper(alpha, medium_IOR, black_transmission);
        for (float cos_theta_o : { 0.1f, 0.5f, 0.9f }) {
            Math::Vector3f wo = BSDFTestUtils::w_from_cos_theta(cos_theta_o);

            for (int i = 0; i < 128; i++) {
                Math::Vector3f rng_sample = BSDFTestUtils::bsdf_rng_sample3f(i, 128);
                auto sample = ggx.sample(wo, rng_sample);

                // Test that the sample always represents a reflection.
                EXPECT_GE(sample.direction.z, 0.0f);
            }
        }
    }
}

GTEST_TEST(Assets_Shading_BSDFs_GGX, fully_grazing_evaluates_to_black) {
    Math::Vector3f grazing_wo = { 0.0f, 1.0f, 0.0f };
    Math::Vector3f grazing_wi = { 0.0f, -1.0f, 0.0f };

    for (float alpha : { 0.0f, 0.5f, 1.0f }) {
        for (float ior_i_over_o : { 0.5f, 0.9f, 1.1f, 1.5f }) {
            auto ggx = GGXWrapper(alpha, ior_i_over_o);
            ggx.overwrite_specularity(1.0f);

            for (float z_offset : { -0.1f, 0.0f, 0.1f }) {
                Math::Vector3f w_offset = { 0, 0, z_offset };

                float grazing_wo_f = ggx.evaluate(grazing_wo, normalize(grazing_wi + w_offset)).r;
                EXPECT_FLOAT_EQ(grazing_wo_f, 0.0f) << ggx.to_string();

                float grazing_wi_f = ggx.evaluate(normalize(grazing_wo + w_offset), grazing_wi).r;
                EXPECT_FLOAT_EQ(grazing_wi_f, 0.0f) << ggx.to_string();
            }
        }
    }
}

} // NS Bifrost::Assets::Shading::BSDFs

#endif // _BIFROST_ASSETS_SHADING_BSDFS_GGX_TEST_H_