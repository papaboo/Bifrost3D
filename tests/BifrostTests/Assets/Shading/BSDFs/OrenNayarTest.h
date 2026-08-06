// Test Bifrost's OrenNayar BRDF.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _BIFROST_ASSETS_SHADING_BSDFS_OREN_NAYAR_TEST_H_
#define _BIFROST_ASSETS_SHADING_BSDFS_OREN_NAYAR_TEST_H_

#include <Assets/Shading/BSDFTestUtils.h>
#include <Expects.h>

#include <Bifrost/Assets/Shading/BSDFs/OrenNayar.h>

#include <gtest/gtest.h>

namespace Bifrost::Assets::Shading::BSDFs {

class OrenNayarWrapper {
public:
    float m_roughness;
    Math::RGB m_albedo;

    OrenNayarWrapper(float roughness, Math::RGB albedo = { 1, 1, 1})
        : m_roughness(roughness), m_albedo(albedo) {}

    Math::RGB evaluate(Math::Vector3f wo, Math::Vector3f wi) const {
        return Shading::BSDFs::OrenNayar::evaluate(m_albedo, m_roughness, wo, wi, true);
    }

    PDF pdf(Math::Vector3f wo, Math::Vector3f wi) const {
        return Shading::BSDFs::OrenNayar::pdf(m_roughness, wo, wi);
    }

    BSDFResponse evaluate_with_PDF(Math::Vector3f wo, Math::Vector3f wi) const {
        return Shading::BSDFs::OrenNayar::evaluate_with_PDF(m_albedo, m_roughness, wo, wi, true);
    }

    BSDFSample sample(Math::Vector3f wo, Math::Vector3f random_sample) const {
        return Shading::BSDFs::OrenNayar::sample(m_albedo, m_roughness, wo, Math::Vector2f(random_sample.x, random_sample.y), true);
    }

    std::string to_string() const {
        std::ostringstream out;
        out << "OrenNayar: roughness: " << m_roughness << ", albedo: [" << m_albedo.r << ", " << m_albedo.g << ", " << m_albedo.b << "]";
        return out.str();
    }
};

GTEST_TEST(Assets_Shading_BSDFs_OrenNayar, power_conservation) {
    Math::Vector3f wo = Math::normalize(Math::Vector3f(1.0f, 1.0f, 1.0f));
    for (float roughness : {0.0f, 0.2f, 0.4f, 0.6f, 0.8f, 1.0f}) {
        auto oren_nayar = OrenNayarWrapper(roughness);
        auto res = BSDFTestUtils::directional_hemispherical_reflectance_function(oren_nayar, wo, 2048u);
        EXPECT_RGB_EQ_EPS(Math::RGB::white(), res.reflectance, 0.00045f) << oren_nayar.to_string();
    }
}

GTEST_TEST(Assets_Shading_BSDFs_OrenNayar, Helmholtz_reciprocity) {
    Math::Vector3f wo = Math::normalize(Math::Vector3f(1.0f, 1.0f, 1.0f));
    for (float roughness : { 0.0f, 0.5f, 1.0f }) {
        auto oren_nayar = OrenNayarWrapper(roughness);
        BSDFTestUtils::helmholtz_reciprocity(oren_nayar, wo, 16u);
    }
}

GTEST_TEST(Assets_Shading_BSDFs_OrenNayar, function_consistency) {
    Math::Vector3f wo = Math::normalize(Math::Vector3f(1.0f, 1.0f, 1.0f));
    for (float roughness : { 0.0f, 0.5f, 1.0f }) {
        auto oren_nayar = OrenNayarWrapper(roughness);
        BSDFTestUtils::BSDF_consistency_test(oren_nayar, wo, 16u);
    }
}

GTEST_TEST(Assets_Shading_BSDFs_OrenNayar, sampling_standard_deviation) {
    float roughness[5] = { 0.0f, 0.25f, 0.5f, 0.75f, 1.0f };
    float expected_rho_std_devs[5] = { 0.0f, 0.074f, 0.095f, 0.114f, 0.135f };
    for (int i = 0; i < 5; i++) {
        auto oren_nayar = OrenNayarWrapper(roughness[i]);
        BSDFTestUtils::BSDF_sampling_variance_test(oren_nayar, 1024, expected_rho_std_devs[i]);
    }
}

GTEST_TEST(Assets_Shading_BSDFs_OrenNayar, input_albedo_equals_actual_reflectance) {
    Math::RGB albedo = { 0.25f, 0.5f, 0.75f };
    for (float roughness : { 0.25f, 0.5f, 0.75f }) {
        auto oren_nayar = OrenNayarWrapper(roughness, albedo);
        for (float cos_theta_o : { 0.1f, 0.5f, 0.9f }) {
            Math::Vector3f wo = BSDFTestUtils::w_from_cos_theta(cos_theta_o);
            auto actual_albedo = BSDFTestUtils::directional_hemispherical_reflectance_function(oren_nayar, wo, 2048).reflectance;
            EXPECT_RGB_EQ_EPS(albedo, actual_albedo, 0.0006f) << oren_nayar.to_string();
        }
    }
}

GTEST_TEST(Assets_Shading_BSDFs_OrenNayar, E_approx_consistency) {
    for (float cos_theta : { 0.1f, 0.5f, 0.9f }) {
        for (float roughness : { 0.1f, 0.5f, 0.9f }) {
            float e_exact = Shading::BSDFs::OrenNayar::E_FON_exact(cos_theta, roughness);
            float e_approx = Shading::BSDFs::OrenNayar::E_FON_approx(cos_theta, roughness);
            EXPECT_FLOAT_EQ_EPS(e_exact, e_approx, 0.001f);
        }
    }
}

} // NS Bifrost::Assets::Shading::BSDFs

#endif // _BIFROST_ASSETS_SHADING_BSDFS_OREN_NAYAR_TEST_H_