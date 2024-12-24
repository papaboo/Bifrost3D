// Test OptiXRenderer's OrenNayar BRDF.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _OPTIXRENDERER_BSDFS_OREN_NAYAR_TEST_H_
#define _OPTIXRENDERER_BSDFS_OREN_NAYAR_TEST_H_

#include <BSDFTestUtils.h>

#include <OptiXRenderer/Shading/BSDFs/OrenNayar.h>

#include <gtest/gtest.h>

namespace OptiXRenderer {

class OrenNayarWrapper {
public:
    float m_roughness;
    optix::float3 m_tint;

    OrenNayarWrapper(float roughness, optix::float3 tint = { 1, 1, 1})
        : m_roughness(roughness), m_tint(tint) {}

    optix::float3 evaluate(optix::float3 wo, optix::float3 wi) const {
        return Shading::BSDFs::OrenNayar::evaluate(m_tint, m_roughness, wo, wi, true);
    }

    float PDF(optix::float3 wo, optix::float3 wi) const {
        return Shading::BSDFs::OrenNayar::PDF(m_roughness, wo, wi);
    }

    BSDFResponse evaluate_with_PDF(optix::float3 wo, optix::float3 wi) const {
        return Shading::BSDFs::OrenNayar::evaluate_with_PDF(m_tint, m_roughness, wo, wi, true);
    }

    BSDFSample sample(optix::float3 wo, optix::float3 random_sample) const {
        return Shading::BSDFs::OrenNayar::sample(m_tint, m_roughness, wo, optix::make_float2(random_sample), true);
    }

    std::string to_string() const {
        std::ostringstream out;
        out << "OrenNayar: roughness: " << m_roughness << ", tint: [" << m_tint.x << ", " << m_tint.y << ", " << m_tint.z << "]";
        return out.str();
    }
};

GTEST_TEST(OrenNayar, power_conservation) {
    optix::float3 white = { 1, 1, 1 };
    optix::float3 wo = optix::normalize(optix::make_float3(1.0f, 1.0f, 1.0f));
    for (float roughness : {0.0f, 0.2f, 0.4f, 0.6f, 0.8f, 1.0f}) {
        auto oren_nayar = OrenNayarWrapper(roughness);
        auto res = BSDFTestUtils::directional_hemispherical_reflectance_function(oren_nayar, wo, 2048u);
        EXPECT_FLOAT3_EQ_EPS(res.reflectance, white, 0.0002f) << oren_nayar.to_string();
    }
}

GTEST_TEST(OrenNayar, Helmholtz_reciprocity) {
    optix::float3 wo = optix::normalize(optix::make_float3(1.0f, 1.0f, 1.0f));
    for (float roughness : {0.0f, 0.2f, 0.4f, 0.6f, 0.8f, 1.0f}) {
        auto oren_nayar = OrenNayarWrapper(roughness);
        BSDFTestUtils::helmholtz_reciprocity(oren_nayar, wo, 16u);
    }
}

GTEST_TEST(OrenNayar, function_consistency) {
    optix::float3 wo = optix::normalize(optix::make_float3(1.0f, 1.0f, 1.0f));
    for (float roughness : {0.0f, 0.2f, 0.4f, 0.6f, 0.8f, 1.0f}) {
        auto oren_nayar = OrenNayarWrapper(roughness);
        BSDFTestUtils::BSDF_consistency_test(oren_nayar, wo, 16u);
    }
}

GTEST_TEST(OrenNayar, sampling_standard_deviation) {
    float roughness[5] = { 0.0f, 0.25f, 0.5f, 0.75f, 1.0f };
    float expected_rho_std_devs[5] = { 0.0f, 0.074f, 0.095f, 0.114f, 0.135f };
    for (int i = 0; i < 5; i++) {
        auto oren_nayar = OrenNayarWrapper(roughness[i]);
        BSDFTestUtils::BSDF_sampling_variance_test(oren_nayar, 1024, expected_rho_std_devs[i]);
    }
}

GTEST_TEST(OrenNayar, input_albedo_equals_actual_reflectance) {
    optix::float3 albedo = { 0.25f, 0.5f, 0.75f };
    for (float roughness : {0.25f, 0.5f, 0.75f }) {
        auto oren_nayar = OrenNayarWrapper(roughness, albedo);
        for (float cos_theta : {0.1f, 0.5f, 0.9f }) {
            optix::float3 wo = BSDFTestUtils::wo_from_cos_theta(cos_theta);
            auto reflectance = BSDFTestUtils::directional_hemispherical_reflectance_function(oren_nayar, wo, 2048).reflectance;
            EXPECT_FLOAT3_EQ_EPS(reflectance, albedo, 0.0002f) << oren_nayar.to_string();
        }
    }
}

GTEST_TEST(OrenNayar, E_approx_consistency) {
    for (float cos_theta : {0.1f, 0.5f, 0.9f }) {
        for (float roughness : {0.1f, 0.5f, 0.9f }) {
            float e_exact = Shading::BSDFs::OrenNayar::E_FON_exact(cos_theta, roughness);
            float e_approx = Shading::BSDFs::OrenNayar::E_FON_approx(cos_theta, roughness);
            EXPECT_FLOAT_EQ_EPS(e_exact, e_approx, 0.001f);
        }
    }
}

} // NS OptiXRenderer

#endif // _OPTIXRENDERER_BSDFS_OREN_NAYAR_TEST_H_