// Test Bifrost's Burley BRDF.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _BIFROST_ASSETS_SHADING_BSDFS_BURLEY_TEST_H_
#define _BIFROST_ASSETS_SHADING_BSDFS_BURLEY_TEST_H_

#include <Assets/Shading/BSDFTestUtils.h>
#include <Expects.h>

#include <Bifrost/Assets/Shading/BSDFs/Burley.h>

#include <gtest/gtest.h>

namespace Bifrost::Assets::Shading::BSDFs {

class BurleyWrapper {
public:
    float m_roughness;
    Math::RGB m_tint = Math::RGB::white();

    BurleyWrapper(float roughness)
        : m_roughness(roughness) {}

    Math::RGB evaluate(Math::Vector3f wo, Math::Vector3f wi) const {
        return Shading::BSDFs::Burley::evaluate(m_tint, m_roughness, wo, wi);
    }

    Math::MonteCarlo::PDF pdf(Math::Vector3f wo, Math::Vector3f wi) const {
        return Shading::BSDFs::Burley::pdf(m_roughness, wo, wi);
    }

    BSDFResponse evaluate_with_PDF(Math::Vector3f wo, Math::Vector3f wi) const {
        return Shading::BSDFs::Burley::evaluate_with_PDF(m_tint, m_roughness, wo, wi);
    }

    BSDFSample sample(Math::Vector3f wo, Math::Vector3f random_sample) const {
        return Shading::BSDFs::Burley::sample(m_tint, m_roughness, wo, Math::Vector2f(random_sample.x, random_sample.y));
    }

    std::string to_string() const {
        std::ostringstream out;
        out << "Burley: roughness: " << m_roughness;
        return out.str();
    }
};

GTEST_TEST(Assets_Shading_BSDFs_Burley, power_conservation) {
    Math::Vector3f wo = Math::normalize(Math::Vector3f(1.0f, 1.0f, 1.0f));
    for (float roughness : {0.0f, 0.2f, 0.4f, 0.6f, 0.8f, 1.0f}) {
        BurleyWrapper burley = BurleyWrapper(roughness);
        auto res = BSDFTestUtils::directional_hemispherical_reflectance_function(burley, wo, 1024u);
        EXPECT_RGB_EQ_EPS(Math::RGB::white(), res.reflectance, 1.00045f) << burley.to_string();
    }
}

GTEST_TEST(Assets_Shading_BSDFs_Burley, Helmholtz_reciprocity) {
    Math::Vector3f wo = Math::normalize(Math::Vector3f(1.0f, 1.0f, 1.0f));
    for (float roughness : {0.0f, 0.2f, 0.4f, 0.6f, 0.8f, 1.0f}) {
        BurleyWrapper burley = BurleyWrapper(roughness);
        BSDFTestUtils::helmholtz_reciprocity(burley, wo, 16u);
    }
}

GTEST_TEST(Assets_Shading_BSDFs_Burley, function_consistency) {
    Math::Vector3f wo = Math::normalize(Math::Vector3f(1.0f, 1.0f, 1.0f));
    for (float roughness : {0.0f, 0.2f, 0.4f, 0.6f, 0.8f, 1.0f}) {
        BurleyWrapper burley = BurleyWrapper(roughness);
        BSDFTestUtils::BSDF_consistency_test(burley, wo, 16u);
    }
}

} // NS Bifrost::Assets::Shading::BSDFs

#endif // _BIFROST_ASSETS_SHADING_BSDFS_BURLEY_TEST_H_