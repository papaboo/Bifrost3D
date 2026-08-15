// Test Bifrost's Lambert BSDF.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _BIFROST_ASSETS_SHADING_BSDFS_LAMBERT_TEST_H_
#define _BIFROST_ASSETS_SHADING_BSDFS_LAMBERT_TEST_H_

#include <Assets/Shading/BSDFTestUtils.h>
#include <Expects.h>

#include <Bifrost/Assets/Shading/BSDFs/Lambert.h>

#include <gtest/gtest.h>

namespace Bifrost::Assets::Shading::BSDFs {

class LambertWrapper {
public:
    Math::RGB m_albedo = Math::RGB::white();

    LambertWrapper() {}

    Math::RGB evaluate(Math::Vector3f wo, Math::Vector3f wi) const {
        return Shading::BSDFs::Lambert::evaluate(m_albedo);
    }

    Math::MonteCarlo::PDF pdf(Math::Vector3f wo, Math::Vector3f wi) const {
        return Shading::BSDFs::Lambert::pdf(wo, wi);
    }

    BSDFResponse evaluate_with_PDF(Math::Vector3f wo, Math::Vector3f wi) const {
        return Shading::BSDFs::Lambert::evaluate_with_PDF(m_albedo, wo, wi);
    }

    BSDFSample sample(Math::Vector3f wo, Math::Vector3f random_sample) const {
        return Shading::BSDFs::Lambert::sample(m_albedo, Math::Vector2f(random_sample.x, random_sample.y));
    }

    std::string to_string() const {
        return "Lambert";
    }
};

GTEST_TEST(Assets_Shading_BSDFs_Lambert, power_conservation) {
    Math::Vector3f wo = Math::normalize(Math::Vector3f(1.0f, 1.0f, 1.0f));
    LambertWrapper lambert = LambertWrapper();
    auto res = BSDFTestUtils::directional_hemispherical_reflectance_function(lambert, wo, 1024u);
    EXPECT_RGB_EQ_EPS(Math::RGB::white(), res.reflectance, 1e-6f);
}

GTEST_TEST(Assets_Shading_BSDFs_Lambert, Helmholtz_reciprocity) {
    Math::Vector3f wo = Math::normalize(Math::Vector3f(1.0f, 1.0f, 1.0f));
    LambertWrapper lambert = LambertWrapper();
    BSDFTestUtils::helmholtz_reciprocity(lambert, wo, 16u);
}

GTEST_TEST(Assets_Shading_BSDFs_Lambert, function_consistency) {
    Math::Vector3f wo = Math::normalize(Math::Vector3f(1.0f, 1.0f, 1.0f));
    LambertWrapper lambert = LambertWrapper();
    BSDFTestUtils::BSDF_consistency_test(lambert, wo, 16u);
}

} // NS Bifrost::Assets::Shading::BSDFs

#endif // _BIFROST_ASSETS_SHADING_BSDFS_LAMBERT_TEST_H_