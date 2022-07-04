// Test OptiXRenderer's Lambert BSDF.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _OPTIXRENDERER_BSDFS_LAMBERT_TEST_H_
#define _OPTIXRENDERER_BSDFS_LAMBERT_TEST_H_

#include <BSDFTestUtils.h>

#include <OptiXRenderer/Shading/BSDFs/Lambert.h>

#include <gtest/gtest.h>

namespace OptiXRenderer {

class LambertWrapper {
public:
    optix::float3 m_tint = optix::make_float3(1);

    LambertWrapper() {}

    optix::float3 evaluate(optix::float3 wo, optix::float3 wi) const {
        return Shading::BSDFs::Lambert::evaluate(m_tint, wo, wi);
    }

    float PDF(optix::float3 wo, optix::float3 wi) const {
        return Shading::BSDFs::Lambert::PDF(wo, wi);
    }

    BSDFResponse evaluate_with_PDF(optix::float3 wo, optix::float3 wi) const {
        return Shading::BSDFs::Lambert::evaluate_with_PDF(m_tint, wo, wi);
    }

    BSDFSample sample(optix::float3 wo, optix::float3 random_sample) const {
        return Shading::BSDFs::Lambert::sample(m_tint, optix::make_float2(random_sample));
    }
};

GTEST_TEST(Lambert, power_conservation) {
    optix::float3 wo = optix::normalize(optix::make_float3(1.0f, 1.0f, 1.0f));
    LambertWrapper lambert = LambertWrapper();
    auto res = BSDFTestUtils::directional_hemispherical_reflectance_function(lambert, wo, 1024u);
    EXPECT_LE(res.reflectance, 1.0f);
}

GTEST_TEST(Lambert, Helmholtz_reciprocity) {
    optix::float3 wo = optix::normalize(optix::make_float3(1.0f, 1.0f, 1.0f));
    LambertWrapper lambert = LambertWrapper();
    BSDFTestUtils::helmholtz_reciprocity(lambert, wo, 16u);
}

GTEST_TEST(Lambert, consistent_PDF) {
    optix::float3 wo = optix::normalize(optix::make_float3(1.0f, 1.0f, 1.0f));
    LambertWrapper lambert = LambertWrapper();
    BSDFTestUtils::PDF_consistency_test(lambert, wo, 16u);
}

GTEST_TEST(Lambert, evaluate_with_PDF) {
    optix::float3 wo = optix::normalize(optix::make_float3(1.0f, 1.0f, 1.0f));
    LambertWrapper lambert = LambertWrapper();
    BSDFTestUtils::evaluate_with_PDF_consistency_test(lambert, wo, 16u);
}

} // NS OptiXRenderer

#endif // _OPTIXRENDERER_BSDFS_LAMBERT_TEST_H_