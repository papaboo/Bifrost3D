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
    optix::float3 m_tint = optix::make_float3(1);

    OrenNayarWrapper(float roughness)
        : m_roughness(roughness) {}

    optix::float3 evaluate(optix::float3 wo, optix::float3 wi) const {
        return Shading::BSDFs::OrenNayar::evaluate(m_tint, m_roughness, wo, wi);
    }

    float PDF(optix::float3 wo, optix::float3 wi) const {
        return Shading::BSDFs::OrenNayar::PDF(m_roughness, wo, wi);
    }

    BSDFResponse evaluate_with_PDF(optix::float3 wo, optix::float3 wi) const {
        return Shading::BSDFs::OrenNayar::evaluate_with_PDF(m_tint, m_roughness, wo, wi);
    }

    BSDFSample sample(optix::float3 wo, optix::float3 random_sample) const {
        return Shading::BSDFs::OrenNayar::sample(m_tint, m_roughness, wo, optix::make_float2(random_sample));
    }
};

GTEST_TEST(OrenNayar, power_conservation) {
    optix::float3 wo = optix::normalize(optix::make_float3(1.0f, 1.0f, 1.0f));
    for (float roughness : {0.0f, 0.2f, 0.4f, 0.6f, 0.8f, 1.0f}) {
        OrenNayarWrapper oren_nayar = OrenNayarWrapper(roughness);
        auto res = BSDFTestUtils::directional_hemispherical_reflectance_function(oren_nayar, wo, 1024u);
        EXPECT_LE(res.reflectance, 1.0f);
    }
}

GTEST_TEST(OrenNayar, Helmholtz_reciprocity) {
    optix::float3 wo = optix::normalize(optix::make_float3(1.0f, 1.0f, 1.0f));
    for (float roughness : {0.0f, 0.2f, 0.4f, 0.6f, 0.8f, 1.0f}) {
        OrenNayarWrapper oren_nayar = OrenNayarWrapper(roughness);
        BSDFTestUtils::helmholtz_reciprocity(oren_nayar, wo, 16u);
    }
}

GTEST_TEST(OrenNayar, consistent_PDF) {
    optix::float3 wo = optix::normalize(optix::make_float3(1.0f, 1.0f, 1.0f));
    for (float roughness : {0.0f, 0.2f, 0.4f, 0.6f, 0.8f, 1.0f}) {
        OrenNayarWrapper oren_nayar = OrenNayarWrapper(roughness);
        BSDFTestUtils::PDF_consistency_test(oren_nayar, wo, 16u);
    }
}

GTEST_TEST(OrenNayar, evaluate_with_PDF) {
    optix::float3 wo = optix::normalize(optix::make_float3(1.0f, 1.0f, 1.0f));
    for (float roughness : {0.0f, 0.2f, 0.4f, 0.6f, 0.8f, 1.0f}) {
        OrenNayarWrapper oren_nayar = OrenNayarWrapper(roughness);
        BSDFTestUtils::evaluate_with_PDF_consistency_test(oren_nayar, wo, 16u);
    }
}

} // NS OptiXRenderer

#endif // _OPTIXRENDERER_BSDFS_OREN_NAYAR_TEST_H_