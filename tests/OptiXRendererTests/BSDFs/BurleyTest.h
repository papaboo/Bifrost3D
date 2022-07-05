// Test OptiXRenderer's Burley BRDF.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _OPTIXRENDERER_BSDFS_BURLEY_TEST_H_
#define _OPTIXRENDERER_BSDFS_BURLEY_TEST_H_

#include <BSDFTestUtils.h>

#include <OptiXRenderer/Shading/BSDFs/Burley.h>

#include <gtest/gtest.h>

namespace OptiXRenderer {

class BurleyWrapper {
public:
    float m_roughness;
    optix::float3 m_tint = optix::make_float3(1);

    BurleyWrapper(float roughness)
        : m_roughness(roughness) {}

    optix::float3 evaluate(optix::float3 wi, optix::float3 wo) const {
        return Shading::BSDFs::Burley::evaluate(m_tint, m_roughness, wi, wo);
    }

    float PDF(optix::float3 wi, optix::float3 wo) const {
        return Shading::BSDFs::Burley::PDF(m_roughness, wi, wo);
    }

    BSDFResponse evaluate_with_PDF(optix::float3 wo, optix::float3 wi) const {
        return Shading::BSDFs::Burley::evaluate_with_PDF(m_tint, m_roughness, wo, wi);
    }

    BSDFSample sample(optix::float3 wo, optix::float3 random_sample) const {
        return Shading::BSDFs::Burley::sample(m_tint, m_roughness, wo, optix::make_float2(random_sample));
    }
};

GTEST_TEST(Burley, power_conservation) {
    optix::float3 wo = optix::normalize(optix::make_float3(1.0f, 1.0f, 1.0f));
    for (float roughness : {0.0f, 0.2f, 0.4f, 0.6f, 0.8f, 1.0f}) {
        BurleyWrapper burley = BurleyWrapper(roughness);
        auto res = BSDFTestUtils::directional_hemispherical_reflectance_function(burley, wo, 1024u);
        EXPECT_LE(res.reflectance, 1.0f);
    }
}

GTEST_TEST(Burley, Helmholtz_reciprocity) {
    optix::float3 wo = optix::normalize(optix::make_float3(1.0f, 1.0f, 1.0f));
    for (float roughness : {0.0f, 0.2f, 0.4f, 0.6f, 0.8f, 1.0f}) {
        BurleyWrapper burley = BurleyWrapper(roughness);
        BSDFTestUtils::helmholtz_reciprocity(burley, wo, 16u);
    }
}

GTEST_TEST(Burley, function_consistency) {
    optix::float3 wo = optix::normalize(optix::make_float3(1.0f, 1.0f, 1.0f));
    for (float roughness : {0.0f, 0.2f, 0.4f, 0.6f, 0.8f, 1.0f}) {
        BurleyWrapper burley = BurleyWrapper(roughness);
        BSDFTestUtils::BSDF_consistency_test(burley, wo, 16u);
    }
}

} // NS OptiXRenderer

#endif // _OPTIXRENDERER_BSDFS_BURLEY_TEST_H_