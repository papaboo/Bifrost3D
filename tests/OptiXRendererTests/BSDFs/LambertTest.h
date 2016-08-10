// Test OptiXRenderer's Lambert BSDF.
// ---------------------------------------------------------------------------
// Copyright (C) 2015-2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _OPTIXRENDERER_BSDFS_LAMBERT_TEST_H_
#define _OPTIXRENDERER_BSDFS_LAMBERT_TEST_H_

#include <Utils.h>

#include <Cogwheel/Math/Utils.h>

#include <OptiXRenderer/RNG.h>
#include <OptiXRenderer/Shading/BSDFs/Lambert.h>

#include <gtest/gtest.h>

namespace OptiXRenderer {

GTEST_TEST(Lambert, power_conservation) {
    using namespace optix;

    const unsigned int MAX_SAMPLES = 1024u;
    const float3 tint = make_float3(1.0f, 1.0f, 1.0f);

    float ws[MAX_SAMPLES];
    for (unsigned int i = 0u; i < MAX_SAMPLES; ++i) {
        BSDFSample sample = Shading::BSDFs::Lambert::sample(tint, RNG::sample02(i));
        ws[i] = sample.weight.x * sample.direction.z / sample.PDF; // f * ||cos_theta|| / pdf
    }

    float average_w = Cogwheel::Math::sort_and_pairwise_summation(ws, ws + MAX_SAMPLES) / float(MAX_SAMPLES);
    EXPECT_TRUE(almost_equal_eps(average_w, 1.0f, 0.0001f));
}

GTEST_TEST(Lambert, evaluate_with_PDF) {
    using namespace optix;

    const unsigned int MAX_SAMPLES = 128;
    const float3 tint = make_float3(1.0f, 1.0f, 1.0f);
    const float3 wo = normalize(make_float3(1, 1, 1));

    for (unsigned int i = 0u; i < MAX_SAMPLES; ++i) {
        BSDFSample sample = Shading::BSDFs::Lambert::sample(tint, RNG::sample02(i));

        BSDFResponse response = Shading::BSDFs::Lambert::evaluate_with_PDF(tint, wo, sample.direction);
        EXPECT_NORMAL_EQ(sample.weight, response.weight, 0.000000001f);
        EXPECT_FLOAT_EQ(sample.PDF, response.PDF);
    }
}

} // NS OptiXRenderer

#endif // _OPTIXRENDERER_BSDFS_LAMBERT_TEST_H_