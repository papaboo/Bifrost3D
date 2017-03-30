// Test miscellaneous parts of the OptiXRenderer.
// ---------------------------------------------------------------------------
// Copyright (C) 2015-2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _OPTIXRENDERER_MISC_TEST_H_
#define _OPTIXRENDERER_MISC_TEST_H_

#include <Utils.h>

#include <OptiXRenderer/EncodedNormal.h>
#include <OptiXRenderer/RNG.h>

#include <gtest/gtest.h>

namespace OptiXRenderer {

GTEST_TEST(EncodedNormal, encode_z_sign) {
    for (int x = -10; x < 11; ++x)
        for (int y = -10; y < 11; ++y)
            for (int z = -10; z < 11; ++z) {
                if (x == 0 && y == 0 && z == 0)
                    continue;
                optix::float3 normal = optix::normalize(optix::make_float3(float(x), float(y), float(z)));
                EncodedNormal encoded_normal = EncodedNormal(normal.x, normal.y, normal.z);
                optix::float3 decoded_normal = encoded_normal.decode();
                EXPECT_NORMAL_EQ(normal, decoded_normal, 0.00046f);
            }
}

GTEST_TEST(PowerHeuristic, invariants) {
    // Sanity checks.
    EXPECT_FLOAT_EQ(0.5f, RNG::power_heuristic(1.0f, 1.0f));
    EXPECT_FLOAT_EQ(0.1f, RNG::power_heuristic(1.0f, 3.0f));

    // The power heuristic should return 0 if one of the two inputs are NaN.
    EXPECT_EQ(0.0f, RNG::power_heuristic(1.0f, NAN));
    EXPECT_EQ(0.0f, RNG::power_heuristic(NAN, 1.0f));

    float almost_inf = 3.4028e+38f;
    EXPECT_FALSE(isinf(almost_inf));
    EXPECT_TRUE(isinf(almost_inf * almost_inf));

    // The power heuristic should handle values that squared become infinity.
    EXPECT_EQ(0.0f, RNG::power_heuristic(1.0f, almost_inf));
    EXPECT_EQ(1.0f, RNG::power_heuristic(almost_inf, 1.0f));
}

} // NS OptiXRenderer

#endif // _OPTIXRENDERER_MISC_TEST_H_