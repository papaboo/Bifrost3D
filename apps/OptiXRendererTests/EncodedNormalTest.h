// Test OptiXRenderer's normal encoding
// ---------------------------------------------------------------------------
// Copyright (C) 2015-2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _OPTIXRENDERER_NORMAL_ENCODING_TEST_H_
#define _OPTIXRENDERER_NORMAL_ENCODING_TEST_H_

#include <Utils.h>

#include <OptiXRenderer/EncodedNormal.h>

#include <gtest/gtest.h>

namespace OptiXRenderer {

GTEST_TEST(EncodedNormal, encode_z_sign) {
    for (int x = -10; x < 11; ++x)
        for (int y = -10; y < 11; ++y)
            for (int z = -10; z < 11; ++z) {
                if (x == 0 && y == 0 && z == 0)
                    continue;
                optix::float3 normal = optix::normalize(optix::make_float3(float(x), float(y), float(z)));
                EncodedNormal encoded_normal = EncodedNormal(normal);
                optix::float3 decoded_normal = encoded_normal.decode();
                EXPECT_NORMAL_EQ(normal, decoded_normal, 0.00046f);
            }

}

} // NS OptiXRenderer

#endif // _OPTIXRENDERER_NORMAL_ENCODING_TEST_H_