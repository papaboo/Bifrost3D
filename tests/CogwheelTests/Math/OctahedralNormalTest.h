// Test Cogwheel octahedral normal encoding.
// ------------------------------------------------------------------------------------------------
// Copyright (C) 2015, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _COGWHEEL_MATH_OCTAHEDRAL_NORMAL_TEST_H_
#define _COGWHEEL_MATH_OCTAHEDRAL_NORMAL_TEST_H_

#include <Cogwheel/Math/OctahedralNormal.h>
#include <Cogwheel/Math/RNG.h>

#include <gtest/gtest.h>

namespace Cogwheel {
namespace Math {

GTEST_TEST(Math_OctahedralNormal, encode_decode) {
    const float max_error = 0.000047f;

    for (int x = -10; x < 11; ++x)
        for (int y = -10; y < 11; ++y)
            for (int z = -10; z < 11; ++z) {
                if (x == 0 && y == 0 && z == 0)
                    continue;
                Vector3f normal = normalize(Vector3f(float(x), float(y), float(z)));
                Vector3f decoded_normal = OctahedralNormal::encode_precise(normal).decode();
                EXPECT_NORMAL_EQ(normal, decoded_normal, max_error);
            }

    for (int s = 0; s < 10000; ++s) {
        Vector3f normal = normalize(Distributions::Sphere::Sample(RNG::sample02(s)));
        Vector3f decoded_normal = OctahedralNormal::encode_precise(normal).decode();
        EXPECT_NORMAL_EQ(normal, decoded_normal, max_error);
    }
}

} // NS Math
} // NS Cogwheel

#endif // _COGWHEEL_MATH_OCTAHEDRAL_NORMAL_TEST_H_