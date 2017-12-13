// DX11Renderer prefix sum compute shader test.
// ------------------------------------------------------------------------------------------------
// Copyright (C) 2017, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _DX11RENDERER_PREFIX_SUM_H_
#define _DX11RENDERER_PREFIX_SUM_H_

#include <gtest/gtest.h>
#include <Utils.h>

#include <DX11Renderer/PrefixSum.h>

namespace DX11Renderer {

// ------------------------------------------------------------------------------------------------
// DX11Renderer prefix sum compute shader test.
// Future tests:
//  * Test for overflow
// ------------------------------------------------------------------------------------------------
GTEST_TEST(PrefixSum, 8_uniform_elements) {
    auto device = create_headless_device1();

    const int element_count = 8;
    unsigned int ds[] = { 1, 1, 1, 1, 1, 1, 1, 1 };
    auto prefix_sum_op = PrefixSum(*device.device, DX11_SHADER_ROOT);
    prefix_sum_op.apply(*device.device, ds, ds + element_count);

    for (int i = 0; i < element_count; ++i)
        EXPECT_EQ(i, ds[i]);
}

GTEST_TEST(PrefixSum, 6_random_elements) {
    auto device = create_headless_device1();

    const int element_count = 6;
    unsigned int ds[] = { 4, 2, 3, 7, 1, 5 };
    unsigned int prefix[] = { 0, 4, 6, 9, 16, 17 };
    auto prefix_sum_op = PrefixSum(*device.device, DX11_SHADER_ROOT);
    prefix_sum_op.apply(*device.device, ds, ds + element_count);

    for (int i = 0; i < element_count; ++i)
        EXPECT_EQ(prefix[i], ds[i]);
}

GTEST_TEST(PrefixSum, 1024_uniform_elements) {
    auto device = create_headless_device1();

    const int element_count = 1024;
    unsigned int ds[element_count];
    for (int i = 0; i < element_count; ++i)
        ds[i] = 1;

    auto prefix_sum_op = PrefixSum(*device.device, DX11_SHADER_ROOT);
    prefix_sum_op.apply(*device.device, ds, ds + element_count);

    for (int i = 0; i < element_count; ++i)
        EXPECT_EQ(i, ds[i]);
}

GTEST_TEST(PrefixSum, 873_uniform_elements) {
    auto device = create_headless_device1();

    const int element_count = 873;
    unsigned int ds[element_count];
    for (int i = 0; i < element_count; ++i)
        ds[i] = 1;

    auto prefix_sum_op = PrefixSum(*device.device, DX11_SHADER_ROOT);
    prefix_sum_op.apply(*device.device, ds, ds + element_count);

    for (int i = 0; i < element_count; ++i)
        EXPECT_EQ(i, ds[i]);
}

GTEST_TEST(PrefixSum, 873_random_elements) {
    auto device = create_headless_device1();

    unsigned int mini_LCG_state = 12190865u;
    auto mini_LCG = [&]() -> unsigned int {
        mini_LCG_state = 1664525u * mini_LCG_state + 1013904223u;
        return mini_LCG_state;
    };

    const int element_count = 873;
    unsigned int ds[element_count];
    for (int i = 0; i < element_count; ++i)
        ds[i] = mini_LCG() % 16;

    unsigned int rs[element_count];
    rs[0] = 0;
    for (int i = 1; i < element_count; ++i)
        rs[i] = rs[i - 1] + ds[i - 1];
    
    auto prefix_sum_op = PrefixSum(*device.device, DX11_SHADER_ROOT);
    prefix_sum_op.apply(*device.device, ds, ds + element_count);

    for (int i = 0; i < element_count; ++i)
        EXPECT_EQ(rs[i], ds[i]);
}

} // NS DX11Renderer

#endif // _DX11RENDERER_PREFIX_SUM_H_