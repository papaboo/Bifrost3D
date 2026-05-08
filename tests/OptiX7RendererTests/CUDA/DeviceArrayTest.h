// Test Bifrost Array.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _OPTIX7RENDERER_CUDA_DEVICE_ARRAY_TEST_H_
#define _OPTIX7RENDERER_CUDA_DEVICE_ARRAY_TEST_H_

#include <OptiXRenderer/CUDAUtils.h>

#include <gtest/gtest.h>

namespace OptiXRenderer {

GTEST_TEST(CUDA_DeviceArray, upload) {
    // Create array and upload data.
    auto array = DeviceArray<int>({ 0, 1, 2, 3, 4, 5, 6, 7 });

    // Download data using explicit CUDA calls to ensure that we test CUDA allocated memory.
    int downloaded_data[8];
    CUDA_CHECK(cudaMemcpy(downloaded_data, array.data(), sizeof(int) * array.size(), cudaMemcpyDeviceToHost));

    for (int i = 0; i < array.size(); i++)
        EXPECT_EQ(i, downloaded_data[i]);
}

GTEST_TEST(CUDA_DeviceArray, readback) {
    // Create array and upload data.
    auto array = DeviceArray<int>({ 0, 1, 2, 3, 4, 5, 6, 7 });

    // Readback data
    int readback_data[8];
    array.readback(readback_data, 8);

    for (int i = 0; i < array.size(); i++)
        EXPECT_EQ(i, readback_data[i]);
}

GTEST_TEST(CUDA_DeviceArray, resize_to_zero_doesnt_throw) {
    auto array = DeviceArray<int>({ 0, 1, 2, 3, 4, 5, 6, 7 });
    array.resize(0);

    EXPECT_EQ(0u, array.size());
    EXPECT_EQ(nullptr, array.data());
}

GTEST_TEST(CUDA_DeviceArray, resize_preserves_elements) {
    // Create array and upload data.
    const int smaller_size = 4;
    const int larger_size = 12;
    auto array = DeviceArray<int>({ 0, 1, 2, 3, 4, 5, 6, 7 });

    { // Resize to smaller should preserve still valid elements
        array.resize(smaller_size);

        int readback_data[smaller_size];
        array.readback(readback_data, smaller_size);

        for (int i = 0; i < smaller_size; i++)
            EXPECT_EQ(i, readback_data[i]);
    }

    { // Resize to larger should preserve existing elements.
        array.resize(larger_size);

        // Only readback the old valid elements, as value of new elements are undefined and untested.
        int readback_data[smaller_size];
        array.readback(readback_data, smaller_size);

        for (int i = 0; i < smaller_size; i++)
            EXPECT_EQ(i, readback_data[i]);
    }
}

} // NS OptiX7Renderer

#endif // _OPTIX7RENDERER_CUDA_DEVICE_ARRAY_TEST_H_
