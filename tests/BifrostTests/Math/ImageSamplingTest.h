// Test Bifrost fixed point types.
// ------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _BIFROST_MATH_IMAGE_SAMPLING_TEST_H_
#define _BIFROST_MATH_IMAGE_SAMPLING_TEST_H_

#include <Bifrost/Assets/Image.h>
#include <Bifrost/Math/ImageSampling.h>

#include <gtest/gtest.h>

namespace Bifrost::Math {

class Math_ImageSampling : public ::testing::Test {
protected:
    // Per-test set-up and tear-down logic.
    virtual void SetUp() {
        Assets::Images::allocate(8u);
    }
    virtual void TearDown() {
        Assets::Images::deallocate();
    }
};

TEST_F(Math_ImageSampling, bilinear) {
    int width = 4;
    int height = 8;

    // Create test image where the red component in the image corresponds to the u sample and green corresponds to v.
    auto image = Assets::Image::create2D("test", Assets::PixelFormat::RGB_Float, false, Vector2ui(width, height));
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x) {
            RGBA pixel = RGBA(x / (width - 1.0f), y / (height - 1.0f), 0, 1);
            image.set_pixel(pixel, Vector2ui(x, y));
        }
    RGB* pixels = image.get_pixels<RGB>();

    // Sample
    int sample_counts = 3;
    for (int y = -1; y < sample_counts + 1; ++y)
        for (int x = -1; x < sample_counts + 1; ++x) {
            float u = x / (sample_counts - 1.0f);
            float v = y / (sample_counts - 1.0f);

            RGB sample = ImageSampling::bilinear(pixels, width, height, u, v);
            EXPECT_RGB_EQ(RGB(clamp(u, 0.0f, 1.0f), clamp(v, 0.0f, 1.0f), 0), sample);
        }
}

TEST_F(Math_ImageSampling, trilinear) {
    int width = 2;
    int height = 4;
    int depth = 5;

    // Create test image where the red component in the image corresponds to the u sample and green corresponds to v.
    auto image = Assets::Image::create3D("test", Assets::PixelFormat::RGB_Float, false, Vector3ui(width, height, depth));
    for (int z = 0; z < depth; ++z)
        for (int y = 0; y < height; ++y)
            for (int x = 0; x < width; ++x) {
                RGBA pixel = RGBA(x / (width - 1.0f), y / (height - 1.0f), z / (depth - 1.0f), 1);
                image.set_pixel(pixel, Vector3ui(x, y, z));
            }
    RGB* pixels = image.get_pixels<RGB>();

    // Sample
    int sample_counts = 3;
    for (int z = -1; z < sample_counts + 1; ++z)
        for (int y = -1; y < sample_counts + 1; ++y)
            for (int x = -1; x < sample_counts + 1; ++x) {
                float u = x / (sample_counts - 1.0f);
                float v = y / (sample_counts - 1.0f);
                float w = z / (sample_counts - 1.0f);

                RGB sample = ImageSampling::trilinear(pixels, width, height, depth, u, v, w);
                EXPECT_RGB_EQ(RGB(clamp(u, 0.0f, 1.0f), clamp(v, 0.0f, 1.0f), clamp(w, 0.0f, 1.0f)), sample);
            }
}

} // NS Bifrost::Math

#endif // _BIFROST_MATH_IMAGE_SAMPLING_TEST_H_