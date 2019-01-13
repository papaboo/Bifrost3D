// Test Compare operations.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _IMAGE_OPERATIONS_COMPARE_TEST_H_
#define _IMAGE_OPERATIONS_COMPARE_TEST_H_

#include <ImageOperations/Compare.h>
#include <../BifrostTests/Expects.h>

namespace ImageOperations {
namespace Compare {

class ImageOperations_Compare : public ::testing::Test {
protected:
    // Per-test set-up and tear-down logic.
    virtual void SetUp() {
        Bifrost::Assets::Images::allocate(2u);
    }
    virtual void TearDown() {
        Bifrost::Assets::Images::deallocate();
    }
};

inline Image create_image(int width, int height) {
    Image img = Images::create2D("img", PixelFormat::RGBA_Float, 1.0f, Vector2ui(width, height));

    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
            img.set_pixel(RGBA(float(x), float(1.0f / (y + 1)), float(x * x * y * y), float(y - 2)), Vector2ui(x, y));

    return img;
}

TEST_F(ImageOperations_Compare, RMS_identity) {
    int width = 4, height = 4;
    Image img = create_image(width, height);

    EXPECT_EQ(0.0f, rms(img, img));
}

TEST_F(ImageOperations_Compare, RMS) {
    int width = 4, height = 4;

    Image img1 = create_image(width, height);
    Image img2 = create_image(width, height);

    RGBA pixel_1_2 = img2.get_pixel(Vector2ui(1, 2));
    pixel_1_2.rgb() += RGB(1.0f); // RMS 1
    img2.set_pixel(pixel_1_2, Vector2ui(1, 2));
    RGBA pixel_3_1 = img2.get_pixel(Vector2ui(3, 1));
    pixel_3_1.rgb() += RGB(-1.0f); // RMS 1
    img2.set_pixel(pixel_3_1, Vector2ui(3, 1));
    RGBA pixel_0_1 = img2.get_pixel(Vector2ui(0, 1));
    pixel_0_1.rgb() += RGB(2.0f); // RMS 4
    img2.set_pixel(pixel_0_1, Vector2ui(0, 1));

    float expected_rms = sqrt((1 + 1 + 4) / float(width * height));
    EXPECT_EQ(expected_rms, rms(img1, img2));
}

TEST_F(ImageOperations_Compare, SSIM) {
    int width = 4, height = 4;
    Image img1 = create_image(width, height);
    Image img2 = create_image(width, height);
    EXPECT_EQ(1.0f, ssim(img1, img2));

    // Verifying SSIM basically amounts to reimplementing the algorithm, 
    // so instead of verifying it, we test that it behaves as expected.

    RGBA pixel_1_2 = img2.get_pixel(Vector2ui(1, 2));
    pixel_1_2.rgb() += RGB(1.0f);
    img2.set_pixel(pixel_1_2, Vector2ui(1, 2));
    float ssim_1 = ssim(img1, img2);
    
    RGBA pixel_3_1 = img2.get_pixel(Vector2ui(3, 1));
    pixel_3_1.rgb() += RGB(-1.0f);
    img2.set_pixel(pixel_3_1, Vector2ui(3, 1));
    float ssim_2 = ssim(img1, img2);

    RGBA pixel_0_1 = img2.get_pixel(Vector2ui(0, 1));
    pixel_0_1.rgb() += RGB(2.0f);
    img2.set_pixel(pixel_0_1, Vector2ui(0, 1));
    float ssim_3 = ssim(img1, img2);

    EXPECT_LT(ssim_1, 1.0f);
    EXPECT_LT(ssim_2, ssim_1);
    EXPECT_LT(ssim_3, ssim_2);
}

TEST_F(ImageOperations_Compare, MSSIM) {
    int width = 4, height = 4;
    Image img1 = create_image(width, height);
    Image img2 = create_image(width, height);
    EXPECT_EQ(1.0f, mssim(img1, img2, 1));

    // Verifying SSIM basically amounts to reimplementing the algorithm, 
    // so instead of verifying it, we test that it behaves as expected.

    RGBA pixel_1_2 = img2.get_pixel(Vector2ui(1, 2));
    pixel_1_2.rgb() += RGB(1.0f);
    img2.set_pixel(pixel_1_2, Vector2ui(1, 2));
    float mssim_1 = mssim(img1, img2, 1);

    RGBA pixel_3_1 = img2.get_pixel(Vector2ui(3, 1));
    pixel_3_1.rgb() += RGB(-1.0f);
    img2.set_pixel(pixel_3_1, Vector2ui(3, 1));
    float mssim_2 = mssim(img1, img2, 1);

    RGBA pixel_0_1 = img2.get_pixel(Vector2ui(0, 1));
    pixel_0_1.rgb() += RGB(2.0f);
    img2.set_pixel(pixel_0_1, Vector2ui(0, 1));
    float mssim_3 = mssim(img1, img2, 1);

    EXPECT_LT(mssim_1, 1.0f);
    EXPECT_LT(mssim_2, mssim_1);
    EXPECT_LT(mssim_3, mssim_2);
}

} // NS Compare
} // NS ImageOperations

#endif // _IMAGE_OPERATIONS_COMPARE_TEST_H_