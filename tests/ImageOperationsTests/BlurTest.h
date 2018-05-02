// Test blur operations.
// ---------------------------------------------------------------------------
// Copyright (C) 2018, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _IMAGE_OPERATIONS_BLUR_TEST_H_
#define _IMAGE_OPERATIONS_BLUR_TEST_H_

#include <ImageOperations/Blur.h>

#include <../CogwheelTests/Expects.h>

namespace ImageOperations {
namespace Blur {

using namespace Cogwheel::Assets;
using namespace Cogwheel::Math;

class ImageOperations_Blur : public ::testing::Test {
protected:
    // Per-test set-up and tear-down logic.
    virtual void SetUp() {
        Images::allocate(2u);
    }
    virtual void TearDown() {
        Images::deallocate();
    }
};

inline Image create_image(int width, int height) {
    Image img = Images::create2D("img", PixelFormat::RGBA_Float, 1.0f, Vector2ui(width, height));

    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
            img.set_pixel(RGBA(float(x), float(1.0f / (y + 1)), float(x * x * y * y), float(y - 2)), Vector2ui(x, y));

    return img;
}

TEST_F(ImageOperations_Blur, mirroring) {
    const int size = 8;

    Image image = Images::create2D("img", PixelFormat::RGBA_Float, 1.0f, Vector2ui(size, size));
    Image mirrored_image = Images::create2D("img", PixelFormat::RGBA_Float, 1.0f, Vector2ui(size, size));
    for (int y = 0; y < size; ++y)
        for (int x = 0; x < size; ++x) {
            RGB pixel = RGB(float(x), float(1.0f / (y + 1)), float(x * x * y * y));
            image.set_pixel(pixel, Vector2ui(x, y));
            mirrored_image.set_pixel(pixel, Vector2ui(y, x));
        }

    Image blurred_image = gaussian(image.get_ID(), 2);
    Image mirrored_blurred_image = gaussian(mirrored_image.get_ID(), 2);

    for (int y = 0; y < size; ++y)
        for (int x = 0; x < size; ++x) {
            RGB pixel = image.get_pixel(Vector2ui(x, y)).rgb();
            RGB mirrored_pixel = mirrored_image.get_pixel(Vector2ui(y, x)).rgb();
            EXPECT_FLOAT_EQ_PCT(pixel.r, mirrored_pixel.r, 0.00001f);
            EXPECT_FLOAT_EQ_PCT(pixel.g, mirrored_pixel.g, 0.00001f);
            EXPECT_FLOAT_EQ_PCT(pixel.b, mirrored_pixel.b, 0.00001f);
        }
}

} // NS Blur
} // NS ImageOperations

#endif // _IMAGE_OPERATIONS_BLUR_TEST_H_