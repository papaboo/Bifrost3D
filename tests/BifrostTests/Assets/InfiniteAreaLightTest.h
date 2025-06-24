// Test Bifrost infinite area lights.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _BIFROST_ASSETS_INFINITE_AREA_LIGHT_TEST_H_
#define _BIFROST_ASSETS_INFINITE_AREA_LIGHT_TEST_H_

#include <Bifrost/Assets/InfiniteAreaLight.h>
#include <Bifrost/Math/RNG.h>
#include <Bifrost/Math/Utils.h>

#include <gtest/gtest.h>

namespace Bifrost {
namespace Assets {

class Assets_InfiniteAreaLight : public ::testing::Test {
protected:
    // Per-test set-up and tear-down logic.
    virtual void SetUp() {
        Images::allocate(2u);
        Textures::allocate(2u);
    }
    virtual void TearDown() {
        Images::deallocate();
        Textures::deallocate();
    }
};

TEST_F(Assets_InfiniteAreaLight, consistent_PDF_and_evaluate) {
    Image image = Image::create2D("Noisy", PixelFormat::Intensity8, false, Math::Vector2ui(4, 4));

    byte f[] = { 0, 5, 0, 3, 1, 2, 1, 4, 3, 7, 5, 1, 9, 4, 1, 1 };

    byte* pixels = image.get_pixels<byte>();
    std::memcpy(pixels, f, image.get_pixel_count());

    Texture latlong = Texture::create2D(image, MagnificationFilter::Linear, MinificationFilter::Linear, WrapMode::Repeat, WrapMode::Clamp);

    const InfiniteAreaLight light = InfiniteAreaLight(latlong);

    for (int i = 0; i < 32; ++i) {
        auto sample = light.sample(Math::RNG::sample02(i));
        EXPECT_FLOAT_EQ(sample.PDF, light.PDF(sample.direction_to_light));
        EXPECT_RGB_EQ_EPS(sample.radiance, light.evaluate(sample.direction_to_light), 0.000001f);
    }
}

TEST_F(Assets_InfiniteAreaLight, diffuse_integrates_to_white) {
    Image image = Image::create2D("White", PixelFormat::Intensity8, false, Math::Vector2ui(1, 256));

    byte* pixels = image.get_pixels<byte>();
    std::fill(pixels, pixels + image.get_pixel_count(), 255);

    Texture latlong = Texture::create2D(image, MagnificationFilter::Linear, MinificationFilter::Linear, WrapMode::Repeat, WrapMode::Clamp);

    const InfiniteAreaLight light = InfiniteAreaLight(latlong);

    const int MAX_SAMPLES = 8192;
    float radiance[MAX_SAMPLES];

    { // Test diffuse surface with z as up.
        for (int i = 0; i < MAX_SAMPLES; ++i) {
            LightSample sample = light.sample(Math::RNG::sample02(i));
            if (sample.PDF != 0.0f)
                radiance[i] = sample.radiance.r / Math::PI<float>() * Math::max(0.0f, sample.direction_to_light.z) / sample.PDF;
            else
                radiance[i] = 0.0f;
        }
        float average_radiance = Bifrost::Math::sort_and_pairwise_summation(radiance, radiance + MAX_SAMPLES) / float(MAX_SAMPLES);
        EXPECT_FLOAT_IN_RANGE(0.9999f, 1.0001f, average_radiance);
    }

    { // Test diffuse surface with y as up.
        for (int i = 0; i < MAX_SAMPLES; ++i) {
            LightSample sample = light.sample(Math::RNG::sample02(i));
            if (sample.PDF != 0.0f)
                radiance[i] = sample.radiance.r / Math::PI<float>() * Math::max(0.0f, sample.direction_to_light.y) / sample.PDF;
            else
                radiance[i] = 0.0f;
        }
        float average_radiance = Bifrost::Math::sort_and_pairwise_summation(radiance, radiance + MAX_SAMPLES) / float(MAX_SAMPLES);
        EXPECT_FLOAT_IN_RANGE(0.9999f, 1.0001f, average_radiance);
    }
}

TEST_F(Assets_InfiniteAreaLight, PDF_resampling) {
    const int MINIMUM_PDF_HEIGHT = InfiniteAreaLight::MINIMUM_PDF_HEIGHT;

    Image small_image = Image::create2D("Small", PixelFormat::Intensity8, false, Math::Vector2ui(1, 2));
    int small_pixel_count = small_image.get_pixel_count();
    Image large_image = Image::create2D("Large", PixelFormat::Intensity8, false, Math::Vector2ui(1, MINIMUM_PDF_HEIGHT));
    int large_pixel_count = large_image.get_pixel_count();

    // Fill bottom image with black and top image with white
    byte* pixels = small_image.get_pixels<byte>();
    for (int p = 0; p < small_pixel_count; ++p)
        pixels[p] = p < small_pixel_count / 2 ? 0 : 1;
    pixels = large_image.get_pixels<byte>();
    for (int p = 0; p < large_pixel_count; ++p)
        pixels[p] = p < large_pixel_count / 2 ? 0 : 1;

    Texture small_light_texture = Texture::create2D(small_image, MagnificationFilter::None, MinificationFilter::None, WrapMode::Repeat, WrapMode::Clamp);
    Texture large_light_texture = Texture::create2D(large_image, MagnificationFilter::None, MinificationFilter::None, WrapMode::Repeat, WrapMode::Clamp);

    const InfiniteAreaLight small_light = InfiniteAreaLight(small_light_texture);
    const InfiniteAreaLight large_light = InfiniteAreaLight(large_light_texture);

    // Test that the PDF of the small image was resampled and the large image wasn't.
    EXPECT_EQ(small_light.get_PDF_width(), small_image.get_width());
    EXPECT_EQ(small_light.get_PDF_height(), MINIMUM_PDF_HEIGHT);
    EXPECT_EQ(large_light.get_PDF_width(), large_image.get_width());
    EXPECT_EQ(large_light.get_PDF_height(), large_image.get_height());

    // Avoids the first sample as that one is near the pole, where the PDF filtering produces large deviations.
    for (int i = 1; i < 32; ++i) {
        auto sample_uv = Math::RNG::sample02(i);
        LightSample small_sample = small_light.sample(sample_uv);
        LightSample large_sample = large_light.sample(sample_uv);

        EXPECT_NORMAL_EQ(small_sample.direction_to_light, large_sample.direction_to_light, 0.01f);
        EXPECT_FLOAT_EQ_EPS(small_sample.PDF, large_sample.PDF, 0.02f);
    }
}

} // NS Assets
} // NS Bifrost

#endif // _BIFROST_ASSETS_INFINITE_AREA_LIGHT_TEST_H_
