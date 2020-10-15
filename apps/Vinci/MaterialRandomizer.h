// Vinci material randomizer.
// ------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _VINCI_MATERIAL_RANDOMIZER_H_
#define _VINCI_MATERIAL_RANDOMIZER_H_

#include <Bifrost/Assets/Material.h>
#include <Bifrost/Assets/Image.h>
#include <Bifrost/Math/RNG.h>

using namespace Bifrost::Assets;
using namespace Bifrost::Math;

// ------------------------------------------------------------------------------------------------
// Vinci material randomizer.
// Randomizes material and texture properties byt shuffling channels or generating new data.
// ------------------------------------------------------------------------------------------------
class MaterialRandomizer final {
public:
    MaterialRandomizer(unsigned int random_seed)
        : m_rng(RNG::XorShift32(random_seed)) { }

    void update_materials() {
        float scale_tint_prop = 0.5f;
        float scale_roughness_prop = 0.5f;
        for (Material material : Materials::get_iterable()) {
            if (m_rng.sample1f() < scale_tint_prop)
                material.set_tint(RGB(m_rng.sample1f(), m_rng.sample1f(), m_rng.sample1f()));
            if (m_rng.sample1f() < scale_roughness_prop)
                material.set_roughness(m_rng.sample1f());
        }

        float swizzle_image_channel_prop = 0.5f;
        float invert_image_prop = 0.25f;
        for (Image image : Images::get_iterable()) {
            if (m_rng.sample1f() < swizzle_image_channel_prop)
                swizzle_image(image, m_rng);
            if (m_rng.sample1f() < invert_image_prop)
                invert_image(image);
        }
    }

private:
    static void invert_image(Image image) {
        PixelFormat pixel_format = image.get_pixel_format();
        int image_channel_count = channel_count(pixel_format);
        int tint_channel_count = image_channel_count - has_alpha(pixel_format) ? 1 : 0;

        int pixel_count = image.get_pixel_count();

        if (pixel_format == PixelFormat::Intensity8 || pixel_format == PixelFormat::Alpha8 ||
            pixel_format == PixelFormat::RGB24 || pixel_format == PixelFormat::RGBA32) {
            unsigned char* pixels = (unsigned char*)image.get_pixels();
            unsigned char* pixels_end = pixels + pixel_count * image_channel_count;
            while (pixels < pixels_end) {
                for (int c = 0; c < tint_channel_count; ++c)
                    pixels[c] = 255 - pixels[c];
                pixels += image_channel_count;
            }
        } else if (pixel_format == PixelFormat::Intensity_Float || pixel_format == PixelFormat::RGB_Float ||
                   pixel_format == PixelFormat::RGBA_Float) {
            float* pixels = (float*)image.get_pixels();
            float* pixels_end = pixels + pixel_count * image_channel_count;
            while (pixels < pixels_end) {
                for (int c = 0; c < tint_channel_count; ++c)
                    pixels[c] = 1.0f - pixels[c];
                pixels += image_channel_count;
            }
        }
    }

    static void swizzle_image(Image image, RNG::XorShift32 rng) {
        PixelFormat pixel_format = image.get_pixel_format();
        int image_channel_count = channel_count(pixel_format);
        int tint_channel_count = image_channel_count - has_alpha(pixel_format) ? 1 : 0;
        
        // Early out if color channels cannot be swizzled.
        if (tint_channel_count <= 1)
            return;

        // Determine how to swizzle channels
        auto swizzled_channels = std::vector<unsigned char>(tint_channel_count);
        for (unsigned char i = 0; i < tint_channel_count; ++i)
            swizzled_channels[i] = i;
        for (unsigned char i = 0; i < tint_channel_count; ++i) {
            int channel_source = int(rng.sample1f() * (tint_channel_count - 1));
            std::swap(swizzled_channels.data()[i], swizzled_channels.data()[channel_source]);
        }

        int pixel_count = image.get_pixel_count();
        if (pixel_format == PixelFormat::Intensity8 || pixel_format == PixelFormat::Alpha8 ||
            pixel_format == PixelFormat::RGB24 || pixel_format == PixelFormat::RGBA32) {
            unsigned char* pixels = image.get_pixels<unsigned char>();
            unsigned char* pixels_end = pixels + pixel_count * image_channel_count;
            
            unsigned char tmp[4];
            while (pixels < pixels_end) {
                memcpy(tmp, pixels, tint_channel_count);
                for (int c = 0; c < tint_channel_count; ++c)
                    pixels[c] = tmp[swizzled_channels[c]];
                pixels += image_channel_count;
            }
        } else if (pixel_format == PixelFormat::Intensity_Float || pixel_format == PixelFormat::RGB_Float ||
                 pixel_format == PixelFormat::RGBA_Float) {
            float* pixels = image.get_pixels<float>();
            float* pixels_end = pixels + pixel_count * image_channel_count;

            float tmp[4];
            while (pixels < pixels_end) {
                memcpy(tmp, pixels, tint_channel_count);
                for (int c = 0; c < tint_channel_count; ++c)
                    pixels[c] = tmp[swizzled_channels[c]];
                pixels += image_channel_count;
            }
        }
    }

    RNG::XorShift32 m_rng;
};

#endif _VINCI_MATERIAL_RANDOMIZER_H_