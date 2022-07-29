// Bifrost stb image loader.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#pragma warning(disable : 4996)

#include <StbImageLoader/StbImageLoader.h>

#include <Bifrost/Assets/Image.h>

#define STB_IMAGE_IMPLEMENTATION
#include <StbImageLoader/stb_image.h>

using namespace Bifrost::Assets;
using namespace Bifrost::Math;

namespace StbImageLoader {

bool check_HDR_fileformat(const std::string& path) {
    return memcmp(path.c_str() + path.size() - 4, ".hdr", sizeof(unsigned char) * 4) == 0;
}

static PixelFormat resolve_format(int channels, bool is_HDR) {
    if (is_HDR) {
        switch (channels) {
        case 3:
            return PixelFormat::RGB_Float;
        case 4:
            return PixelFormat::RGBA_Float;
        }
    } else {
        switch (channels) {
        case 1:
            return PixelFormat::Intensity8;
        case 3:
            return PixelFormat::RGB24;
        case 2: // [intensity, alpha]. Data is expanded when copied to the datamodel.
        case 4:
            return PixelFormat::RGBA32;
        }
    }
    return PixelFormat::Unknown;
}

unsigned int sizeof_format(PixelFormat format) {
    switch (format) {
    case Bifrost::Assets::PixelFormat::Intensity8:
        return sizeof(unsigned char);
    case Bifrost::Assets::PixelFormat::RGB24:
        return sizeof(unsigned char) * 3;
    case Bifrost::Assets::PixelFormat::RGBA32:
        return sizeof(unsigned char) * 4;
    case Bifrost::Assets::PixelFormat::RGB_Float:
        return sizeof(float) * 3;
    case Bifrost::Assets::PixelFormat::RGBA_Float:
        return sizeof(float) * 4;
    case Bifrost::Assets::PixelFormat::Unknown:
        return 0u;
    }

    return 0u;
}

inline ImageID convert_image(const std::string& name, void* loaded_pixels, int width, int height, int channel_count, bool is_HDR) {
    if (loaded_pixels == nullptr) {
        printf("StbImageLoader::load(%s) error: '%s'\n", name.c_str(), stbi_failure_reason());
        return ImageID::invalid_UID();
    }

    PixelFormat pixel_format = resolve_format(channel_count, is_HDR);
    if (pixel_format == PixelFormat::Unknown) {
        printf("StbImageLoader::load(%s) error: 'Could not resolve format'\n", name.c_str());
        return ImageID::invalid_UID();
    }

    float image_gamma = is_HDR ? 1.0f : 2.2f;
    ImageID image_ID = Images::create2D(name, pixel_format, image_gamma, Vector2ui(width, height));
    Images::PixelData pixel_data = Images::get_pixels(image_ID);
    if (channel_count == 2) {
        unsigned char* pixel_data_uc4 = (unsigned char*)pixel_data;
        unsigned char* loaded_data_uc2 = (unsigned char*)loaded_pixels;
        for (int i = 0; i < width * height; ++i) {
            pixel_data_uc4[4 * i] = loaded_data_uc2[2 * i];
            pixel_data_uc4[4 * i + 1] = loaded_data_uc2[2 * i];
            pixel_data_uc4[4 * i + 2] = loaded_data_uc2[2 * i];
            pixel_data_uc4[4 * i + 3] = loaded_data_uc2[2 * i + 1];
        }
    }
    else
        memcpy(pixel_data, loaded_pixels, sizeof_format(pixel_format) * width * height);

    stbi_image_free(loaded_pixels);

    return image_ID;
}

ImageID load(const std::string& path) {
    stbi_set_flip_vertically_on_load(true);

    void* loaded_pixels = nullptr;
    int width, height, channel_count;
    bool is_HDR = check_HDR_fileformat(path);
    if (is_HDR)
        loaded_pixels = stbi_loadf(path.c_str(), &width, &height, &channel_count, 0);
    else
        loaded_pixels = stbi_load(path.c_str(), &width, &height, &channel_count, 0);

    return convert_image(path, loaded_pixels, width, height, channel_count, is_HDR);
}

Bifrost::Assets::ImageID load_from_memory(const std::string& name, const void* const data, int data_byte_count) {
    stbi_set_flip_vertically_on_load(false);

    int width, height, channel_count;
    void* loaded_pixels = stbi_load_from_memory((stbi_uc*)data, data_byte_count, &width, &height, &channel_count, 0);

    bool is_HDR = false; // NOTE Currently we cannot distinguish LDR from HDR images.
    return convert_image(name, loaded_pixels, width, height, channel_count, is_HDR);
}

} // NS StbImageLoader
