// Cogwheel stb image loader.
// ---------------------------------------------------------------------------
// Copyright (C) 2015-2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#pragma warning(disable : 4996)

#include <StbImageLoader/StbImageLoader.h>

#define STB_IMAGE_IMPLEMENTATION
#include <StbImageLoader/stb_image.h>

using namespace Cogwheel::Assets;
using namespace Cogwheel::Math;

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
        case 3:
            return PixelFormat::RGB24;
        case 4:
            return PixelFormat::RGBA32;
        }
    }
    return PixelFormat::Unknown;
}

unsigned int sizeof_format(PixelFormat format) {
    switch (format) {
    case Cogwheel::Assets::PixelFormat::RGB24:
        return sizeof(unsigned char) * 3;
    case Cogwheel::Assets::PixelFormat::RGBA32:
        return sizeof(unsigned char) * 4;
    case Cogwheel::Assets::PixelFormat::RGB_Float:
        return sizeof(float) * 3;
    case Cogwheel::Assets::PixelFormat::RGBA_Float:
        return sizeof(float) * 4;
    case Cogwheel::Assets::PixelFormat::Unknown:
        return 0u;
    }

    return 0u;
}

Images::UID load(const std::string& path) {
    
    stbi_set_flip_vertically_on_load(true);

    void* loaded_data = nullptr;
    int width, height, channel_count;
    bool is_HDR = check_HDR_fileformat(path);
    if (is_HDR)
        loaded_data = stbi_loadf(path.c_str(), &width, &height, &channel_count, 0);
    else
        loaded_data = stbi_load(path.c_str(), &width, &height, &channel_count, 0);

    if (loaded_data == nullptr) {
        printf("StbImageLoader::load(%s) failed with error: '%s'\n", path.c_str(), stbi_failure_reason());
        return Images::UID::invalid_UID();
    }

    PixelFormat pixel_format = resolve_format(channel_count, is_HDR);
    if (pixel_format == PixelFormat::Unknown)
        return Images::UID::invalid_UID();

    float image_gamma = is_HDR ? 1.0f : 2.2f;
    Images::UID image_ID = Images::create(path, pixel_format, image_gamma, Vector2ui(width, height));
    Images::PixelData pixel_data = Images::get_pixels(image_ID);
    memcpy(pixel_data, loaded_data, sizeof_format(pixel_format) * width * height);

    stbi_image_free(loaded_data);

    return image_ID;
}

} // NS StbImageLoader
