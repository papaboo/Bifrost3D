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

static PixelFormat resolve_format(int channels) {
    switch (channels) {
    case 4:
        return PixelFormat::RGBA32;
    case 3:
        return PixelFormat::RGB24;
    }
    return PixelFormat::Unknown;
}

Images::UID load(const std::string& path) {
    
    int width, height, channel_count;
    unsigned char* loaded_data = stbi_load(path.c_str(), &width, &height, &channel_count, 0);

    if (loaded_data == nullptr) {
        printf("StbImageLoader::load(%s) failed with error: '%s'\n", path.c_str(), stbi_failure_reason());
        return Images::UID::invalid_UID();
    }

    // Flip the image vertically.
    for (int y = 0; y < height / 2; ++y) {
        unsigned char* bottom_row = loaded_data + y * width * channel_count;
        unsigned char* bottom_row_end = bottom_row + width * channel_count;
        unsigned char* top_row = loaded_data + (height - y - 1) * width * channel_count;
        while (bottom_row != bottom_row_end) {
            std::swap(*bottom_row, *top_row);
            ++bottom_row;
            ++top_row;
        }
    }

    PixelFormat pixel_format = resolve_format(channel_count);
    if (pixel_format == PixelFormat::Unknown)
        return Images::UID::invalid_UID();

    Images::UID image_ID = Images::create(path, pixel_format, 2.2f, Vector2ui(width, height));
    void* pixel_data = Images::get_pixels(image_ID);
    memcpy(pixel_data, loaded_data, sizeof(unsigned char) * width * height * channel_count);

    delete[] loaded_data;

    return image_ID;
}

} // NS StbImageLoader
