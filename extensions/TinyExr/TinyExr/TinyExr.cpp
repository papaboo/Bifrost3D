// Bifrost exr loader and saver.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

// #pragma warning(disable : 4996)

#include <TinyExr/TinyExr.h>

#define TINYEXR_IMPLEMENTATION
#include <TinyExr/tiny_exr.h>

using namespace Bifrost::Assets;
using namespace Bifrost::Math;

namespace TinyExr {

Result load_verbose(const std::string& filename, Bifrost::Assets::Images::UID& image_ID) {

    float* rgba = nullptr;
    int width, height;
    const char* error_msg = nullptr;
    Result res = (Result)LoadEXR(&rgba, &width, &height, filename.c_str(), &error_msg);

    if (res == Result::Success) {
        float image_gamma = 1.0f;
        image_ID = Images::create2D(filename, PixelFormat::RGBA_Float, image_gamma, Vector2ui(width, height));
        Images::PixelData pixel_data = Images::get_pixels(image_ID);
        memcpy(pixel_data, rgba, sizeof(float) * 4 * width * height);
    }
    else
    {
        image_ID = Images::UID::invalid_UID();
        printf("TinyExr: %s\n", error_msg);
    }

    delete[] rgba;

    return res;
}

Result store(Bifrost::Assets::Images::UID image_ID, const std::string& filename) {
    Result res;
    const char* error_msg = nullptr;

    Image image = image_ID;
    if (image.get_pixel_format() == PixelFormat::RGB_Float || image.get_pixel_format() == PixelFormat::RGBA_Float) {
        int save_as_fp16 = 0;
        res = (Result)SaveEXR((float*)image.get_pixels(), image.get_width(), image.get_height(), channel_count(image.get_pixel_format()),
                              save_as_fp16, filename.c_str(), &error_msg);
    } else {
        RGBA* pixel_data = new RGBA[image.get_pixel_count()];
        for (unsigned int y = 0; y < image.get_height(); ++y)
            for (unsigned int x = 0; x < image.get_width(); ++x) {
                int index = x + y * image.get_width();
                pixel_data[index] = image.get_pixel(Vector2ui(x, y));
            }

        int save_as_fp16 = 1;
        res = (Result)SaveEXR((float*)pixel_data, image.get_width(), image.get_height(), 4, save_as_fp16, filename.c_str(), &error_msg);

        delete[] pixel_data;
    }

    if (res != Result::Success)
        printf("TinyExr: %s\n", error_msg);

    return res;
}

} // NS TinyExr
