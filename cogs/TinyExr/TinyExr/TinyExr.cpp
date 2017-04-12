// Cogwheel exr loader and saver.
// ---------------------------------------------------------------------------
// Copyright (C) 2015-2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

// #pragma warning(disable : 4996)

#include <TinyExr/TinyExr.h>

#define TINYEXR_IMPLEMENTATION
#include <TinyExr/tiny_exr.h>

using namespace Cogwheel::Assets;
using namespace Cogwheel::Math;

namespace TinyExr {

Result load_verbose(const std::string& filename, Cogwheel::Assets::Images::UID& image_ID) {

    float* rgba = nullptr;
    int width, height;
    const char* error_msg=  nullptr;
    Result res = (Result)LoadEXR(&rgba, &width, &height, filename.c_str(), &error_msg);

    float image_gamma = 1.0f;
    image_ID = Images::create(filename, PixelFormat::RGBA_Float, image_gamma, Vector2ui(width, height));
    Images::PixelData pixel_data = Images::get_pixels(image_ID);
    memcpy(pixel_data, rgba, sizeof(float) * 4 * width * height);

    delete[] rgba;

    return res;
}

Result store(Cogwheel::Assets::Images::UID image_ID, const std::string& filename) {
    Image image = image_ID;
    if (image.get_pixel_format() == PixelFormat::RGBA_Float) {
        Images::PixelData pixel_data = Images::get_pixels(image_ID);
        return (Result)SaveEXR((float*)pixel_data, image.get_width(), image.get_height(), 4, filename.c_str());
    } else {
        RGBA* pixel_data = new RGBA[image.get_pixel_count()];
        for (unsigned int y = 0; y < image.get_height(); ++y)
            for (unsigned int x = 0; x < image.get_width(); ++x) {
                int index = x + y * image.get_width();
                pixel_data[index] = image.get_pixel(Vector2ui(x, y));
            }

        Result res = (Result)SaveEXR((float*)pixel_data, image.get_width(), image.get_height(), 4, filename.c_str());

        delete[] pixel_data;
    }

    return Result::Invalid_parameter;
}

} // NS TinyExr
