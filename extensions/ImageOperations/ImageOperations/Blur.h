// Image blur operations.
// ------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License. 
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _IMAGE_OPERATIONS_BLUR_H_
#define _IMAGE_OPERATIONS_BLUR_H_

#include <Bifrost/Assets/Image.h>

namespace ImageOperations {
namespace Blur {

// ------------------------------------------------------------------------------------------------
// Guassian blur.
// ------------------------------------------------------------------------------------------------

inline void gaussian(Bifrost::Assets::Image image, float std_dev, Bifrost::Assets::Image result) {
    using namespace Bifrost::Assets;
    using namespace Bifrost::Math;

    Vector3ui size = image.get_size_3D();
    int pixel_count = size.x * size.y * size.z;

    int support = int(std_dev * 4.0f + 0.5f);
    float double_variance = 2.0f * std_dev * std_dev;

    auto filter = [=](RGB* pixels, int target_index, int stride, int min_index, int max_index) -> RGB {
        RGB summed_color = RGB::black();
        float total_weight = 0.0f;
        for (int i = -support; i <= support; ++i) {
            float weight = exp(-(i * i) / double_variance);

            int index = target_index + stride * i;
            if (min_index <= index && index < max_index) {
                summed_color += pixels[index] * weight;
                total_weight += weight;
            }
        }
        return summed_color / total_weight;
    };

    RGB* ping = new RGB[pixel_count];
    for (int i = 0; i < pixel_count; ++i)
        ping[i] = image.get_pixel(i).rgb();

    RGB* pong = new RGB[pixel_count];
    
    // Filter x
    if (size.x > 1) {
        #pragma omp parallel for schedule(dynamic, 16)
        for (int i = 0; i < pixel_count; ++i) {
            int min_index = (i / size.x) * size.x;
            pong[i] = filter(ping, i, 1, min_index, min_index + size.x);
        }
        std::swap(ping, pong);
    }

    // Filter y
    if (size.y > 1) {
        int range = size.x * size.y;
        #pragma omp parallel for schedule(dynamic, 16)
        for (int i = 0; i < pixel_count; ++i) {
            int min_index = (i / range) * range;
            pong[i] = filter(ping, i, size.x, min_index, min_index + range);
        }
        std::swap(ping, pong);
    }
    
    // Filter z TODO Optimize by storing directly in result.
    if (size.z > 1) {
        #pragma omp parallel for schedule(dynamic, 16)
        for (int i = 0; i < pixel_count; ++i)
            pong[i] = filter(ping, i, size.x * size.y, 0, pixel_count);
        std::swap(ping, pong);
    }

    for (int i = 0; i < pixel_count; ++i)
        result.set_pixel(ping[i], i);

    delete[] ping;
    delete[] pong;
}

inline Bifrost::Assets::Image gaussian(Bifrost::Assets::Image image, float std_dev) {
    using namespace Bifrost::Assets;

    Image result = Image::create3D("blurred_" + image.get_name(), image.get_pixel_format(), image.is_sRGB(), image.get_size_3D());
    gaussian(image, std_dev, result);
    return result;
}

} // NS Blur
} // NS ImageOperations

#endif // _IMAGE_OPERATIONS_BLUR_H_