// Image blur operations.
// ------------------------------------------------------------------------------------------------
// Copyright (C) 2018, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. 
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _IMAGE_OPERATIONS_BLUR_H_
#define _IMAGE_OPERATIONS_BLUR_H_

#include <Cogwheel/Assets/Image.h>

namespace ImageOperations {
namespace Blur {

// ------------------------------------------------------------------------------------------------
// Guassian blur.
// ------------------------------------------------------------------------------------------------

void gaussian(Cogwheel::Assets::Images::UID image_ID, float std_dev, Cogwheel::Assets::Images::UID result_ID) {
    using namespace Cogwheel::Assets;
    using namespace Cogwheel::Math;

    Image image = image_ID;
    Vector3ui size = Vector3ui(image.get_width(), image.get_height(), image.get_depth());

    int pixel_count = size.x * size.y * size.z;

    int half_extent = int(std_dev * 4.0f + 0.5f);
    float double_variance = 2.0f * std_dev * std_dev;

    auto filter = [=](RGB* pixels, int target_index, int stride, int min_index, int max_index) -> RGB {
        RGB summed_color = RGB::black();
        float total_weight = 0.0f;
        for (int i = -half_extent; i <= half_extent; ++i) {
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

    Image result = result_ID;
    for (int i = 0; i < pixel_count; ++i)
        result.set_pixel(ping[i], i);

    delete[] ping;
    delete[] pong;
}

Cogwheel::Assets::Images::UID gaussian(Cogwheel::Assets::Images::UID image_ID, float std_dev) {
    using namespace Cogwheel::Assets;
    using namespace Cogwheel::Math;

    Image image = image_ID;
    Vector3ui size = Vector3ui(image.get_width(), image.get_height(), image.get_depth());
    Images::UID result = Images::create3D("blurred_" + image.get_name(), image.get_pixel_format(), image.get_gamma(), size);
    gaussian(image_ID, std_dev, result);
    return result;
}


// ------------------------------------------------------------------------------------------------
// Kawase blur.
// ------------------------------------------------------------------------------------------------


} // NS Blur
} // NS ImageOperations

#endif // _IMAGE_OPERATIONS_BLUR_H_