// Image exposure oprations.
// ------------------------------------------------------------------------------------------------
// Copyright (C) 2015-2018, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. 
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#include <ImageOperations/Exposure.h>

namespace ImageOperations {
namespace Exposure {

float summed_log_luminance(Cogwheel::Assets::Images::UID image_ID) {
    Cogwheel::Assets::Image image = image_ID;
    int width = image.get_width(), height = image.get_height(), depth = image.get_depth();
    double summed_log_luminance = 0.0;
    for (int z = 0; z < depth; ++z)
        for (int y = 0; y < height; ++y)
            for (int x = 0; x < width; ++x) {
                auto pixel = image.get_pixel(Cogwheel::Math::Vector3ui(x, y, z)).rgb();
                summed_log_luminance += log2(fmaxf(Cogwheel::Math::luma(pixel), 0.0001f));
            }

    return float(summed_log_luminance);
}

float log_average_luminance(Cogwheel::Assets::Images::UID image_ID) {
    Cogwheel::Assets::Image image = image_ID;
    // Corrects an error in the paper. We have to average summed log luminance BEFORE using exp2.
    // Otherwise the result is always going to be nearly or exactly zero.
    return exp2(summed_log_luminance(image_ID) / image.get_pixel_count());
}

} // NS Exposure
} // NS ImageOperations
