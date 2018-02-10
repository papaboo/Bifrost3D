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

float average_log_luminance(Cogwheel::Assets::Images::UID image_ID) {
    Cogwheel::Assets::Image image = image_ID;
    int width = image.get_width(), height = image.get_height(), depth = image.get_depth();
    double summed_log_luminance = 0.0;
    for (int z = 0; z < depth; ++z)
        for (int y = 0; y < height; ++y)
            for (int x = 0; x < width; ++x) {
                auto pixel = image.get_pixel(Cogwheel::Math::Vector3ui(x, y, z)).rgb();
                summed_log_luminance += fmaxf(0.0001f, log2(Cogwheel::Math::luma(pixel)));
            }

	return float(summed_log_luminance / (width * height * depth));
}

} // NS Exposure
} // NS ImageOperations
