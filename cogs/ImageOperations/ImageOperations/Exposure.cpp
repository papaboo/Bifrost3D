// Image exposure oprations.
// ------------------------------------------------------------------------------------------------
// Copyright (C) 2015-2018, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. 
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#include <ImageOperations/Exposure.h>

using namespace Cogwheel::Assets;
using namespace Cogwheel::Math;

namespace ImageOperations {
namespace Exposure {

float summed_log_luminance(Images::UID image_ID) {
    double summed_log_luminance = 0.0;
    Images::iterate_pixels(image_ID, [&](RGBA pixel) { summed_log_luminance += log2(fmaxf(luma(pixel.rgb()), 0.0001f)); });
    return float(summed_log_luminance);
}

float log_average_luminance(Images::UID image_ID) {
    Image image = image_ID;
    // Corrects an error in the paper. We have to average summed log luminance BEFORE using exp2.
    // Otherwise the result is always going to be nearly or exactly zero.
    return exp2(summed_log_luminance(image_ID) / image.get_pixel_count());
}

} // NS Exposure
} // NS ImageOperations
