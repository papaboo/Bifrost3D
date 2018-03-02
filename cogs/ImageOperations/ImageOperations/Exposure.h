// Image exposure oprations.
// ------------------------------------------------------------------------------------------------
// Copyright (C) 2015-2018, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. 
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _IMAGE_OPERATIONS_EXPOSURE_H_
#define _IMAGE_OPERATIONS_EXPOSURE_H_

#include <Cogwheel/Assets/Image.h>

namespace ImageOperations {
namespace Exposure {

float summed_log_luminance(Cogwheel::Assets::Images::UID image_ID);

// Implements equation (1) in Reinhard et al, 2002, Photographic Tone Reproduction for Digital Images.
float log_average_luminance(Cogwheel::Assets::Images::UID image_ID);

template <typename ForwardIterator>
inline void log_luminance_histogram(Cogwheel::Assets::Images::UID image_ID, float min_log_luminance, float max_log_luminance,
                                    ForwardIterator begin, ForwardIterator end) {
    using namespace Cogwheel::Math;

    int size = unsigned int(end - begin);
    Cogwheel::Assets::Images::iterate_pixels(image_ID, [&](RGBA pixel) {
        float log_luminance = log2(fmaxf(luminance(pixel.rgb()), 0.0001f));
        float normalized_index = inverse_lerp(min_log_luminance, max_log_luminance, log_luminance);
        int index = clamp(int(normalized_index * size), 0, size - 1);
        ++begin[index];
    });
}

} // NS Exposure
} // NS ImageOperations

#endif // _IMAGE_OPERATIONS_EXPOSURE_H_