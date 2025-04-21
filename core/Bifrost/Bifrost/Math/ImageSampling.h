// Bifrost image sampling methods
// ----------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ----------------------------------------------------------------------------

#ifndef _BIFROST_MATH_IMAGE_SAMPLING_H_
#define _BIFROST_MATH_IMAGE_SAMPLING_H_

#include <Bifrost/Math/Utils.h>

namespace Bifrost::Math::ImageSampling {

// Sample the 2D image defined by pixels, width and height at coordinate (u, v).
// The uv coordinates are clamped to the range [0, 1].
template <typename T>
inline T bilinear(T* pixels, int width, int height, float u, float v) {
    u = clamp(u, 0.0f, 1.0f);
    float u_coord = u * (width - 1);
    int lower_u_column = int(u_coord);
    int upper_u_column = min(lower_u_column + 1, width - 1);

    v = clamp(v, 0.0f, 1.0f);
    float v_coord = v * (height - 1);
    int lower_v_row = int(v_coord);
    int upper_v_row = min(lower_v_row + 1, height - 1);

    // Interpolate by u
    float u_t = u_coord - lower_u_column;
    const T* lower_pixel_row = pixels + lower_v_row * width;
    T lower_pixel = lerp(lower_pixel_row[lower_u_column], lower_pixel_row[upper_u_column], u_t);

    const T* upper_pixel_row = pixels + upper_v_row * width;
    T upper_pixel = lerp(upper_pixel_row[lower_u_column], upper_pixel_row[upper_u_column], u_t);

    // Interpolate by v
    float v_t = v_coord - lower_v_row;
    return lerp(lower_pixel, upper_pixel, v_t);
}

// Sample the 3D image defined by pixels, width, height and depth at coordinate (u, v, w).
// The sampling coordinates are clamped to the range [0, 1].
template <typename T>
inline T trilinear(T* pixels, int width, int height, int depth, float u, float v, float w) {
    w = clamp(w, 0.0f, 1.0f);
    float w_coord = w * (depth- 1);
    int lower_w_slice = int(w_coord);
    int upper_w_slice = min(lower_w_slice + 1, depth - 1);

    const T* lower_pixel_slice = pixels + lower_w_slice * width * height;
    const T* upper_pixel_slice = pixels + upper_w_slice * width * height;

    T lower_pixel = bilinear(lower_pixel_slice, width, height, u, v);
    T upper_pixel = bilinear(upper_pixel_slice, width, height, u, v);

    // Interpolate by w
    float w_t = w_coord - lower_w_slice;
    return lerp(lower_pixel, upper_pixel, w_t);
}

} // NS Bifrost::Math::ImageSampling

#endif //_BIFROST_MATH_IMAGE_SAMPLING_H_