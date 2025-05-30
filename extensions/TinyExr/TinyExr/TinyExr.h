// Bifrost exr loader and saver.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _BIFROST_TINY_EXR_H_
#define _BIFROST_TINY_EXR_H_

#include <Bifrost/Assets/Image.h>
#include <string>

namespace TinyExr {

enum class Result {
    Success = 0,
    Invalid_magic_number = -1,
    Invalid_exr_version = -2,
    Invalid_argument = -3,
    Invalid_data = -4,
    Invalid_file = -5,
    Invalid_parameter = -5,
    cant_open_file = -6,
    unsupported_format = -7,
    Invalid_header = -8
};

// -----------------------------------------------------------------------
// Load an exr image file.
// -----------------------------------------------------------------------
Result load_verbose(const std::string& filename, Bifrost::Assets::Image& image);

inline Bifrost::Assets::Image load(const std::string& filename) {
    Bifrost::Assets::Image image;
    load_verbose(filename, image);
    return image;
}

// -----------------------------------------------------------------------
// Store an exr image file.
// -----------------------------------------------------------------------
Result store(Bifrost::Assets::Image image, const std::string& filename);

} // NS TinyExr

#endif // _BIFROST_TINY_EXR_H_
