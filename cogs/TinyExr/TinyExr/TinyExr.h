// Cogwheel exr loader and saver.
// ---------------------------------------------------------------------------
// Copyright (C) 2015-2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _COGWHEEL_TINY_EXR_H_
#define _COGWHEEL_TINY_EXR_H_

#include <Cogwheel/Assets/Image.h>
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
Result load_verbose(const std::string& filename, Cogwheel::Assets::Images::UID& image_ID);

inline Cogwheel::Assets::Images::UID load(const std::string& filename) {
    Cogwheel::Assets::Images::UID image_ID;
    load_verbose(filename, image_ID);
    return image_ID;
}

// -----------------------------------------------------------------------
// Store an exr image file.
// -----------------------------------------------------------------------
Result store(Cogwheel::Assets::Images::UID image_ID, const std::string& filename);

} // NS TinyExr

#endif // _COGWHEEL_TINY_EXR_H_