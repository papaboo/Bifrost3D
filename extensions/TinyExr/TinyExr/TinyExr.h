// Bifrost exr loader and saver.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _BIFROST_TINY_EXR_H_
#define _BIFROST_TINY_EXR_H_

#include <Bifrost/Utils/IdDeclarations.h>
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
Result load_verbose(const std::string& filename, Bifrost::Assets::ImageID& image_ID);

inline Bifrost::Assets::ImageID load(const std::string& filename) {
    Bifrost::Assets::ImageID image_ID;
    load_verbose(filename, image_ID);
    return image_ID;
}

// -----------------------------------------------------------------------
// Store an exr image file.
// -----------------------------------------------------------------------
Result store(Bifrost::Assets::ImageID image_ID, const std::string& filename);

} // NS TinyExr

#endif // _BIFROST_TINY_EXR_H_
