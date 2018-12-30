// Cogwheel stb image loader.
// ---------------------------------------------------------------------------
// Copyright (C) 2015-2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _COGWHEEL_ASSETS_STB_IMAGE_LOADER_H_
#define _COGWHEEL_ASSETS_STB_IMAGE_LOADER_H_

#include <Cogwheel/Assets/Image.h>
#include <string>

namespace StbImageLoader {

// -----------------------------------------------------------------------
// Loads an image file.
// Basic support for png, exr, jpg and more.
// -----------------------------------------------------------------------
Cogwheel::Assets::Images::UID load(const std::string& filename);
Cogwheel::Assets::Images::UID load_from_memory(const std::string& name, const void* const data, int data_byte_count);

} // NS StbImageLoader

#endif // _COGWHEEL_ASSETS_STB_IMAGE_LOADER_H_