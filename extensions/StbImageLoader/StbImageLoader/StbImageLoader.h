// Bifrost stb image loader.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _BIFROST_ASSETS_STB_IMAGE_LOADER_H_
#define _BIFROST_ASSETS_STB_IMAGE_LOADER_H_

#include <Bifrost/Utils/IdDeclarations.h>
#include <string>

namespace StbImageLoader {

// -----------------------------------------------------------------------
// Loads an image file.
// Basic support for png, exr, jpg and more.
// -----------------------------------------------------------------------
Bifrost::Assets::ImageID load(const std::string& filename);
Bifrost::Assets::ImageID load_from_memory(const std::string& name, const void* const data, int data_byte_count);

} // NS StbImageLoader

#endif // _BIFROST_ASSETS_STB_IMAGE_LOADER_H_
