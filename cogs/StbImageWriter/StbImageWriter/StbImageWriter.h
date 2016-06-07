// Cogwheel stb image writer.
// ---------------------------------------------------------------------------
// Copyright (C) 2015-2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _COGWHEEL_ASSETS_STB_IMAGE_WRITER_H_
#define _COGWHEEL_ASSETS_STB_IMAGE_WRITER_H_

#include <Cogwheel/Assets/Image.h>
#include <string>

namespace StbImageWriter {

// -----------------------------------------------------------------------
// Writes an image file.
// Basic support for png, exr, jpg and more.
// TODO
// * Return an actual error about why a file could not be loaded.
// * Define convention bout where (0,0) is in image2D space.
//   Should the images be flipped around vertically?
// -----------------------------------------------------------------------
bool write(const std::string& filename, Cogwheel::Assets::Image image);

inline bool write(const std::string& filename, Cogwheel::Assets::Images::UID imageID) {
    return write(filename, Cogwheel::Assets::Image(imageID));
}

} // NS StbImageWriter

#endif // _COGWHEEL_ASSETS_STB_IMAGE_WRITER_H_