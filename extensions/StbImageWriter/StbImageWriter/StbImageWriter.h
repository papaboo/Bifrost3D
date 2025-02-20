// Bifrost stb image writer.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _BIFROST_ASSETS_STB_IMAGE_WRITER_H_
#define _BIFROST_ASSETS_STB_IMAGE_WRITER_H_

#include <Bifrost/Assets/Image.h>
#include <Bifrost/Math/Color.h>
#include <string>

namespace StbImageWriter {

// -----------------------------------------------------------------------
// Writes an image file.
// Basic support for png, hdr, bmp and tga file formats.
// Future work
// * Return an actual error about why a file could not be written.
// -----------------------------------------------------------------------
bool write(Bifrost::Assets::Image image, const std::string& path);

inline bool write(Bifrost::Assets::ImageID imageID, const std::string& path) {
    return write(Bifrost::Assets::Image(imageID), path);
}

bool write(unsigned char* pixels, unsigned int width, unsigned int height, unsigned int channel_count, const std::string& path);
inline bool write(Bifrost::Math::RGB24* pixels, unsigned int width, unsigned int height, const std::string& path) {
    return write(&pixels[0].r.raw, width, height, 3, path);
}
inline bool write(Bifrost::Math::RGBA32* pixels, unsigned int width, unsigned int height, const std::string& path) {
    return write(&pixels[0].r.raw, width, height, 4, path);
}

bool write(float* pixels, unsigned int width, unsigned int height, unsigned int channel_count, const std::string& path);
inline bool write(Bifrost::Math::RGB* pixels, unsigned int width, unsigned int height, const std::string& path) {
    return write(&pixels[0].r, width, height, 3, path);
}
inline bool write(Bifrost::Math::RGBA* pixels, unsigned int width, unsigned int height, const std::string& path) {
    return write(&pixels[0].r, width, height, 4, path);
}

} // NS StbImageWriter

#endif // _BIFROST_ASSETS_STB_IMAGE_WRITER_H_