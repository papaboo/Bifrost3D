// Bifrost stb image writer.
// ---------------------------------------------------------------------------
// Copyright (C) 2015-2016, Bifrost. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#pragma warning(disable : 4996)

#include <StbImageWriter/StbImageWriter.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <StbImageWriter/stb_image_write.h>

using namespace Bifrost::Assets;
using namespace Bifrost::Math;

namespace StbImageWriter {

enum class FileType {
    BMP, HDR, PNG, TGA, Unknown
};

enum class ChannelType {
    Char, Float, Unknown
};

FileType get_file_type(const std::string& path) {
    if (path[path.size() - 4] != '.')
        return FileType::Unknown;

    char type_first_letter = path[path.size() - 3];
    switch (type_first_letter) {
    case 'b':
    case 'B':
        return FileType::BMP;
    case 'h':
    case 'H':
        return FileType::HDR;
    case 'p':
    case 'P':
        return FileType::PNG;
    case 't':
    case 'T':
        return FileType::TGA;
    default:
        return FileType::Unknown;
    }
}

bool write(Image image, const std::string& path) {

    if (image.get_depth() != 1)
        return false;

    FileType file_type = get_file_type(path);
    if (file_type == FileType::Unknown) {
        printf("StbImageWriter found unsupported file type. Path: '%s'\n", path.c_str());
        return false;
    }
    unsigned int width = image.get_width(), height = image.get_height();
    int channel_count = Bifrost::Assets::channel_count(image.get_pixel_format());

    // Flip texture vertically, as stb_image_writer uses the upper left corner as origo,
    // and ensure that the image format is correct.
    // NOTE This could be optimized if the image already has the correct format and gamma.
    void* data = nullptr;
    if (file_type == FileType::HDR) {
        float* float_data = new float[width * height * channel_count];
        for (unsigned int y = 0; y < height; ++y)
            for (unsigned int x = 0; x < width; ++x) {
                int data_index = x + (height - 1 - y) * width;
                float* pixel_data = float_data + data_index * channel_count;
                RGBA pixel = image.get_pixel(Vector2ui(x, y));
                memcpy(pixel_data, pixel.begin(), sizeof(float) * channel_count);
            }
        data = float_data;
    } else {
        unsigned char* char_data = new unsigned char[width * height * channel_count];
        float gamma = 1.0f / 2.2f;
        for (unsigned int y = 0; y < height; ++y)
            for (unsigned int x = 0; x < width; ++x) {
                int data_index = x + (height - 1 - y) * width;
                unsigned char* pixel_data = char_data + data_index * channel_count;
                RGBA pixel = image.get_pixel(Vector2ui(x, y));
                for (int c = 0; c < channel_count; ++c) {
                    float channel_intensity = pow(pixel[c], gamma);
                    pixel_data[c] = unsigned char(clamp(channel_intensity, 0.0f, nearly_one) * 256);
                }
            }
        data = char_data;
    }

    bool did_succeed = false;
    switch (file_type) {
    case FileType::BMP:
        did_succeed = stbi_write_bmp(path.c_str(), width, height, channel_count, data) != 0;
        break;
    case FileType::HDR:
        did_succeed = stbi_write_hdr(path.c_str(), width, height, channel_count, static_cast<float*>(data)) != 0;
        break;
    case FileType::PNG:
        did_succeed = stbi_write_png(path.c_str(), width, height, channel_count, data, 0) != 0;
        break;
    case FileType::TGA:
        did_succeed = stbi_write_tga(path.c_str(), width, height, channel_count, data) != 0;
        break;
    case FileType::Unknown: // Silence compiler warning.
        break; // Cannot happen. Checked above.
    }

    delete[] data;

    return did_succeed;
}

} // NS StbImageWriter
