// Bifrost stb image writer.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#pragma warning(disable : 4996)

#include <StbImageWriter/StbImageWriter.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <StbImageWriter/stb_image_write.h>

using namespace Bifrost::Assets;
using namespace Bifrost::Math;

namespace StbImageWriter {

enum class FileType {
    BMP, HDR, JPG, PNG, TGA, Unknown
};

FileType get_file_type(const std::string& path) {
    if (path[path.size() - 4] != '.')
        return FileType::Unknown;

    const char* filetype_begin = path.data() + (path.size() - 3);
    if (strcmp("bmp", filetype_begin) == 0)
        return FileType::BMP;
    else if (strcmp("hdr", filetype_begin) == 0)
        return FileType::HDR;
    else if (strcmp("jpg", filetype_begin) == 0)
        return FileType::JPG;
    else if (strcmp("png", filetype_begin) == 0)
        return FileType::PNG;
    else if (strcmp("tga", filetype_begin) == 0)
        return FileType::TGA;
    else
        return FileType::Unknown;
}

bool save_image(const std::string& path, FileType file_type, unsigned int width, unsigned int height, int channel_count, void* pixels) {
    switch (file_type) {
    case FileType::BMP:
        return stbi_write_bmp(path.c_str(), width, height, channel_count, pixels) != 0;
    case FileType::HDR:
        return stbi_write_hdr(path.c_str(), width, height, channel_count, static_cast<float*>(pixels)) != 0;
    case FileType::JPG:
        return stbi_write_jpg(path.c_str(), width, height, channel_count, pixels, 90) != 0;
    case FileType::PNG:
        return stbi_write_png(path.c_str(), width, height, channel_count, pixels, 0) != 0;
    case FileType::TGA:
        return stbi_write_tga(path.c_str(), width, height, channel_count, pixels) != 0;
    default:
        printf("StbImageWriter found unsupported file type. Path: '%s'\n", path.c_str());
        return false;
    }
    return false;
}

// Flip texture vertically, as stb_image_writer uses the upper left corner as origo,
template <typename T>
std::vector<T> flip_horizontally(T* pixels, unsigned int width, unsigned int height, int channel_count) {
    auto flipped_pixels = std::vector<T>(width * height * channel_count);
    T* flipped_pixels_ptr = flipped_pixels.data();

    int row_element_count = width * channel_count;
    int row_size = row_element_count * sizeof(T);
    for (unsigned int row = 0; row < height; row++) {
        T* src_row = pixels + row * row_element_count;
        T* dst_row = flipped_pixels_ptr + (height - row - 1) * row_element_count;
        memcpy(dst_row, src_row, row_size);
    }

    return flipped_pixels;
}

bool write(Image image, const std::string& path) {

    if (image.get_depth() != 1)
        return false;

    FileType file_type = get_file_type(path);

    unsigned int width = image.get_width(), height = image.get_height();
    int channel_count = Bifrost::Assets::channel_count(image.get_pixel_format());

    // Flip texture vertically, as stb_image_writer uses the upper left corner as origo,
    // and ensure that the image format is correct.
    // NOTE This could be optimized if the image already has the correct format and gamma.
    bool did_succeed = false;
    if (file_type == FileType::HDR) {
        float* pixels = new float[width * height * channel_count];
        for (unsigned int y = 0; y < height; ++y)
            for (unsigned int x = 0; x < width; ++x) {
                int data_index = x + (height - 1 - y) * width;
                float* pixel_data = pixels + data_index * channel_count;
                RGBA pixel = image.get_pixel(Vector2ui(x, y));
                memcpy(pixel_data, pixel.begin(), sizeof(float) * channel_count);
            }

        did_succeed = save_image(path, file_type, width, height, channel_count, pixels);
        delete[] pixels;
    } else {
        unsigned char* pixels = new unsigned char[width * height * channel_count];
        float gamma = 1.0f / 2.2f;
        for (unsigned int y = 0; y < height; ++y)
            for (unsigned int x = 0; x < width; ++x) {
                int data_index = x + (height - 1 - y) * width;
                unsigned char* pixel_data = pixels + data_index * channel_count;
                RGBA pixel = image.get_pixel(Vector2ui(x, y));
                for (int c = 0; c < channel_count; ++c) {
                    float channel_intensity = c != 3 ? pow(pixel[c], gamma) : pixel[c]; // Gamma correct colors but leave alpha as linear.
                    pixel_data[c] = unsigned char(clamp(channel_intensity, 0.0f, 1.0f) * 255 + 0.5f);
                }
            }

        did_succeed = save_image(path, file_type, width, height, channel_count, pixels);
        delete[] pixels;
    }

    return did_succeed;
}

bool write(unsigned char* pixels, unsigned int width, unsigned int height, unsigned int channel_count, const std::string& path) {
    auto flipped_pixels = flip_horizontally(pixels, width, height, channel_count);

    FileType file_type = get_file_type(path);
    return save_image(path, file_type, width, height, channel_count, flipped_pixels.data());
}

bool write(float* pixels, unsigned int width, unsigned int height, unsigned int channel_count, const std::string& path) {
    auto flipped_pixels = flip_horizontally(pixels, width, height, channel_count);

    FileType file_type = get_file_type(path);
    return save_image(path, file_type, width, height, channel_count, flipped_pixels.data());
}

} // NS StbImageWriter
