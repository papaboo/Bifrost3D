// Komodo utilities.
// ------------------------------------------------------------------------------------------------
// Copyright (C) 2015, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _KOMODO_UTILS_H_
#define _KOMODO_UTILS_H_

#include <Cogwheel/Assets/Image.h>
#include <Cogwheel/Core/Window.h>

#include <StbImageLoader/StbImageLoader.h>
#include <StbImageWriter/StbImageWriter.h>
#include <TinyExr/TinyExr.h>

#include <io.h>
#include <string>

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <GL/gl.h>
#undef RGB

inline bool validate_image_extension(const std::string& path) {
    std::string file_extension = std::string(path, path.length() - 4);
    if (!(file_extension.compare(".bmp") == 0 ||
        file_extension.compare(".exr") == 0 ||
        file_extension.compare(".hdr") == 0 ||
        file_extension.compare(".jpg") == 0 ||
        file_extension.compare(".png") == 0 ||
        file_extension.compare(".tga") == 0)) {
        printf("Unsupported file format: %s\nSupported formats are: bmp, exr, hdr, png and tga.\n", file_extension.c_str());
        return false;
    }

    return true;
}

inline Cogwheel::Assets::Image load_image(const std::string& path) {
    const int read_only_flag = 4;
    if (_access(path.c_str(), read_only_flag) >= 0) {
        std::string file_extension = std::string(path, path.length() - 3);
        if (file_extension.compare("exr") == 0)
            return TinyExr::load(path);
        else
            return StbImageLoader::load(path);
    }
    
    std::string new_path = path;

    // Test tga.
    new_path[path.size() - 3] = 't'; new_path[path.size() - 2] = 'g'; new_path[path.size() - 1] = 'a';
    if (_access(new_path.c_str(), read_only_flag) >= 0)
        return StbImageLoader::load(new_path);

    // Test png.
    new_path[path.size() - 3] = 'p'; new_path[path.size() - 2] = 'n'; new_path[path.size() - 1] = 'g';
    if (_access(new_path.c_str(), read_only_flag) >= 0)
        return StbImageLoader::load(new_path);

    // Test jpg.
    new_path[path.size() - 3] = 'j'; new_path[path.size() - 2] = 'p'; new_path[path.size() - 1] = 'g';
    if (_access(new_path.c_str(), read_only_flag) >= 0)
        return StbImageLoader::load(new_path);

    // Test exr.
    new_path[path.size() - 3] = 'e'; new_path[path.size() - 2] = 'x'; new_path[path.size() - 1] = 'r';
    if (_access(new_path.c_str(), read_only_flag) >= 0)
        return TinyExr::load(new_path);

    // No dice. Report error and return an invalid ID.
    printf("No image found at '%s'\n", path.c_str());
    return Cogwheel::Assets::Image();
}

inline void store_image(Cogwheel::Assets::Image image, const std::string& path) {
    std::string file_extension = std::string(path, path.length() - 3);
    if (file_extension.compare("exr") == 0)
        TinyExr::store(image.get_ID(), path);
    else
        StbImageWriter::write(image.get_ID(), path);
}

// Create a red and white error image.
inline Cogwheel::Assets::Images::UID create_error_image() {
    using namespace Cogwheel::Assets;

    Image error_img = Images::create2D("No images loaded", PixelFormat::RGBA32, 2.2f, Cogwheel::Math::Vector2ui(16, 16));
    unsigned char* pixels = (unsigned char*)error_img.get_pixels();
    for (unsigned int y = 0; y < error_img.get_height(); ++y) {
        for (unsigned int x = 0; x < error_img.get_width(); ++x) {
            unsigned char* pixel = pixels + (x + y * error_img.get_width()) * 4u;
            unsigned char intensity = ((x & 1) == (y & 1)) ? 2 : 255;
            pixel[0] = 255u;
            pixel[1] = pixel[2] = intensity;
            pixel[3] = 255u;
        }
    }
    return error_img.get_ID();
}

inline void render_image(Cogwheel::Core::Window& window, GLuint texture_ID, int image_width, int image_height) {
    glViewport(0, 0, window.get_width(), window.get_height());

    { // Setup matrices. I really don't need to do this every frame, since they never change.
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(-1, 1, -1.f, 1.f, 1.f, -1.f);

        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
    }

    glClear(GL_COLOR_BUFFER_BIT);

    float window_aspect = window.get_width() / float(window.get_height());
    float image_aspect = image_width / float(image_height);

    float x_coord = 1.0f, y_coord = 1.0f;
    if (window_aspect < image_aspect)
        // Vertical margin
        y_coord -= (image_aspect - window_aspect) / image_aspect;
    else if (window_aspect > image_aspect)
        // Horizontal margin
        x_coord -= (window_aspect - image_aspect) / window_aspect;

    glBindTexture(GL_TEXTURE_2D, texture_ID);
    glBegin(GL_QUADS); {

        glTexCoord2f(0.0f, 0.0f);
        glVertex3f(-x_coord, -y_coord, 0.f);

        glTexCoord2f(1.0f, 0.0f);
        glVertex3f(x_coord, -y_coord, 0.f);

        glTexCoord2f(1.0f, 1.0f);
        glVertex3f(x_coord, y_coord, 0.f);

        glTexCoord2f(0.0f, 1.0f);
        glVertex3f(-x_coord, y_coord, 0.f);

    } glEnd();
}

#endif // _KOMODO_UTILS_H_