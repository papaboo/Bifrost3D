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

#include <StbImageLoader/StbImageLoader.h>
#include <TinyExr/TinyExr.h>

#include <io.h>
#include <string>

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

#endif // _KOMODO_UTILS_H_