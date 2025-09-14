// Environment map generator for simple environment maps
// ------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#include <Bifrost/Assets/Image.h>
#include <Bifrost/Math/Color.h>

#include <StbImageWriter/StbImageWriter.h>

#include <algorithm>
#include <cstdio>
#include <cstring>

using namespace Bifrost::Assets;
using namespace Bifrost::Math;

// ------------------------------------------------------------------------------------------------
// Options
// ------------------------------------------------------------------------------------------------

struct Options {
    std::string name = "EnvironmentMap";
    RGB background_color = RGB(0.1f);
    float ring_light_strength = 1.0f;
    float window_light_strength = 0.3f;

    // String representation is assumed to be "[r, g, b]".
    inline static RGB parse_RGB(const char* const rgb_str) {
        const char* r_begin = rgb_str + 1; // Skip [
        char* channel_end;
        float red = strtof(r_begin, &channel_end);

        char* g_begin = channel_end + 1; // Skip ,
        float green = strtof(g_begin, &channel_end);

        char* b_begin = channel_end + 1; // Skip ,
        float blue = strtof(b_begin, &channel_end);

        return RGB(red, green, blue);
    }

    static Options parse(int argc, char** argv) {
        char** argument_end = argv + argc;
        Options res;
        for (char** argument = argv + 1; argument < argument_end; ++argument) {
            if (strcmp(*argument, "--background-color") == 0)
                res.background_color = parse_RGB(*argument);
            else
                printf("Unsupported argument: '%s'\n", *argv);
        }
        return res;
    }
};

// ------------------------------------------------------------------------------------------------
// Main
// ------------------------------------------------------------------------------------------------

void print_usage() {
    char* usage =
        "usage:\n"
        "  -h | --help: Show command line usage.\n"
        "     | --lcg: Generate samples using a linear congruential random number generator.\n"
        "     | --pj: Generate samples using a progressive jittered generator.\n"
        "     | --pmj: Generate samples using a progressive multi-jittered generator.\n"
        "     | --pmjbn: Generate samples using a progressive multi-jittered pseudo blue noise generator.\n"
        "  -s | --samplecount: Number of samples generates. Will be rounded up to next power of two.\n"
        "  -d | --dimensions: Number of dimensions in the samples. 2 or 3 supported.\n"
        "     | --samplepattern2D: Pattern for translating the PMJ samples in 2D.\n"
        "     | --output: Output the generated sample images to C:\\temp\\.\n";
    printf("%s", usage);
}

int main(int argc, char** argv) {
    printf("Environment map generator\n");

    // Check if usage should be printed
    if (argc == 1 || strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0) {
        print_usage();
        return 0;
    }

    Images::allocate(1u);

    auto options = Options::parse(argc, argv);

    int height = 1024, width = 2 * height;
    Image map = Image::create2D("Environment map", PixelFormat::RGB24, true, Vector2ui(width, height));
    RGB24* pixels = map.get_pixels<RGB24>();

    // Fill in background color with slight gradient. Darker near the bottom and brighter near the top.
    RGB bottom_color = 0.8f * options.background_color;
    RGB top_color = 1.2f * options.background_color;
    for (int y = 0; y < height; y++) {
        float smooth_t = smoothstep(0.0f, 1.0f, (y + 0.5f) / height);
        RGB row_color_f = saturate(linear_to_sRGB(lerp(bottom_color, top_color, smooth_t)));
        RGB24 row_color = { row_color_f.r, row_color_f.g, row_color_f.b };

        RGB24* row_begin = pixels + y * width;
        std::fill(row_begin, row_begin + width, row_color);
    }

    // Add square window
    float sRGB_window_strength = linear_to_sRGB(options.window_light_strength);
    RGB24 window_color = { sRGB_window_strength, sRGB_window_strength, sRGB_window_strength };
    for (int y = int(height * 0.3f); y < int(height * 0.7f); y++) {
        RGB24* row_begin = pixels + y * width;
        for (int x = int(width * 0.45f); x < int(width * 0.7f); x++)
            row_begin[x] = window_color;
    }

    // Add ring light in center of map.
    float ring_radius = height * 0.05f;
    float ring_falloff_distance = ring_radius * 1.5f;
    int max_radius = int(ring_radius + ring_falloff_distance + 0.5f);
    for (int y = -max_radius; y <= max_radius; ++y) {
        int row_index = height / 2 + y;
        RGB24* row_begin = pixels + row_index * width;
        for (int x = -max_radius; x <= max_radius; ++x) {
            float dist_to_ring = sqrtf(float(x * x + y * y)) - ring_radius;
            float smooth_t = 1.0f - smoothstep(0.0f, ring_falloff_distance, abs(dist_to_ring));
            if (smooth_t > 0.0f) {
                int x_index = width / 2 + x;
                RGB24 pixel = row_begin[x_index];
                RGB sRGB_color = { pixel.r, pixel.g, pixel.b };
                RGB linear_color = sRGB_to_linear(sRGB_color);
                linear_color = saturate(linear_color + smooth_t * options.ring_light_strength);
                
                sRGB_color = linear_to_sRGB(linear_color);
                row_begin[x_index] = { sRGB_color.r, sRGB_color.g, sRGB_color.b };
            }
        }
    }

    StbImageWriter::write(map, "C:/Temp/EnvironmentMap.png");
}