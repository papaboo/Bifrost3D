// Progressive multi-jittered sample distributions.
// Progressive Multi-Jittered Sample Sequences, Christensen et al., 2018
// Efficient Generation of Points that Satisfy Two-Dimensional Elementary Intervals, Matt Pharr, 2019
// ------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#include <ProgressiveJittered.h>
#include <ProgressiveMultijittered.h>

#include <Bifrost/Core/Engine.h>
#include <Bifrost/Core/Window.h>
#include <Bifrost/Input/Keyboard.h>

#include <StbImageWriter/stb_image_write.h>

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <GL/gl.h>
#undef RGB

#include <chrono>

using namespace Bifrost::Core;
using namespace Bifrost::Input;
using namespace Bifrost::Math;
using namespace std;

// ------------------------------------------------------------------------------------------------
// Tests
// ------------------------------------------------------------------------------------------------

bool is_multijittered(Vector2f* samples, unsigned int sample_count) {
    auto strata = vector<bool>(sample_count);

    // Check that the samples are stratified along x.
    fill(strata.begin(), strata.end(), false);
    for (unsigned int s = 0; s < sample_count; ++s) {
        int stratum = int(sample_count * samples[s].x);
        strata[stratum] = true;
    }

    for (unsigned int s = 0; s < sample_count; ++s)
        if (!strata[s])
            return false;

    // Check that the samples are stratified along x.
    fill(strata.begin(), strata.end(), false);
    for (unsigned int s = 0; s < sample_count; ++s) {
        int stratum = int(sample_count * samples[s].y);
        strata[stratum] = true;
    }

    for (unsigned int s = 0; s < sample_count; ++s)
        if (!strata[s])
            return false;

    return true;
}

// ------------------------------------------------------------------------------------------------
// Output image
// ------------------------------------------------------------------------------------------------

void output_image(const std::vector<Vector2f>& samples, const char* const path) {
    int width = int(samples.size());
    int height = int(samples.size());
    int channel_count = 1;

    unsigned char* pixels = new unsigned char[width * height * channel_count];
    fill(pixels, pixels + width * height * channel_count, 0);
    for (Vector2f sample : samples) {
        int x = int(sample.x * width);
        int y = int(sample.y * height);
        pixels[x + y * width] = 255;
    }

    bool did_succeed = stbi_write_png(path, width, height, channel_count, pixels, 0) != 0;
    if (!did_succeed)
        printf("Failed to output sample image to '%s'.\n", path);
}

int main(int argc, char** argv) {
    printf("Progressive multi-jittered distributions\n");

    // Generate samples.
    auto starttime = std::chrono::system_clock::now();
    auto samples = generate_progressive_multijittered_samples(6);
    auto endtime = std::chrono::system_clock::now();
    float delta_miliseconds = (float)std::chrono::duration_cast<std::chrono::milliseconds>(endtime - starttime).count();
    printf("Time to generate %d samples: %.3fseconds\n", int(samples.size()), delta_miliseconds / 1000);

    output_image(samples, "C:\\temp\\samples.png");

    // TODO Test if progressive??

    if (is_multijittered(samples.data(), (unsigned int)samples.size()))
        printf("... is multijittered\n");
    else
        printf("... is not multijittered\n");
}