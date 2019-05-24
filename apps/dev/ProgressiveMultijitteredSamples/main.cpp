// Progressive multi-jittered sample distributions.
// Progressive Multi-Jittered Sample Sequences, Christensen et al., 2018
// Efficient Generation of Points that Satisfy Two-Dimensional Elementary Intervals, Matt Pharr, 2019
// ------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#include <Tests.h>
#include <ProgressiveJittered.h>
#include <ProgressiveMultijittered.h>
#include <ProgressiveMultijitteredBlueNoise.h>

#include <StbImageWriter/stb_image_write.h>

#include <chrono>
#include <functional>

using namespace Bifrost::Math;
using namespace std;

std::vector<Vector2f> generate_linear_congruential_random_samples(unsigned int subdivisions) {
    unsigned int m = 1 << subdivisions;
    unsigned int M = m * m;
    auto samples = std::vector<Vector2f>(M);

    auto rng = RNG::LinearCongruential(19349669);
    for (unsigned int s = 0; s < M; ++s)
        samples[s] = { rng.sample1f(), rng.sample1f() };

    return samples;
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
    auto samples = generate_progressive_multijittered_bluenoise_samples(3);
    auto endtime = std::chrono::system_clock::now();
    float delta_miliseconds = (float)std::chrono::duration_cast<std::chrono::milliseconds>(endtime - starttime).count();
    printf("Time to generate %d samples: %.3fseconds\n", int(samples.size()), delta_miliseconds / 1000);

    output_image(samples, "C:\\temp\\samples.png");

    // TODO Test if progressive??

    if (Test::is_multijittered(samples.data(), (unsigned int)samples.size()))
        printf("... is multijittered\n");
    else
        printf("... is not multijittered\n");

    // Convergence tests
    auto test = [&](const char* name, std::function<float(Vector2f*, int)> test) {
        printf("%s convergence:\n", name);
        printf("  1 sample: error %.4f\n", test(samples.data(), 1));
        for (int s = 2; s <= samples.size(); s *= 2)
            printf("  %u samples: error %.4f\n", s, test(samples.data(), s));
        printf("\n");
    };

    test("Disc", Test::disc_convergence);
    test("Triangle", Test::triangle_convergence);
    test("Step", Test::step_convergence);
    test("Gaussian", Test::gaussian_convergence);
    test("Bilinear", Test::bilinear_convergence);
}