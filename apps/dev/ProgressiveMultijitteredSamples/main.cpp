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

enum class Generator { LCG, PJ, PMJ, PMJBN };

const char* generator_name(Generator generator) {
    switch (generator) {
    case Generator::LCG: return "linear congruential";
    case Generator::PJ: return "progressive jittered";
    case Generator::PMJ: return "progressive multijittered";
    case Generator::PMJBN: return "progressive multijittered blue-noise";
    }
    return "unknown";
};

const char* generator_shortname(Generator generator) {
    switch (generator) {
    case Generator::LCG: return "LCG";
    case Generator::PJ: return "PJ";
    case Generator::PMJ: return "PMJ";
    case Generator::PMJBN: return "PMJBN";
    }
    return "unknown";
};

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

void print_usage() {
    char* usage =
        "usage:\n"
        "  -h | --help: Show command line usage.\n"
        "     | --lcg: Generate samples using a linear congruential random number generator.\n"
        "     | --pj: Generate samples using a progressive jittered generator.\n"
        "     | --pmj: Generate samples using a progressive multi-jittered generator.\n"
        "     | --pmjbn: Generate samples using a progressive multi-jittered pseudo blue noise generator.\n"
        "  -s | --subdivisions: Number of image subdivisions. Total sample count is subdivisions^4\n";
    printf("%s", usage);
}

int main(int argc, char** argv) {
    printf("Progressive multi-jittered distributions\n");

    // Generate samples.
    if (argc == 1 || strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0) {
        print_usage();
        return 0;
    }

    // Parse options
    Generator generator = Generator::PMJBN;
    int subdivisions = 3;

    for (int argument = 1; argument < argc; ++argument) {
        if (strcmp(argv[argument], "--lcg") == 0)
            generator = Generator::LCG;
        else if (strcmp(argv[argument], "--pj") == 0)
            generator = Generator::PJ;
        else if (strcmp(argv[argument], "--pmj") == 0)
            generator = Generator::PMJ;
        else if (strcmp(argv[argument], "--pmjbn") == 0)
            generator = Generator::PMJBN;
        else if (strcmp(argv[argument], "-s") == 0 || strcmp(argv[argument], "--subdivisions") == 0)
            subdivisions = atoi(argv[++argument]);
    }

    printf("Generate %s samples.\n", generator_name(generator));

    // Generate samples.
    auto generata_samples = [](Generator generator, int subdivisions) -> std::vector<Vector2f> {
        switch (generator) {
        case Generator::LCG: return generate_linear_congruential_random_samples(subdivisions);
        case Generator::PJ: return generate_progressive_jittered_samples(subdivisions);
        case Generator::PMJ: return generate_progressive_multijittered_samples(subdivisions);
        case Generator::PMJBN: return generate_progressive_multijittered_bluenoise_samples(subdivisions);
        }
        return std::vector<Vector2f>();
    };

    auto starttime = std::chrono::system_clock::now();
    auto samples = generata_samples(generator, subdivisions);
    auto endtime = std::chrono::system_clock::now();
    float delta_miliseconds = (float)std::chrono::duration_cast<std::chrono::milliseconds>(endtime - starttime).count();
    printf("Time to generate %d samples: %.3fseconds\n", int(samples.size()), delta_miliseconds / 1000);

    std::ostringstream path;
    path << "C:\\temp\\" << subdivisions << "_" << generator_shortname(generator) << "_samples.png";
    printf("Output image to %s\n", path.str().c_str());
    output_image(samples, path.str().c_str());

    // Property tests.
    // TODO Test if progressive??

    if (Test::is_multijittered(samples.data(), (unsigned int)samples.size()))
        printf("... is multijittered\n");
    else
        printf("... is not multijittered\n");

    float blue_noise_score = Test::compute_blue_noise_score(samples.data(), (unsigned int)samples.size());
    printf("... has blue noise score %f\n", blue_noise_score);

    // Convergence tests.
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