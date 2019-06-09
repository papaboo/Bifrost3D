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

#include <algorithm>
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

std::vector<Vector2f> generate_linear_congruential_random_samples(unsigned int sample_count) {
    auto samples = std::vector<Vector2f>(sample_count);

    auto rng = RNG::LinearCongruential(19349669);
    for (unsigned int s = 0; s < sample_count; ++s)
        samples[s] = { rng.sample1f(), rng.sample1f() };

    return samples;
}

std::vector<Vector3f> generate_3D_linear_congruential_random_samples(unsigned int sample_count) {
    auto samples = std::vector<Vector3f>(sample_count);

    auto rng = RNG::LinearCongruential(19349669);
    for (unsigned int s = 0; s < sample_count; ++s)
        samples[s] = { rng.sample1f(), rng.sample1f(), rng.sample1f() };

    return samples;
}

// ------------------------------------------------------------------------------------------------
// Output images of random samples.
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

// Flatten the samples along each dimension and output the 2D image.
void output_images(const std::vector<Vector3f>& samples, const char* const path) {

    auto flat_samples = std::vector<Vector2f>(samples.size());

    // remove the last 4 chars of the path, e.g. the extension and the .
    std::string path_sans_extension = std::string(path, path + strlen(path) - 4);

    for (int i = 0; i < samples.size(); ++i)
        flat_samples[i] = { samples[i].x, samples[i].y };
    output_image(flat_samples, (path_sans_extension + "_z.png").c_str());

    for (int i = 0; i < samples.size(); ++i)
        flat_samples[i] = { samples[i].x, samples[i].z };
    output_image(flat_samples, (path_sans_extension + "_y.png").c_str());

    for (int i = 0; i < samples.size(); ++i)
        flat_samples[i] = { samples[i].y, samples[i].z };
    output_image(flat_samples, (path_sans_extension + "_x.png").c_str());
}

// ------------------------------------------------------------------------------------------------
// Random number generators
// ------------------------------------------------------------------------------------------------

struct SampleGenerator {
private:
    std::vector<Vector2f>& m_samples;
    int m_index = 0;
    int m_dimension = 0;

public:
    SampleGenerator(std::vector<Vector2f>& samples, int index)
        : m_samples(samples), m_index(index) {}

    int get_index() const {
        return (m_index >> m_dimension) % m_samples.size();
    }

    Vector2f sample2f() {
        int index = get_index();
        m_dimension += 2;
        return m_samples[index];
    }

    float sample1f() {
        int index = get_index();
        m_dimension += 1;
        return m_samples[index].x;
    }
};

// ------------------------------------------------------------------------------------------------
// Options
// ------------------------------------------------------------------------------------------------

struct Options {
    Generator generator = Generator::PMJBN;
    int sample_count = 64;
    int dimensions = 2;
    int blue_noise_candidates = 8;
    bool output_images = false;

    static Options parse(int argc, char** argv) {
        Options res;
        for (int argument = 1; argument < argc; ++argument) {
            if (strcmp(argv[argument], "--lcg") == 0)
                res.generator = Generator::LCG;
            else if (strcmp(argv[argument], "--pj") == 0)
                res.generator = Generator::PJ;
            else if (strcmp(argv[argument], "--pmj") == 0)
                res.generator = Generator::PMJ;
            else if (strcmp(argv[argument], "--pmjbn") == 0)
                res.generator = Generator::PMJBN;
            else if (strcmp(argv[argument], "-s") == 0 || strcmp(argv[argument], "--samplecount") == 0)
                res.sample_count = next_power_of_two(atoi(argv[++argument]));
            else if (strcmp(argv[argument], "-d") == 0 || strcmp(argv[argument], "--dimensions") == 0)
                res.dimensions = atoi(argv[++argument]);
            else if (strcmp(argv[argument], "--output") == 0)
                res.output_images = true;
            else
                printf("Unsupported argument: '%s'\n", argv[argument]);
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
        "     | --output: Output the generated sample images to C:\\temp\\.\n";
    printf("%s", usage);
}

int main(int argc, char** argv) {
    printf("Progressive multi-jittered distributions\n");

    // Generate samples.
    if (argc == 1 || strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0) {
        print_usage();
        return 0;
    }

    auto options = Options::parse(argc, argv);

    printf("Generate %d %dD %s samples.\n", options.sample_count, options.dimensions, generator_name(options.generator));

    std::ostringstream path;
    path << "C:\\temp\\" << options.sample_count << "_" << generator_shortname(options.generator) << "_samples.png";

    if (options.dimensions == 2) {
        // Generate samples.
        auto generata_samples = [](Generator generator, int sample_count) -> std::vector<Vector2f> {
            switch (generator) {
            case Generator::LCG: return generate_linear_congruential_random_samples(sample_count);
            case Generator::PJ: return generate_progressive_jittered_samples(sample_count);
            case Generator::PMJ: return generate_progressive_multijittered_samples(sample_count);
            case Generator::PMJBN: return generate_progressive_multijittered_bluenoise_samples(sample_count);
            }
            return std::vector<Vector2f>();
        };

        auto starttime = std::chrono::system_clock::now();
        auto samples = generata_samples(options.generator, options.sample_count);
        auto endtime = std::chrono::system_clock::now();
        float delta_miliseconds = (float)std::chrono::duration_cast<std::chrono::milliseconds>(endtime - starttime).count();
        printf("Time to generate %d samples: %.3fseconds\n", int(samples.size()), delta_miliseconds / 1000);

        if (options.output_images) {
            printf("Output image to %s\n", path.str().c_str());
            output_image(samples, path.str().c_str());
        }

        { // Property tests.
            if (Test::is_multijittered(samples.data(), (unsigned int)samples.size()))
                printf("... is multijittered\n");
            else
                printf("... is not multijittered\n");

            float blue_noise_score = Test::compute_blue_noise_score(samples.data(), (unsigned int)samples.size());
            printf("... has blue noise score %f\n", blue_noise_score);
        }

        { // Convergence tests.
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

        { // Sample sequence tests.
            int distribution_count = 2;
            int bsdf_dimension_count = 2;
            int bsdf_count = bsdf_dimension_count * bsdf_dimension_count;
            int samples_pr_dimension = bsdf_count * distribution_count;
            int dimension_count = 2;
            int sample_bin_count = int(pow(samples_pr_dimension, dimension_count));
            auto bins = std::vector<int>(sample_bin_count);
            int samples_pr_bin = 1;

            for (int i = 0; i < samples_pr_bin * sample_bin_count; ++i) {
                int bin_begin = 0;
                int bin_end = sample_bin_count;
                auto partition_bins = [&](int index, int sample_count) {
                    int bins_pr_sample = (bin_end - bin_begin) / sample_count;
                    bin_begin += index * bins_pr_sample;
                    bin_end = bin_begin + bins_pr_sample;
                };

                printf("iteration %u:", i);

                auto rnd = SampleGenerator(samples, i + int(samples.size()));
                for (int d = 0; d < dimension_count; ++d) {
                    int index0 = rnd.get_index();
                    int distribution_index = int(rnd.sample1f() * distribution_count);
                    partition_bins(distribution_index, distribution_count);
                    int index1 = rnd.get_index();
                    Vector2f distribution_sample = rnd.sample2f();
                    Vector2i distribution_dimension_bin = Vector2i(distribution_sample * float(bsdf_dimension_count));
                    partition_bins(distribution_dimension_bin.x, bsdf_dimension_count);
                    partition_bins(distribution_dimension_bin.y, bsdf_dimension_count);
                    printf(" %d -> %d ->", index0, index1);
                }
                printf(" bin index %d\n", bin_begin);
                bins[bin_begin] += 1;
            }

            float bin_error = 0.0f;
            for (int b = 0; b < bins.size(); ++b)
                bin_error += abs(bins[b] - samples_pr_bin);
            bin_error /= (bins.size() * samples_pr_bin);

            printf("Bin count: [%d", bins[0]);
            for (int b = 1; b < bins.size(); ++b)
                printf(", %d", bins[b]);
            printf("]\n");
            printf("Bin error: %f\n", bin_error);
        }
    } else if (options.dimensions == 3) {
        // Generate samples.
        auto generata_samples = [](Generator generator, int sample_count) -> std::vector<Vector3f> {
            switch (generator) {
            case Generator::LCG: return generate_3D_linear_congruential_random_samples(sample_count);
            case Generator::PJ:
                printf("3D progressive jittered samples not supported. Generate PMJ samples instead.\n");
            case Generator::PMJ: return generate_3D_progressive_multijittered_samples(sample_count, 1);
            case Generator::PMJBN: return generate_3D_progressive_multijittered_samples(sample_count, 8);
            default:
                return std::vector<Vector3f>();
            }
        };

        auto starttime = std::chrono::system_clock::now();
        auto samples = generata_samples(options.generator, options.sample_count);
        auto endtime = std::chrono::system_clock::now();
        float delta_miliseconds = (float)std::chrono::duration_cast<std::chrono::milliseconds>(endtime - starttime).count();
        printf("Time to generate %d samples: %.3fseconds\n", int(samples.size()), delta_miliseconds / 1000);

        // Flatten along each dimension and save the resulting image.
        if (options.output_images) {
            printf("Output image to %s\n", path.str().c_str());
            output_images(samples, path.str().c_str());
        }

        { // Property tests.
            float blue_noise_score = Test::compute_blue_noise_score(samples.data(), (unsigned int)samples.size());
            printf("... has blue noise score %f\n", blue_noise_score);
        }

        { // Convergence tests.
            auto test = [&](const char* name, std::function<float(std::vector<Vector3f>&, int)> test) {
                printf("%s convergence:\n", name);
                printf("  1 sample: error %.4f\n", test(samples, 1));
                for (int s = 2; s <= samples.size(); s *= 2) {
                    printf("  %u samples: error %.4f\n", s, test(samples, s));
                    int ss = int(s * 1.5f);
                    if (ss < samples.size())
                        printf("  %u samples: error %.4f\n", ss, test(samples, ss));
                }
                printf("\n");
            };

            auto test_2D = [&](std::vector<Vector3f>& vs, int count) -> float {
                float split = 1 / PI<float>();
                auto second_partition_begin = std::partition(vs.begin(), vs.begin() + count, [&](Vector3f v) -> bool { return v.x < split;  });
                int first_partition_count = int(second_partition_begin - vs.begin());
                auto samples_2D = std::vector<Vector2f>(count);
                for (int s = 0; s < count; ++s)
                    samples_2D[s] = { vs[s].y, vs[s].z };

                float error1 = Test::step_convergence(samples_2D.data(), first_partition_count);
                float error2 = Test::gaussian_convergence(samples_2D.data() + first_partition_count, count - first_partition_count);

                float t = first_partition_count / float(count);

                return error1 * t + error2 * (1 - t);
            };

            test("Step and gaussian", test_2D);
        }
    }
}