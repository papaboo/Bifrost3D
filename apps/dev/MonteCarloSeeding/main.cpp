// Test bed for different monte carlo seeding strategies..
// -----------------------------------------------------------------------------------------------
// Copyright (C) 2017, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// -----------------------------------------------------------------------------------------------

#include <Cogwheel/Math/RNG.h>
#include <Cogwheel/Math/Statistics.h>

#include <vector>
#include <limits.h>

using namespace Cogwheel::Math;
using namespace std;

// ------------------------------------------------------------------------------------------------
// Morton encoding
// ------------------------------------------------------------------------------------------------

// Insert a 0 bit in between each of the 16 low bits of v.
unsigned int part_by_1(unsigned int v) {
    v &= 0x0000ffff;                 // v = ---- ---- ---- ---- fedc ba98 7654 3210
    v = (v ^ (v << 8)) & 0x00ff00ff; // v = ---- ---- fedc ba98 ---- ---- 7654 3210
    v = (v ^ (v << 4)) & 0x0f0f0f0f; // v = ---- fedc ---- ba98 ---- 7654 ---- 3210
    v = (v ^ (v << 2)) & 0x33333333; // v = --fe --dc --ba --98 --76 --54 --32 --10
    v = (v ^ (v << 1)) & 0x55555555; // v = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
    return v;
}

unsigned int morton_encode(unsigned int x, unsigned int y) {
    return part_by_1(y) | (part_by_1(x) << 1);
}

// ------------------------------------------------------------------------------------------------
// Linear congruential random number generator.
// ------------------------------------------------------------------------------------------------
struct LinearCongruential {
private:
    unsigned int m_state;

public:
    static const unsigned int multiplier = 1664525u;
    static const unsigned int increment = 1013904223u;
    static const unsigned int max = 0xFFFFFFFFu; // uint32 max.

    void seed(unsigned int seed) { m_state = seed; }
    unsigned int get_seed() const { return m_state; }

    unsigned int sample1ui() {
        m_state = multiplier * m_state + increment;
        return m_state;
    }

    float sample1f() {
        const float inv_max = 1.0f / (float(max) + 1.0f);
        return float(sample1ui()) * inv_max;
    }
};

struct SeederStatistics {
    Statistics<double> pixel_stats;
    Statistics<double> neighbourhood_stats;
};

template <typename Seeder>
SeederStatistics seeder_statistics(int width, int height, int sample_count, const Seeder& seeder) {
    vector<Statistics<double> > per_pixel_stats; per_pixel_stats.resize(width * height);
    vector<Statistics<double> > per_neighbourhood_stats; per_neighbourhood_stats.resize(width * height);

    auto sampler = [width, seeder](int x, int y, int sample_count) -> vector<float> {
        vector<float> samples; samples.resize(sample_count);
        LinearCongruential rng;
        for (int s = 0; s < sample_count; ++s) {
            rng.seed(seeder(x, y, s));
            samples[s] = rng.sample1f();
        }
        return samples;
    };

    // Compute error, i.e. the distance in between the samples subtracted by the expected error.
    auto compute_error = [](vector<float>& samples) -> vector<float> {
        sort(samples.begin(), samples.end());

        int sample_count = (int)samples.size();
        float expected_distance = 1.0f / sample_count;
        vector<float> errors; errors.resize(sample_count);
        for (int s = 0; s < sample_count - 1; ++s) {
            float sample_distance = samples[s + 1] - samples[s];
            errors[s] = sample_distance - expected_distance;
        }
        // The error of the largest sample value is computed by wrapping around, e.g. add 1 to the lowest value.
        errors[sample_count - 1] = (samples[0] + 1.0f) - samples[sample_count - 1] - expected_distance;

        // Normalize error by sample count, so the error doesn't just magically drop as the sample count increases.
        for (int e = 0; e < sample_count; ++e)
            errors[e] *= sample_count;

        return errors;
    };

    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x) {
            int pixel_index = x + y * width;

            // Pr pixel error statistics.
            vector<float> pixel_samples = sampler(x, y, sample_count);
            vector<float> pixel_errors = compute_error(pixel_samples);
            per_pixel_stats[pixel_index] = Statistics<double>(pixel_errors.begin(), pixel_errors.end());

            // Compute error for neighbourhood around pixel. TODO togglable wrap around mode.
            pixel_samples.reserve(5 * sample_count);
            for (float s : sampler((x > 0 ? x : width) - 1, y, sample_count))
                pixel_samples.push_back(s);
            for (float s : sampler(x < width - 1 ? (x + 1) : 0, y, sample_count))
                pixel_samples.push_back(s);
            for (float s : sampler(x, (y > 0 ? y : height) - 1, sample_count))
                pixel_samples.push_back(s);
            for (float s : sampler(x, y < height - 1 ? (y + 1) : 0, sample_count))
                pixel_samples.push_back(s);
            vector<float> neighbour_errors = compute_error(pixel_samples);
            per_neighbourhood_stats[pixel_index] = Statistics<double>(neighbour_errors.begin(), neighbour_errors.end());
        }

    // Merge all stats
    auto pixel_stats = Statistics<double>::merge(per_pixel_stats.begin(), per_pixel_stats.end());
    auto neighbourhood_stats = Statistics<double>::merge(per_neighbourhood_stats.begin(), per_neighbourhood_stats.end());
    SeederStatistics stats = { pixel_stats, neighbourhood_stats };
    return stats;
}

template <typename Seeder>
void test_seeder(const std::string& name, int width, int height, int sample_count, const Seeder& seeder) {
    auto statistics = seeder_statistics(width, height, sample_count, seeder);

    // Output
    printf("%s:\n", name.c_str());
    printf("  Pixel RMS: %f\n", statistics.pixel_stats.rms());
    printf("  Neighbour RMS: %f\n", statistics.neighbourhood_stats.rms());
}

template <typename Seeder>
SeederStatistics seeder_statistics(int width, int height, int sample_count, int dimension_count, const Seeder& seeder) {
    auto statistics = seeder_statistics(width, height, sample_count, seeder);

    for (int i = 1; i < dimension_count; ++i) {
        auto seeder1 = [seeder, i](unsigned int x, unsigned int y, int sample) -> unsigned int {
            LinearCongruential rng; rng.seed(seeder(x, y, sample));
            for (int d = 0; d < i; ++d)
                rng.sample1ui();
            return rng.get_seed();
        };

        auto local_stats = seeder_statistics(width, height, sample_count, seeder1);
        statistics.pixel_stats.merge_with(local_stats.pixel_stats);
        statistics.neighbourhood_stats.merge_with(local_stats.neighbourhood_stats);
    }

    return statistics;
}

template <typename Seeder>
void test_seeder_in_dimensions(const std::string& name, int width, int height, int sample_count, int dimension_count, const Seeder& seeder) {
    auto statistics = seeder_statistics(width, height, sample_count, dimension_count, seeder);

    // Output
    printf("%s with %u dimensions:\n", name.c_str(), dimension_count);
    printf("  Pixel RMS: %f\n", statistics.pixel_stats.rms());
    printf("  Neighbour RMS: %f\n", statistics.neighbourhood_stats.rms());
}

// Distribute a set of ints inside a cube rotated by 45 degrees placed in a grid.
// Fx for radius 1 the pattern looks like this.
// | - | 1 | - |
// | 0 | 2 | 4 |
// | - | 3 | - |
// So far the valid patterns are only valid for the specific rombe that is tested and not when the rombe is moved along x or y.
// A more general approach could fix this.
void build_rombe_pattern(int radius) {
    auto print_grid = [](std::vector<int> grid, int grid_size) {
        for (int y = 0; y < grid_size; ++y) {
            printf("|");
            for (int x = 0; x < grid_size; ++x) {
                int v = grid[x + y * grid_size];
                if (v < 10)
                    printf("  %u |", v);
                else
                    printf(" %u |", v);
            }
            printf("\n");
        }
    };

    int grid_size = radius * 2 + 1;
    int cell_count = grid_size * grid_size;

    int internal_cell_count = radius * 2 + 1;
    for (int r = 0; r < radius; ++r)
        internal_cell_count += 2 * (r * 2 + 1);

    vector<int> grid; grid.resize(cell_count);
    std::vector<int> value_occurence; value_occurence.resize(internal_cell_count);
    for (int my = 0; my < grid_size; ++my)
        for (int mx = 0; mx < grid_size; ++mx) {
            // Clear value occurences.
            for (int i = 0; i < grid_size; ++i)
                value_occurence[i] = 0;

            // Fill grid
            for (int y = 0; y < grid_size; ++y)
                for (int x = 0; x < grid_size; ++x) {
                    int value = (x * mx + y * my) % internal_cell_count;
                    grid[x + y * grid_size] = value;

                    // If the distance as x + y from the center is less than or equal to the radius, then the cell is part of the pattern.
                    int distance = abs(x - radius) + abs(y - radius);
                    if (distance <= radius)
                        value_occurence[value] += 1;
                }

            // Check validity.
            bool pattern_valid = true;
            for (int i = 0; i < grid_size; ++i)
                pattern_valid &= value_occurence[i] == 1;

            if (pattern_valid) {
                printf("mx %u, my %u, is valid: %s\n", mx, my, pattern_valid ? "true" : "false");
                print_grid(grid, grid_size);
                printf("\n");
            }
        }
}

int main(int argc, char** argv) {
    printf("Monte carlo seeding strategies\n");

    int width = 128, height = 128, sample_count = 5;

    // Sampling initialized by jenkins hash.
    auto jenkins_hash = [width](int x, int y, int sample) -> unsigned int {
        return RNG::jenkins_hash(x + y * width) + reverse_bits(sample);
    };
    test_seeder("Jenkins hash", width, height, sample_count, jenkins_hash);
    test_seeder_in_dimensions("Jenkins hash", width, height, sample_count, 4, jenkins_hash);

    // Uniform sampling
    auto uniform_seeder = [width](int x, int y, int sample) -> unsigned int {
        return reverse_bits(sample);
    };
    test_seeder("Uniform", width, height, sample_count, uniform_seeder);

    // Morton encoding seed
    auto morton_seeder = [width](int x, int y, int sample) -> unsigned int {
        unsigned int encoded_index = reverse_bits(morton_encode(x, y));
        return (encoded_index ^ (encoded_index >> 16)) ^ (1013904223 * sample);
    };
    test_seeder("Morton encoding", width, height, sample_count, morton_seeder);

    // Sobol encoding seed
    auto sobol_seeder = [width](int x, int y, int sample) -> unsigned int {
        auto sobol2 = [](unsigned int n, unsigned int scramble) -> unsigned int {
            for (unsigned int v = 1u << 31u; n != 0; n >>= 1u, v ^= v >> 1u)
                if (n & 0x1) scramble ^= v;
            return scramble;
        };

        unsigned int encoded_index = reverse_bits(morton_encode(sobol2(x, 0u), y));
        return (encoded_index ^ (encoded_index >> 16)) + reverse_bits(sample);
    };
    test_seeder("Sobol encoding", width, height, sample_count, sobol_seeder);

    // Follows the pattern.
    // 0.0 | 0.4 | 0.8 | 0.2 | 0.6 | 0.0
    // 0.2 | 0.6 | 0.0 | 0.4 | 0.8 | 0.2
    // 0.4 | 0.8 | 0.2 | 0.6 | 0.0 | 0.4
    // Suboptimal for anything outside the 3x3 cross and quickly breaks down.
    auto optimal3x3 = [](unsigned int x, unsigned int y, int sample) -> unsigned int {
        unsigned int seed = x * 1717986918 + y * 858993459;
        seed = (seed - LinearCongruential::increment) / LinearCongruential::multiplier;
        return seed ^ reverse_bits(sample);
    };
    test_seeder("Optimial3x3 encoding", width, height, sample_count, optimal3x3);
    test_seeder_in_dimensions("Optimial3x3 encoding", width, height, sample_count, 4, optimal3x3);

    // Teschner et al, 2013
    auto teschner_2D_hash = [](unsigned int x, unsigned int y, int sample) -> unsigned int {
        return reverse_bits(RNG::teschner_hash(x, y) ^ sample);
    };
    test_seeder("Teschner 2D hash", width, height, sample_count, teschner_2D_hash);
    test_seeder_in_dimensions("Teschner 2D hash", width, height, sample_count, 4, teschner_2D_hash);

    // Teschner et al, 2013
    auto teschner_3D_hash = [](unsigned int x, unsigned int y, int sample) -> unsigned int {
        return reverse_bits(RNG::teschner_hash(x, y, sample));
    };
    test_seeder("Teschner 3D hash", width, height, sample_count, teschner_3D_hash);
    test_seeder_in_dimensions("Teschner 3D hash", width, height, sample_count, 4, teschner_3D_hash);

    // build_rombe_pattern(3);
}