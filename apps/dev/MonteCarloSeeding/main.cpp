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

unsigned int reverse_bits(unsigned int n) {
    // Reverse bits of n.
    n = (n << 16) | (n >> 16);
    n = ((n & 0x00ff00ff) << 8) | ((n & 0xff00ff00) >> 8);
    n = ((n & 0x0f0f0f0f) << 4) | ((n & 0xf0f0f0f0) >> 4);
    n = ((n & 0x33333333) << 2) | ((n & 0xcccccccc) >> 2);
    n = ((n & 0x55555555) << 1) | ((n & 0xaaaaaaaa) >> 1);
    return n;
}

// ------------------------------------------------------------------------------------------------
// Linear congruential random number generator.
// ------------------------------------------------------------------------------------------------
struct LinearCongruential {
private:
    static const unsigned int multiplier = 1664525u;
    static const unsigned int increment = 1013904223u;
    static const unsigned int max = 0xFFFFFFFFu; // uint32 max.

    unsigned int m_state;

public:
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

template <typename Sampler>
void test_sampler(int width, int height, int sample_count, const Sampler& sampler) {
    vector<Statistics<double> > per_pixel_stats; per_pixel_stats.resize(width * height);
    vector<Statistics<double> > per_neighbourhood_stats; per_neighbourhood_stats.resize(width * height);

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
    printf("Pixel RMS: %f\n", pixel_stats.rms());

    auto neighbourhood_stats = Statistics<double>::merge(per_neighbourhood_stats.begin(), per_neighbourhood_stats.end());
    printf("Neighbour RMS: %f\n", neighbourhood_stats.rms());
}

int main(int argc, char** argv) {
    printf("Monte carlo seeding strategies\n");

    int width = 128, height = 128, sample_count = 7;

    auto hash_sampler = [width](int x, int y, int sample_count) -> vector<float> {
        vector<float> samples; samples.resize(sample_count);
        int hash = RNG::hash(x + y * width);
        LinearCongruential rng;
        for (int s = 0; s < sample_count; ++s) {
            rng.seed(hash + reverse_bits(s));
            samples[s] = rng.sample1f();
        }
        return samples;
    };

    test_sampler(width, height, sample_count, hash_sampler);
}