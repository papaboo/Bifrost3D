// Bifrost random number generators and utilities.
// ----------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ----------------------------------------------------------------------------

#include <Bifrost/Math/RNG.h>
#include <Bifrost/Math/Utils.h>

#include <cassert>
#include <vector>

namespace Bifrost {
namespace Math {
namespace RNG {

// ------------------------------------------------------------------------------------------------
// Generate progressive multi-jittered samples with a blue noise approximation.
// Progressive Multi-Jittered Sample Sequences - Supplemental materials, Christensen et al., 2018
// http://graphics.pixar.com/library/ProgressiveMultiJitteredSampling/pmj_suppl.pdf.
// The nearest neighbour search is implemented by searcing nearby strata for their random samples.
// ------------------------------------------------------------------------------------------------
void fill_progressive_multijittered_bluenoise_samples(Vector2f* samples_begin, Vector2f* samples_end, unsigned int blue_noise_samples) {
    unsigned int sample_count = unsigned int(samples_end - samples_begin);
    assert(is_power_of_two(sample_count));

    auto rng = RNG::LinearCongruential(19349669);
    auto rnd = [&]() -> float { return rng.sample1f(); };

    unsigned short next_sample_index = 0;

    // Create occupied array.
    const unsigned short FREE_STRATUM = 65535;
    auto stratum_samples_x = std::vector<unsigned short>(sample_count);
    auto stratum_samples_y = std::vector<unsigned short>(sample_count);

    auto generate_sample_point = [&](Vector2f oldpt, int i, int j, int xhalf, int yhalf, int n, int N) {
        int NN = 2 * N;

        Vector2f best_pt = { NAN, NAN };
        float best_distance = 0;

        for (unsigned int s = 0; s < blue_noise_samples; ++s) {

            Vector2f pt;
            // Generate candidate sample x coord
            do {
                pt.x = (i + 0.5f * (xhalf + rnd())) / n;
            } while (stratum_samples_x[int(NN * pt.x)] != FREE_STRATUM);

            // Generate candidate sample y coord
            do {
                pt.y = (j + 0.5f * (yhalf + rnd())) / n;
            } while (stratum_samples_y[int(NN * pt.y)] != FREE_STRATUM);

            int xstratum = int(NN * pt.x);
            int ystratum = int(NN * pt.y);

            float distance_to_neighbour = magnitude(oldpt - pt);
            int max_search_stratum = int(NN * distance_to_neighbour);

            for (int offset = 1; offset <= max_search_stratum; ++offset) {
                auto test_neighbour_sample = [&](unsigned short neighbour_sample_index) {
                    if (neighbour_sample_index == FREE_STRATUM)
                        return;

                    auto neighbour_sample = samples_begin[neighbour_sample_index];
                    // Samples should be distributed wrt a repeating sample pattern, so modify the sample such that it is closest in this pattern.
                    if (neighbour_sample.x < pt.x - 0.5f)
                        neighbour_sample.x += 1.0f;
                    else if (neighbour_sample.x > pt.x + 0.5f)
                        neighbour_sample.x -= 1.0f;
                    if (neighbour_sample.y < pt.y - 0.5f)
                        neighbour_sample.y += 1.0f;
                    else if (neighbour_sample.y > pt.y + 0.5f)
                        neighbour_sample.y -= 1.0f;

                    distance_to_neighbour = fminf(distance_to_neighbour, magnitude(neighbour_sample - pt));
                    max_search_stratum = int(NN * distance_to_neighbour);
                };

                test_neighbour_sample(stratum_samples_x[(xstratum + offset) % stratum_samples_x.size()]);
                test_neighbour_sample(stratum_samples_x[(xstratum + stratum_samples_x.size() - offset) % stratum_samples_x.size()]);
                test_neighbour_sample(stratum_samples_y[(ystratum + offset) % stratum_samples_y.size()]);
                test_neighbour_sample(stratum_samples_y[(ystratum + stratum_samples_y.size() - offset) % stratum_samples_y.size()]);
            }

            if (best_distance < distance_to_neighbour) {
                best_distance = distance_to_neighbour;
                best_pt = pt;
            }
        }

        // Mark 1D strata as occupied
        int xstratum = int(NN * best_pt.x);
        int ystratum = int(NN * best_pt.y);
        stratum_samples_x[xstratum] = stratum_samples_y[ystratum] = next_sample_index;

        // Assign new sample point
        samples_begin[next_sample_index++] = best_pt;
    };

    // Mark all occupied 1D strata.
    auto mark_occupied_strata = [&](unsigned int N) {
        unsigned int NN = 2 * N;
        stratum_samples_x.resize(NN);
        stratum_samples_y.resize(NN);
        for (unsigned int i = 0; i < NN; ++i)
            stratum_samples_x[i] = stratum_samples_y[i] = FREE_STRATUM;

        for (unsigned int s = 0; s < N; ++s) {
            int xstratum = int(NN * samples_begin[s].x);
            int ystratum = int(NN * samples_begin[s].y);
            stratum_samples_x[xstratum] = stratum_samples_y[ystratum] = unsigned short(s);
        }
    };

    // Generate next N sample points(for N being an even power of two)
    auto extend_sequence_even = [&](unsigned int N) {
        unsigned int n = (unsigned int)sqrt(N);

        // Mark already occupied 1D strata so we can avoid them
        mark_occupied_strata(N);

        // Loop over N old samples and generate 1 new sample for each
        for (unsigned int s = 0; s < N; ++s) {
            Vector2f oldpt = samples_begin[s];
            int i = int(n * oldpt.x);
            int j = int(n * oldpt.y);
            int xhalf = int(2 * (n * oldpt.x - i));
            int yhalf = int(2 * (n * oldpt.y - j));
            // Select the diagonally opposite subquadrant
            xhalf = 1 - xhalf;
            yhalf = 1 - yhalf;
            // Generate a sample point
            generate_sample_point(oldpt, i, j, xhalf, yhalf, n, N);
        }
    };

    // Generate next N sample points(for N being an odd power of two)
    auto extend_sequence_odd = [&](unsigned int N) {
        unsigned int n = (unsigned int)sqrt(N / 2);
        // Mark already occupied 1D strata so we can avoid them
        mark_occupied_strata(N);

        // Loop over N/2 old samples and generate 2 new samples for each – one at a time to keep the order consecutive (for "greedy" best candidates)

        // Select one of the two remaining subquadrants
        for (unsigned int s = 0; s < N / 2; ++s) {
            Vector2f oldpt = samples_begin[s];
            int i = int(n * oldpt.x);
            int j = int(n * oldpt.y);
            int xhalf = int(2 * (n * oldpt.x - i));
            int yhalf = int(2 * (n * oldpt.y - j));

            // Randomly select one of the two remaining subquadrants
            if (rnd() > 0.5)
                xhalf = 1 - xhalf;
            else
                yhalf = 1 - yhalf;
            // Generate a sample point
            generate_sample_point(oldpt, i, j, xhalf, yhalf, n, N);
        }

        // And finally fill in the last subquadrants opposite to the previous subquadrant filled.
        for (unsigned int s = 0; s < N / 2; ++s) {
            Vector2f oldpt = samples_begin[s + N];
            int i = int(n * oldpt.x);
            int j = int(n * oldpt.y);

            int old_xhalf = int(2 * (n * oldpt.x - i));
            int old_yhalf = int(2 * (n * oldpt.y - j));

            int xhalf = 1 - old_xhalf;
            int yhalf = 1 - old_yhalf;

            // Generate a sample point
            generate_sample_point(oldpt, i, j, xhalf, yhalf, n, N);
        }
    };

    samples_begin[next_sample_index++] = { rnd(), rnd() };
    unsigned int N = 1;
    while (N < sample_count) {
        extend_sequence_even(N); // N even pow2
        if (2 * N < sample_count)
            extend_sequence_odd(2 * N); // 2N odd pow2
        N *= 4;
    }
}

} // NS RNG
} // NS Math
} // NS Bifrost