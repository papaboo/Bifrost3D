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

namespace Bifrost {
namespace Math {
namespace RNG {

// ------------------------------------------------------------------------------------------------
// Generate progressive multi-jittered samples with a blue noise approximation.
// Progressive Multi-Jittered Sample Sequences - Supplemental materials, Christensen et al., 2018
// http://graphics.pixar.com/library/ProgressiveMultiJitteredSampling/pmj_suppl.pdf.
// The nearest neighbour search is implemented by searching nearby strata for their random samples.
// ------------------------------------------------------------------------------------------------
void fill_progressive_multijittered_bluenoise_samples(Vector2f* samples_begin, Vector2f* samples_end, unsigned int blue_noise_samples) {
    unsigned int total_sample_count = unsigned int(samples_end - samples_begin);
    assert(is_power_of_two(total_sample_count));

    auto rng = RNG::LinearCongruential(19349669);
    auto rnd = [&]() -> float { return rng.sample1f(); };

    blue_noise_samples = max(1u, blue_noise_samples);

    unsigned short next_sample_index = 0;

    // Create occupied array.
    const unsigned short FREE_STRATUM = 65535;
    auto* tmp_storage = new unsigned short[2 * total_sample_count];
    auto stratum_samples_x = tmp_storage;
    auto stratum_samples_y = tmp_storage + total_sample_count;

    auto generate_sample_point = [&](Vector2f oldpt, int i, int j, int xhalf, int yhalf, int prev_grid_size, int prev_sample_count) {
        int next_sample_count = 2 * prev_sample_count;

        Vector2f best_pt = { NAN, NAN };
        float best_distance = 0;

        for (unsigned int s = 0; s < blue_noise_samples; ++s) {

            Vector2f pt;
            // Generate candidate sample x coord
            do {
                pt.x = (i + 0.5f * (xhalf + rnd())) / prev_grid_size;
            } while (stratum_samples_x[int(next_sample_count * pt.x)] != FREE_STRATUM);

            // Generate candidate sample y coord
            do {
                pt.y = (j + 0.5f * (yhalf + rnd())) / prev_grid_size;
            } while (stratum_samples_y[int(next_sample_count * pt.y)] != FREE_STRATUM);

            int xstratum = int(next_sample_count * pt.x);
            int ystratum = int(next_sample_count * pt.y);

            float distance_to_neighbour = magnitude_squared(oldpt - pt);
            int max_search_stratum = int(next_sample_count * sqrt(distance_to_neighbour));

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

                    float local_distance_to_neighbour = magnitude_squared(neighbour_sample - pt);
                    if (local_distance_to_neighbour < distance_to_neighbour) {
                        distance_to_neighbour = local_distance_to_neighbour;
                        max_search_stratum = int(next_sample_count * sqrt(distance_to_neighbour));
                    }
                };

                test_neighbour_sample(stratum_samples_x[(xstratum + offset) % next_sample_count]);
                test_neighbour_sample(stratum_samples_x[(xstratum + next_sample_count - offset) % next_sample_count]);
                test_neighbour_sample(stratum_samples_y[(ystratum + offset) % next_sample_count]);
                test_neighbour_sample(stratum_samples_y[(ystratum + next_sample_count - offset) % next_sample_count]);
            }

            if (best_distance < distance_to_neighbour) {
                best_distance = distance_to_neighbour;
                best_pt = pt;
            }
        }

        // Mark 1D strata as occupied
        int xstratum = int(next_sample_count * best_pt.x);
        int ystratum = int(next_sample_count * best_pt.y);
        stratum_samples_x[xstratum] = stratum_samples_y[ystratum] = next_sample_index;

        // Assign new sample point
        samples_begin[next_sample_index++] = best_pt;
    };

    // Mark all occupied 1D strata.
    auto mark_occupied_strata = [&](unsigned int prev_sample_count) {
        unsigned int next_sample_count = 2 * prev_sample_count;
        for (unsigned int i = 0; i < next_sample_count; ++i)
            stratum_samples_x[i] = stratum_samples_y[i] = FREE_STRATUM;

        for (unsigned int s = 0; s < prev_sample_count; ++s) {
            int xstratum = int(next_sample_count * samples_begin[s].x);
            int ystratum = int(next_sample_count * samples_begin[s].y);
            stratum_samples_x[xstratum] = stratum_samples_y[ystratum] = unsigned short(s);
        }
    };

    // Generate next N sample points(for N being an even power of two)
    auto extend_sequence_even = [&](unsigned int prev_sample_count) {
        unsigned int prev_grid_size = (unsigned int)sqrt(prev_sample_count);

        // Mark already occupied 1D strata so we can avoid them
        mark_occupied_strata(prev_sample_count);

        // Loop over N old samples and generate 1 new sample for each
        for (unsigned int s = 0; s < prev_sample_count; ++s) {
            Vector2f oldpt = samples_begin[s];
            int i = int(prev_grid_size * oldpt.x);
            int j = int(prev_grid_size * oldpt.y);
            int xhalf = int(2 * (prev_grid_size * oldpt.x - i));
            int yhalf = int(2 * (prev_grid_size * oldpt.y - j));
            // Select the diagonally opposite subquadrant
            xhalf = 1 - xhalf;
            yhalf = 1 - yhalf;
            // Generate a sample point
            generate_sample_point(oldpt, i, j, xhalf, yhalf, prev_grid_size, prev_sample_count);
        }
    };

    // Generate next N sample points(for N being an odd power of two)
    auto extend_sequence_odd = [&](unsigned int prev_sample_count) {
        unsigned int prev_grid_size = (unsigned int)sqrt(prev_sample_count / 2);
        // Mark already occupied 1D strata so we can avoid them
        mark_occupied_strata(prev_sample_count);

        // Loop over the first half of the samples, the ones used in extend_sequence_even as well,
        // and generate 2 new samples for each – one at a time to keep the order consecutive (for "greedy" best candidates)

        // Select one of the two remaining subquadrants
        for (unsigned int s = 0; s < prev_sample_count / 2; ++s) {
            Vector2f oldpt = samples_begin[s];
            int i = int(prev_grid_size * oldpt.x);
            int j = int(prev_grid_size * oldpt.y);
            int xhalf = int(2 * (prev_grid_size * oldpt.x - i));
            int yhalf = int(2 * (prev_grid_size * oldpt.y - j));

            // Randomly select one of the two remaining subquadrants
            if (rnd() > 0.5)
                xhalf = 1 - xhalf;
            else
                yhalf = 1 - yhalf;
            // Generate a sample point
            generate_sample_point(oldpt, i, j, xhalf, yhalf, prev_grid_size, prev_sample_count);
        }

        // And finally fill in the last subquadrants opposite to the previous subquadrant filled.
        for (unsigned int s = 0; s < prev_sample_count / 2; ++s) {
            Vector2f oldpt = samples_begin[s + prev_sample_count];
            int i = int(prev_grid_size * oldpt.x);
            int j = int(prev_grid_size * oldpt.y);

            int old_xhalf = int(2 * (prev_grid_size * oldpt.x - i));
            int old_yhalf = int(2 * (prev_grid_size * oldpt.y - j));

            int xhalf = 1 - old_xhalf;
            int yhalf = 1 - old_yhalf;

            // Generate a sample point
            generate_sample_point(oldpt, i, j, xhalf, yhalf, prev_grid_size, prev_sample_count);
        }
    };

    samples_begin[next_sample_index++] = { rnd(), rnd() };
    unsigned int current_sample_count = 1;
    while (current_sample_count < total_sample_count) {
        extend_sequence_even(current_sample_count); // current_sample_count is even pow2
        if (2 * current_sample_count < total_sample_count)
            extend_sequence_odd(2 * current_sample_count); // 2 * current_sample_count is odd pow2
        current_sample_count *= 4;
    }

    delete[] tmp_storage;
}

} // NS RNG
} // NS Math
} // NS Bifrost