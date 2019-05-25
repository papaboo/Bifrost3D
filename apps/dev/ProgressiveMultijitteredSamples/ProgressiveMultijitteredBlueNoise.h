// Progressive multi-jittered blue noise sample distribution.
// Progressive Multi-Jittered Sample Sequences, Christensen et al., 2018
// ------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _PMJ_PMJBN_H_
#define _PMJ_PMJBN_H_

#include <Bifrost/Math/RNG.h>
#include <Bifrost/Math/Vector.h>

#include <vector>

// ------------------------------------------------------------------------------------------------
// Generate progressive mullti-jittered samples with a blue noise approximation.
// Progressive Multi-Jittered Sample Sequences - Supplemental materials, Christensen et al., 2018
// http://graphics.pixar.com/library/ProgressiveMultiJitteredSampling/pmj_suppl.pdf.
// The nearest neighbour search is implemented by searcing nearby strata for their random samples.
// ------------------------------------------------------------------------------------------------

std::vector<Bifrost::Math::Vector2f> generate_progressive_multijittered_bluenoise_samples(unsigned int subdivisions, unsigned int blue_noise_samples = 8) {
    using namespace Bifrost::Math;

    auto rng = RNG::LinearCongruential(19349669);
    auto rnd = [&]() -> float { return rng.sample1f(); };

    unsigned int m = 1 << subdivisions;
    unsigned int M = m * m;
    auto samples = std::vector<Vector2f>(); samples.reserve(M);

    // Create occupied array. The array is filled with the sample occupying it or nan if unoccupied.
    auto stratum_samples_x = std::vector<Vector2f>(M);
    auto stratum_samples_y = std::vector<Vector2f>(M);

    // Create xhalves and yhalves used by extend_sequence_odd once. TODO Make bool/bit array
    auto xhalves = std::vector<int>(M / 2);
    auto yhalves = std::vector<int>(M / 2);

    auto generate_sample_point = [&](Vector2f oldpt, int i, int j, int xhalf, int yhalf, int n, int N) {
        int NN = 2 * N;

        Vector2f best_pt = { NAN, NAN };
        float best_distance = 0;

        for (unsigned int s = 0; s < blue_noise_samples; ++s) {

            Vector2f pt;
            // Generate candidate sample x coord
            do {
                pt.x = (i + 0.5f * (xhalf + rnd())) / n;
            } while (!isnan(stratum_samples_x[int(NN * pt.x)].x));

            // Generate candidate sample y coord
            do {
                pt.y = (j + 0.5f * (yhalf + rnd())) / n;
            } while (!isnan(stratum_samples_y[int(NN * pt.y)].x));

            int xstratum = int(NN * pt.x);
            int ystratum = int(NN * pt.y);

            float distance_to_neighbour = magnitude(oldpt - pt);
            int max_search_stratum = int(NN * distance_to_neighbour);

            for (int offset = 1; offset <= max_search_stratum; ++offset) {
                auto test_neighbour_sample = [&](Vector2f neighbour_sample) {
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
        stratum_samples_x[xstratum] = best_pt;
        stratum_samples_y[ystratum] = best_pt;

        // Assign new sample point
        samples.push_back(best_pt);
    };

    // Mark all occupied 1D strata.
    auto mark_occupied_strata = [&](unsigned int N) {
        unsigned int NN = 2 * N;
        stratum_samples_x.resize(NN);
        stratum_samples_y.resize(NN);
        for (unsigned int i = 0; i < NN; ++i)
            stratum_samples_x[i] = stratum_samples_y[i] = { NAN, NAN }; // init array

        for (unsigned int s = 0; s < N; ++s) {
            int xstratum = int(NN * samples[s].x);
            int ystratum = int(NN * samples[s].y);
            stratum_samples_x[xstratum] = stratum_samples_y[ystratum] = samples[s];
        }
    };

    // Generate next N sample points(for N being an even power of two)
    auto extend_sequence_even = [&](unsigned int N) {
        unsigned int n = (unsigned int)sqrt(N);
        
        // Mark already occupied 1D strata so we can avoid them
        mark_occupied_strata(N);
        
        // Loop over N old samples and generate 1 new sample for each
        for (unsigned int s = 0; s < N; ++s) {
            Vector2f oldpt = samples[s];
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
        
        // Optionally:
        // 1) Classify occupied sub-pixels: odd or even diagonal
        // 2) Pre-select well-balanced subquadrants here for better sample distribution between powers of two samples)
        // Loop over N/2 old samples and generate 2 new samples for each � one at a time to keep the order consecutive (for "greedy" best candidates)
        
        // Select one of the two remaining subquadrants
        for (unsigned int s = 0; s < N / 2; ++s) {
            Vector2f oldpt = samples[s];
            int i = int(n * oldpt.x);
            int j = int(n * oldpt.y);
            int xhalf = int(2 * (n * oldpt.x - i));
            int yhalf = int(2 * (n * oldpt.y - j));

            // Randomly select one of the two remaining subquadrants (Or optionally use the well-balanced subquads chosen above)
            if (rnd() > 0.5)
                xhalf = 1 - xhalf;
            else
                yhalf = 1 - yhalf;
            xhalves[s] = xhalf;
            yhalves[s] = yhalf;
            // Generate a sample point
            generate_sample_point(oldpt, i, j, xhalf, yhalf, n, N);
        }
            
        // And finally fill in the last subquadrants
        for (unsigned int s = 0; s < N / 2; ++s) {
            Vector2f oldpt = samples[s];
            int i = int(n * oldpt.x);
            int j = int(n * oldpt.y);
            int xhalf = 1 - xhalves[s];
            int yhalf = 1 - yhalves[s];
            
            // Generate a sample point
            generate_sample_point(oldpt, i, j, xhalf, yhalf, n, N);
        }
    };

    samples.push_back( { rnd(), rnd() } );
    unsigned int N = 1;
    while (N < M) {
        extend_sequence_even(N); // N even pow2
        extend_sequence_odd(2 * N); // 2N odd pow2
        N *= 4;
    }

    return samples;
}

#endif // _PMJ_PMJBN_H_