// Progressive multi-jittered sample distribution.
// Progressive Multi-Jittered Sample Sequences, Christensen et al., 2018
// ------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _PMJ_PMJ_H_
#define _PMJ_PMJ_H_

#include <Bifrost/Math/RNG.h>
#include <Bifrost/Math/Vector.h>

#include <vector>

// ------------------------------------------------------------------------------------------------
// Generate progressive mullti-jittered samples
// Progressive Multi-Jittered Sample Sequences - Supplemental materials, Christensen et al., 2018
// http://graphics.pixar.com/library/ProgressiveMultiJitteredSampling/pmj_suppl.pdf
// ------------------------------------------------------------------------------------------------

std::vector<Bifrost::Math::Vector2f> generate_progressive_multijittered_samples(unsigned int subdivisions) {
    using namespace Bifrost::Math;

    auto rng = RNG::LinearCongruential(19349669);
    auto rnd = [&]() -> float { return rng.sample1f(); };

    unsigned int m = 1 << subdivisions;
    unsigned int M = m * m;
    auto samples = std::vector<Vector2f>(M);
    unsigned int sample_count = 0;

    // Create occupied array
    auto occupied1Dx = std::vector<bool>(M);
    auto occupied1Dy = std::vector<bool>(M);

    // Create xhalves and yhalves used by extend_sequence_odd once. TODO Make bool/bit array
    auto xhalves = std::vector<int>(M / 2);
    auto yhalves = std::vector<int>(M / 2);

    auto generate_sample_point = [&](int i, int j, int xhalf, int yhalf, int n, int N) {
        int NN = 2 * N;

        Vector2f pt;
        // Generate candidate sample x coord
        do {
            pt.x = (i + 0.5f * (xhalf + rnd())) / n;
        } while (occupied1Dx[int(NN * pt.x)]);
        
        // Generate candidate sample y coord
        do {
            pt.y = (j + 0.5f * (yhalf + rnd())) / n;
        } while (occupied1Dy[int(NN * pt.y)]);
        
        // Mark 1D strata as occupied
        int xstratum = int(NN * pt.x);
        int ystratum = int(NN * pt.y);
        occupied1Dx[xstratum] = true;
        occupied1Dy[ystratum] = true;

        // Assign new sample point
        samples[sample_count] = pt;
        ++sample_count;
    };

    // Mark all occupied 1D strata.
    auto mark_occupied_strata = [&](unsigned int N) {
        unsigned int NN = 2 * N;
        for (unsigned int i = 0; i < NN; ++i)
            occupied1Dx[i] = occupied1Dy[i] = false; // init array

        for (unsigned int s = 0; s < N; ++s) {
            int xstratum = int(NN * samples[s].x);
            int ystratum = int(NN * samples[s].y);
            occupied1Dx[xstratum] = true;
            occupied1Dy[ystratum] = true;
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
            generate_sample_point(i, j, xhalf, yhalf, n, N);
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
        // Loop over N/2 old samples and generate 2 new samples for each – one at a time to keep the order consecutive (for "greedy" best candidates)
        
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
            generate_sample_point(i, j, xhalf, yhalf, n, N);
        }
            
        // And finally fill in the last subquadrants
        for (unsigned int s = 0; s < N / 2; ++s) {
            Vector2f oldpt = samples[s];
            int i = int(n * oldpt.x);
            int j = int(n * oldpt.y);
            int xhalf = 1 - xhalves[s];
            int yhalf = 1 - yhalves[s];
            
            // Generate a sample point
            generate_sample_point(i, j, xhalf, yhalf, n, N);
        }
    };

    samples[0] = { rnd(), rnd() };
    sample_count = 1;
    unsigned int N = 1;
    while (N < M) {
        extend_sequence_even(N); // N even pow2
        extend_sequence_odd(2 * N); // 2N odd pow2
        N *= 4;
    }

    return samples;
}

#endif // _PMJ_PMJ_H_