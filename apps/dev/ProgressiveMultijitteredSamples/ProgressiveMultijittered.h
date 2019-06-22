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
#include <Bifrost/Math/Utils.h>

#include <cassert>
#include <vector>

// ------------------------------------------------------------------------------------------------
// Generate progressive mullti-jittered samples
// Progressive Multi-Jittered Sample Sequences - Supplemental materials, Christensen et al., 2018
// http://graphics.pixar.com/library/ProgressiveMultiJitteredSampling/pmj_suppl.pdf
// ------------------------------------------------------------------------------------------------

std::vector<Bifrost::Math::Vector2f> generate_progressive_multijittered_samples(unsigned int sample_count) {
    using namespace Bifrost::Math;
    assert(is_power_of_two(sample_count));

    auto rng = RNG::LinearCongruential(19349669);
    auto rnd = [&]() -> float { return rng.sample1f(); };

    auto samples = std::vector<Vector2f>(); samples.reserve(sample_count);

    // Create occupied array
    auto occupied1Dx = std::vector<bool>(sample_count);
    auto occupied1Dy = std::vector<bool>(sample_count);

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
        samples.push_back(pt);
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
        
        // Loop over N old samples and generate 1 new sample for each if the sample buffer isn't full.
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
        
        // Select one of the two remaining subquadrants
        for (unsigned int s = 0; s < N / 2; ++s) {
            Vector2f oldpt = samples[s];
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
            generate_sample_point(i, j, xhalf, yhalf, n, N);
        }
            
        // And finally fill in the last subquadrants
        for (unsigned int s = 0; s < N / 2; ++s) {
            Vector2f oldpt = samples[s + N];
            int i = int(n * oldpt.x);
            int j = int(n * oldpt.y);

            int old_xhalf = int(2 * (n * oldpt.x - i));
            int old_yhalf = int(2 * (n * oldpt.y - j));

            int xhalf = 1 - old_xhalf;
            int yhalf = 1 - old_yhalf;

            // Generate a sample point
            generate_sample_point(i, j, xhalf, yhalf, n, N);
        }
    };

    samples.push_back( { rnd(), rnd() } );
    unsigned int N = 1;
    while (N < sample_count) {
        extend_sequence_even(N); // N even pow2
        if (2 * N < sample_count)
            extend_sequence_odd(2 * N); // 2N odd pow2
        N *= 4;
    }

    return samples;
}

#endif // _PMJ_PMJ_H_