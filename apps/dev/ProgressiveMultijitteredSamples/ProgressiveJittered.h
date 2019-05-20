// Progressive jittered sample distribution.
// Progressive Multi-Jittered Sample Sequences, Christensen et al., 2018
// ------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _PMJ_PJ_H_
#define _PMJ_PJ_H_

#include <Bifrost/Math/RNG.h>
#include <Bifrost/Math/Vector.h>

#include <vector>

// ------------------------------------------------------------------------------------------------
// Generate progressive jittered samples
// Progressive Multi-Jittered Sample Sequences - Supplemental materials, Christensen et al., 2018
// http://graphics.pixar.com/library/ProgressiveMultiJitteredSampling/pmj_suppl.pdf
// ------------------------------------------------------------------------------------------------

std::vector<Bifrost::Math::Vector2f> generate_progressive_jittered_samples(unsigned int subdivisions) {
    using namespace Bifrost::Math;

    auto rnd = RNG::LinearCongruential(19349669);
    
    unsigned int m = 1 << subdivisions;
    unsigned int M = m * m;
    auto samples = std::vector<Vector2f>(M);
    
    auto generate_sample_point = [&](int i, int j, int xhalf, int yhalf, int n) -> Vector2f {
        float x = (i + 0.5f * (xhalf + rnd.sample1f())) / n;
        float y = (j + 0.5f * (yhalf + rnd.sample1f())) / n;
        return { x, y };
    };

    // Generate next 3N sample points.
    auto extend_sequence = [&](unsigned int N) {
        unsigned int n = (unsigned int)sqrt(N);
        // Loop over N old samples and generate 3 new samples for each old sample
        for (unsigned int s = 0; s < N; ++s) {
            // Determine sub-quadrant of existing sample point
            Vector2f oldpt = samples[s];
            int i = int(n * oldpt.x);
            int j = int(n * oldpt.y);
            int xhalf = int(2 * (n * oldpt.x - i));
            int yhalf = int(2 * (n * oldpt.y - j));

            // First select the diagonally opposite sub-quadrant
            xhalf = 1 - xhalf;
            yhalf = 1 - yhalf;
            samples[N + s] = generate_sample_point(i, j, xhalf, yhalf, n);

            // Then randomly select one of the two remaining sub-quadrants
            if (rnd.sample1f() > 0.5f)
                xhalf = 1 - xhalf;
            else
                yhalf = 1 - yhalf;
            samples[2 * N + s] = generate_sample_point(i, j, xhalf, yhalf, n);

            // And finally select the last sub-quadrant
            xhalf = 1 - xhalf;
            yhalf = 1 - yhalf;
            samples[3 * N + s] = generate_sample_point(i, j, xhalf, yhalf, n);
        }
    };

    samples[0] = { rnd.sample1f(), rnd.sample1f() };
    unsigned int N = 1;
    while (N < M) {
        extend_sequence(N);
        N *= 4;
    }

    return samples;
}

#endif // _PMJ_PJ_H_