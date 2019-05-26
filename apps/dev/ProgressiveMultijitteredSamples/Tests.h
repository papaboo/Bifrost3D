// Progressive multi-jittered sample tests.
// ------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _PMJ_TESTS_H_
#define _PMJ_TESTS_H_

#include <Bifrost/Math/Constants.h>
#include <Bifrost/Math/Vector.h>

#include <vector>

namespace Test {

using namespace Bifrost::Math;
using namespace std;

inline float pow2(float v) { return v*v; }
inline double pow2(double v) { return v*v; }

// ------------------------------------------------------------------------------------------------
// Test sample properties.
// ------------------------------------------------------------------------------------------------

inline bool is_multijittered(const Vector2f* const samples, unsigned int sample_count) {
    auto strata = vector<bool>(sample_count);

    // Check that the samples are stratified along x.
    fill(strata.begin(), strata.end(), false);
    for (unsigned int s = 0; s < sample_count; ++s) {
        int stratum = int(sample_count * samples[s].x);
        strata[stratum] = true;
    }

    for (unsigned int s = 0; s < sample_count; ++s)
        if (!strata[s])
            return false;

    // Check that the samples are stratified along x.
    fill(strata.begin(), strata.end(), false);
    for (unsigned int s = 0; s < sample_count; ++s) {
        int stratum = int(sample_count * samples[s].y);
        strata[stratum] = true;
    }

    for (unsigned int s = 0; s < sample_count; ++s)
        if (!strata[s])
            return false;

    return true;
}

inline float compute_blue_noise_score(const Vector2f* const samples, unsigned int sample_count) {
    double error = 0.0;
    double squared_error = 0.0;
    for (unsigned int s = 0; s < sample_count; ++s) {
        Vector2f sample = samples[s];
        float shortest_distance = 2;
        for (unsigned int i = 0; i < sample_count; ++i) {
            if (s != i) {
                float distance = magnitude(samples[i] - sample);
                shortest_distance = fmin(shortest_distance, distance);
            }
        }
        error += shortest_distance;
        squared_error += shortest_distance * shortest_distance;
    }

    double mean_error = error / sample_count;
    double mean_squared_error = squared_error / sample_count;
    double variance = mean_squared_error - mean_error * mean_error;
    double std_dev = sqrt(fmax(0.0, variance));

    return float(std_dev / mean_error);
}

// ------------------------------------------------------------------------------------------------
// Convergence tests.
// See section 3 in Progressive Multi-Jittered Sample Sequences, Christensen et al., 2018
// ------------------------------------------------------------------------------------------------

// Estimate the area of a disc with center in (0,0) and radius 2/PI.
// The reference value is 0.5 and the function returns the unsigned deviation from 0.5.
float disc_convergence(const Vector2f* samples, int sample_count) {
    double area = 0.0f;
    auto* samples_end = samples + sample_count;
    while (samples != samples_end) {
        area += magnitude_squared(*samples) < (2 / PI<double>()) ? 1 : 0;
        ++samples;
    }

    return (float)abs(area / sample_count - 0.5f);
}

// Estimate the area of a triangle defined by sample.y < sample.x.
// The reference value is 0.5 and the function returns the unsigned deviation from 0.5.
float triangle_convergence(const Vector2f* samples, int sample_count) {
    double area = 0.0f;
    auto* samples_end = samples + sample_count;
    while (samples != samples_end) {
        area += samples->y < samples->x ? 1 : 0;
        ++samples;
    }

    return (float)abs(area / sample_count - 0.5f);
}

// Estimate the area of a step function with the transition at x < 1 / PI.
// The reference value is 1 / PI and the function returns the unsigned deviation from 1 / PI.
float step_convergence(const Vector2f* samples, int sample_count) {
    double area = 0.0f;
    auto* samples_end = samples + sample_count;
    while (samples != samples_end) {
        area += samples->x < 1 / PI<double>() ? 1 : 0;
        ++samples;
    }

    return (float)abs(area / sample_count - 1 / PI<double>());
}

// Estimate the integral of the smooth gaussian function e^(-x^2 - y^2)
// The reference value is PI/4 * erf^2(1) and the function returns the unsigned deviation from the reference value.
float gaussian_convergence(const Vector2f* samples, int sample_count) {
    double integral = 0.0f;
    auto* samples_end = samples + sample_count;
    while (samples != samples_end) {
        integral += exp(-pow2(samples->x) - pow2(samples->y));
        ++samples;
    }
    double ref_value = PI<double>() * 0.25 * pow2(erf(1));
    return (float)abs(integral / sample_count - ref_value);
}


// Estimate the integral of the smooth bilinar function given by sample.x * sample.y.
// The reference value is 0.25 and the function returns the unsigned deviation from 0.25.
float bilinear_convergence(const Vector2f* samples, int sample_count) {
    double integral = 0.0f;
    auto* samples_end = samples + sample_count;
    while (samples != samples_end) {
        integral += samples->x * samples->y;
        ++samples;
    }

    return (float)abs(integral / sample_count - 0.25);
}

} // NS Test

#endif // _PMJ_TESTS_H_