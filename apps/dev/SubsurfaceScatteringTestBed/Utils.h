// Utils.
// ------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _SUBSURFACE_SCATTER_TEST_BED_UTILS_H_
#define _SUBSURFACE_SCATTER_TEST_BED_UTILS_H_

#include <Bifrost/Math/Distributions.h>
#include <Bifrost/Math/Ray.h>

#include <OptiXRenderer/RNG.h>

#include <functional>

struct ScatteringSobolRNG {
private:
    unsigned int m_path_index;
    unsigned int m_pixel_hash;
    unsigned int m_dimension;

public:
    ScatteringSobolRNG(unsigned int path_index, unsigned int pixel_hash = 0)
        : m_path_index(path_index), m_pixel_hash(pixel_hash), m_dimension(0) {}

    inline Bifrost::Math::Vector4f sample4f() {
        optix::float4 v = OptiXRenderer::RNG::PracticalScrambledSobol::sample4f(m_path_index, m_pixel_hash, m_dimension++);
        return { v.x, v.y, v.z, v.w };
    }
};

// Volumetric integration based on Ray Tracing Inhomogeneous Volumes, Ray Tracing Gems 1, chapter 28
template <class RNG>
inline void scatter_in_media(Bifrost::Math::Ray ray, float sigma_t, float single_scattering_albedo, RNG& rng,
                             std::function<float(Bifrost::Math::Ray)> distance_to_medium_boundary, std::function<void(Bifrost::Math::Ray, float, float)> ray_exiting_media,
                             int max_bounce_count = 250) {
    using namespace Bifrost::Math;

    float throughput = 1.0f;
    for (int bounce_count = 0; throughput > 0.0001f; ++bounce_count) {
        // Terminate the ray when max number of bounces is reached.
        if (bounce_count == max_bounce_count)
            return;

        // Draw all four samples at once to better support some QMC RNGs.
        Vector4f rng_sample = rng.sample4f();
        float distance_rng_sample = rng_sample.x;
        Vector2f direction_rng_sample = { rng_sample.y, rng_sample.z };
        float russian_roulette_sample = rng_sample.w;

        // Sample distance to next scattering event
        auto scattering_distance = Distributions::Exponential::sample_distance(sigma_t, distance_rng_sample);

        // Intersect with the scene
        float distance_to_surface = distance_to_medium_boundary(ray);
        bool ray_can_exit = !isinf(distance_to_surface);
        bool exits_medium = ray_can_exit && distance_to_surface < scattering_distance;

        // Check if the ray leaves the medium and apply light if that's the case.
        if (!exits_medium) {
            // Sample scattering direction from isotropic (spherical) distribution
            auto direction_sample = Distributions::Sphere::sample(direction_rng_sample);

            throughput *= single_scattering_albedo;

            // Russian roulette based on ray throughput.
            if (russian_roulette_sample < throughput)
                throughput = 1.0f; // Should be throughput / RR_decision_probability, but that simplifies to throughput / throughput in the one channel case.
            else
                throughput = 0.0f;

            // Create new ray from scattering event.
            Vector3f scattering_position = ray.position_at(scattering_distance);
            ray = Ray(scattering_position, direction_sample);
        } else {
            ray_exiting_media(ray, distance_to_surface, throughput);
            return;
        }
    }
}

#endif // _SUBSURFACE_SCATTER_TEST_BED_UTILS_H_
