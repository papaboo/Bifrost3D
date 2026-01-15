// Plot subsurface scattering in a slab.
// ------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _SUBSURFACE_SCATTER_TEST_BED_PLOT_H_
#define _SUBSURFACE_SCATTER_TEST_BED_PLOT_H_

#include <Utils.h>

#include <Bifrost/Assets/Image.h>
#include <Bifrost/Assets/Media.h>
#include <Bifrost/Core/Array.h>
#include <Bifrost/Math/Intersect.h>
#include <Bifrost/Math/Plane.h>
#include <Bifrost/Math/Ray.h>

#include <OptiXRenderer/RNG.h>
#include <OptiXRenderer/Shading/BSDFs/BurleySSS.h>

#include <StbImageWriter/StbImageWriter.h>

namespace Plot {

using namespace Bifrost::Assets;
using namespace Bifrost::Assets::Media;
using namespace Bifrost::Core;
using namespace Bifrost::Math;
using namespace OptiXRenderer::Shading::BSDFs;

struct TestSetup {
    float diffuse_albedo;
    RGB plot_color;
};

static const int test_count = 5;
static TestSetup tests[test_count] = {
    { 0.1f, RGB(0.5f, 0, 0) },
    { 0.3f, RGB(0.4f, 0, 0.5f) },
    { 0.5f, RGB(0, 0.5f, 0) },
    { 0.7f, RGB(0.5f, 0.27f, 0) },
    { 0.9f, RGB(0, 0, 0.5f) },
};

// Recreate diffuse light experiment from Approximate Reflectance Profiles for Efficient Subsurface Scattering.
inline void plot_random_walk(float slab_thickness) {
    const int size = 200;
    const float max_radius = 8;
    const int path_count = 1048576;
    const float path_energy = 1.0f / path_count;

    // Storage for the reflectance at the discretized radii.
    Array<float> reflectance[test_count];
    #pragma omp parallel for schedule(dynamic, 1)
    for (int p = 0; p < test_count; ++p) {
        // Diffuse light illumination.
        // We scatter rays looking straight down through origi into the medium and illuminate them by a diffuse area light where they exit,
        // meaning that all exiting paths are illuminated equally.
        // Instead of only counting path that exit at a specific radius, which would be impossible, we instead associate each radius with a ring around origo.
        // The reflectance contribution to each ring is estimated using density estimation, similar to photon mapping.

        ArtisticScatteringParameters artistic_params = { RGB(tests[p].diffuse_albedo), RGB(1.0f) };
        auto scattering_params = MeasuredScatteringParameters::from_artistic_parameters(artistic_params);
        float sigma_t = scattering_params.get_attenuation_coefficient().r;
        float single_scattering_albedo = scattering_params.get_single_scattering_albedo().r;

        Plane front_surface = Plane::from_point_normal(Vector3f::zero(), Vector3f(0, 0, 1));
        Plane back_surface = Plane::from_point_normal(Vector3f(0, 0, -slab_thickness), Vector3f(0, 0, -1));

        auto distance_to_medium_boundary = [=](Ray ray) {
            float distance = INFINITY;

            float distance_to_front = intersect(ray, front_surface);
            if (distance_to_front >= 0.0f)
                distance = fminf(distance_to_front, distance);

            float distance_to_back = intersect(ray, back_surface);
            if (distance_to_back >= 0.0f)
                distance = fminf(distance_to_back, distance);

            return distance;
        };

        // Scatter paths
        reflectance[p] = Array<float>(size);
        std::fill_n(reflectance[p].begin(), size, 0.0f);
        for (int i = 0; i < path_count; i++) {
            auto rng = ScatteringSobolRNG(i);

            // Ray starting right past the surface at x,y and looking into the medium along negative z.
            Ray ray = Ray(Vector3f(0.0f, 0.0f, -1e-6f), Vector3f(0, 0, -1));

            auto ray_exiting_media = [&](Ray ray, float distance_to_surface, float throughput) {
                Vector3f medium_exit_position = ray.position_at(distance_to_surface);

                // Discard samples exiting backside
                bool ray_exited_frontside = ray.direction.z >= 0.0f;
                if (!ray_exited_frontside)
                    return;

                // Store sample in radius reflectance bin.
                float radius = magnitude(Vector2f(medium_exit_position.x, medium_exit_position.y));
                int radius_bin = int(radius / max_radius * size);
                if (radius_bin < size)
                    reflectance[p][radius_bin] += throughput * path_energy;
            };

            scatter_in_media(ray, sigma_t, single_scattering_albedo, rng, distance_to_medium_boundary, ray_exiting_media);
        }
    }

    // Normalize the path contribution by dividing by the area of the region that the paths contributed to.
    float bin_width = max_radius / size;
    for (int radius_ring_index = 0; radius_ring_index < size; ++radius_ring_index) {
        float inner_radius = radius_ring_index * bin_width;
        float inner_area = PI<float>() * pow2(inner_radius);
        float outer_radius = (radius_ring_index + 1) * bin_width;
        float outer_area = PI<float>() * pow2(outer_radius);
        float bin_ring_area = outer_area - inner_area;

        for (int i = 0; i < test_count; ++i)
            reflectance[i][radius_ring_index] /= bin_ring_area;
    }

    // Output image
    Image plot = Image::create2D("Random walk plot", PixelFormat::RGB24, true, Vector2ui(size, size));
    plot.clear(RGBA::white());
    for (int i = 0; i < test_count; ++i) {
        RGBA plot_color = tests[i].plot_color;
        for (int radius_ring_index = 0; radius_ring_index < size; ++radius_ring_index) {
            float radius = max_radius * (radius_ring_index + 0.5f) / size;
            float f = reflectance[i][radius_ring_index];
            float y = radius * f; // Y-axis is r*R(r)

            int plot_y = (int)round(10.0f * size * y);
            if (plot_y < size)
                plot.set_pixel(plot_color, Vector2ui(radius_ring_index, plot_y));
        }
    }

    std::string output_path = "C:/Temp/plot_diffuse_light_random_walk_slab_thickness_" + std::to_string(slab_thickness) + ".png";
    StbImageWriter::write(plot, output_path);
}

// Recreate search light and diffuse light experiment from Approximate Reflectance Profiles for Efficient Subsurface Scattering.
inline void plot_burley(BurleySSS::Parameters::LightConfig light_config, float slab_thickness) {
    using namespace optix;

    int size = 200;
    Image plot = Image::create2D("Burley plot", PixelFormat::RGB24, true, Vector2ui(size, size));
    plot.clear(RGBA::white());

    float max_radius = 8;
    float mean_free_path = 1.0f;
    for (auto test : tests) {
        float albedo = test.diffuse_albedo;
        RGBA plot_color = test.plot_color;

        auto params = BurleySSS::Parameters::create({ albedo, albedo, albedo }, { mean_free_path, mean_free_path, mean_free_path }, light_config);
        for (int x = 0; x < size; ++x) {
            float radius = max_radius * (x + 0.5f) / size;
            float f = params.diffuse_albedo.x * BurleySSS::evaluate(radius, params.diffuse_mean_free_path.x);

            // Compute energy lost due to light scattering out the backside.
            float f_energy_loss = 0.0f;
            if (!isinf(slab_thickness)) {
                float backside_distance = sqrt(pow2(radius) + pow2(slab_thickness));
                f_energy_loss = params.diffuse_albedo.x * BurleySSS::evaluate(backside_distance, params.diffuse_mean_free_path.x);
            }

            float y = radius * (f - f_energy_loss); // Y-axis is r*R(r)

            int plot_y = (int)round(10.0f * size * y);
            if (plot_y < size)
                plot.set_pixel(plot_color, Vector2ui(x, plot_y));
        }
    }

    // Output image
    std::string light_config_str = light_config == BurleySSS::Parameters::LightConfig::Search ? "search" : "diffuse";
    std::string output_path = "C:/Temp/plot_" + light_config_str + "_light_burley_slab_thickness_" + std::to_string(slab_thickness) + ".png";
    StbImageWriter::write(plot, output_path);
}

inline void plot(float slab_thickness) {
    plot_random_walk(slab_thickness);
    plot_burley(BurleySSS::Parameters::LightConfig::Search, slab_thickness);
    plot_burley(BurleySSS::Parameters::LightConfig::Diffuse, slab_thickness);
}

} // NS Plot

#endif // _SUBSURFACE_SCATTER_TEST_BED_PLOT_H_
