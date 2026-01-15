// Subsurface scattering integration on slab
// ------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _SUBSURFACE_SCATTER_TEST_BED_SLAB_INTEGRATION_H_
#define _SUBSURFACE_SCATTER_TEST_BED_SLAB_INTEGRATION_H_

#include <Utils.h>

#include <Bifrost/Assets/Image.h>
#include <Bifrost/Assets/Media.h>
#include <Bifrost/Math/Intersect.h>
#include <Bifrost/Math/Plane.h>
#include <Bifrost/Math/Ray.h>
#include <Bifrost/Math/RNG.h>

#include <OptiXRenderer/Shading/BSDFs/BurleySSS.h>

#include <StbImageWriter/StbImageWriter.h>

namespace Integrate {

using namespace Bifrost::Assets;
using namespace Bifrost::Assets::Media;
using namespace Bifrost::Math;

const float pixel_scale = 0.2f;
const int bssrdf_count = 5;
ArtisticScatteringParameters sss_params[bssrdf_count] = { ArtisticScatteringParameters::potato(),
                                                          ArtisticScatteringParameters::marble(),
                                                          ArtisticScatteringParameters::wholemilk(),
                                                          ArtisticScatteringParameters::skin1(),
                                                          ArtisticScatteringParameters::ketchup() };

// The slabs are lit by a 2mm wide light source mirroredd around x=0.
auto is_in_shadow = [=](float position_x) -> bool {
    return position_x < -1 || position_x > 1;
};

inline void random_walk_integrate(Image output, float slab_thickness) {
    const int width = output.get_width();
    const int path_count = 16384;

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

    for (int y = 0; y < bssrdf_count; ++y) {
        // Scattering parameters
        MeasuredScatteringParameters params = MeasuredScatteringParameters::from_artistic_parameters(sss_params[y]);
        RGB sigma_t = params.get_attenuation_coefficient();
        RGB single_scattering_albedo = params.get_single_scattering_albedo();

        #pragma omp parallel for schedule(dynamic, 5)
        for (int x = 0; x < width; ++x) {

            RGB radiance = { 0.0f, 0.0f, 0.0f };
            for (int c = 0; c < 3; ++c) {
                for (int i = 0; i < path_count; i++) {
                    auto rng = ScatteringSobolRNG(i, OptiXRenderer::RNG::pcg2d(x, y).x);

                    // Ray starting right past the surface at x,y and looking into the medium along negative z.
                    Ray ray = Ray(Vector3f(float(x), float(y), -1e-6f) * pixel_scale, Vector3f(0, 0, -1));

                    auto ray_exiting_media = [&](Ray ray, float distance_to_surface, float throughput) {
                        Vector3f medium_exit_position = ray.position_at(distance_to_surface);

                        // Apply light and update radiance if the ray exited out the front.
                        bool ray_exited_frontside = ray.direction.z >= 0.0f;
                        if (ray_exited_frontside) {
                            float l = is_in_shadow(medium_exit_position.x) ? 0.0f : 1.0f;
                            radiance[c] += throughput * l;
                        }
                    };

                    scatter_in_media(ray, sigma_t[c], single_scattering_albedo[c], rng, distance_to_medium_boundary, ray_exiting_media);
                }
            }

            radiance /= path_count;
            output.set_pixel(RGBA(radiance), Vector2ui(x, y));
        }
    }
}

inline void burley_integrator(Image output, float slab_thickness) {
    using namespace optix;
    using namespace OptiXRenderer::Shading::BSDFs;

    BurleySSS::Parameters bssrdf_params[bssrdf_count];
    for (int bssrdf_index = 0; bssrdf_index < bssrdf_count; ++bssrdf_index) {
        RGB albedo = sss_params[bssrdf_index].diffuse_albedo;
        RGB mfp = sss_params[bssrdf_index].mean_free_path;
        bssrdf_params[bssrdf_index] = BurleySSS::Parameters::create({ albedo.r, albedo.g, albedo.b }, { mfp.r, mfp.g, mfp.b }, BurleySSS::Parameters::LightConfig::Search);
    }

    const int width = output.get_width();
    const int sample_count = 4096;
    RNG::PmjbRNG rng(sample_count);

    for (int y = 0; y < bssrdf_count; y++) {
        BurleySSS::Parameters sss_params = bssrdf_params[y];
        #pragma omp parallel for schedule(dynamic, 5)
        for (int x = 0; x < width; x++) {
            float3 radiance = { 0.0f, 0.0f, 0.0f };
            float3 energy_loss = { 0.0f, 0.0f, 0.0f };
            for (int i = 0; i < sample_count; i++) {
                Vector3f rng_sample = rng.sample3f(i, sample_count);

                float3 po = make_float3(float(x), float(y), 0) * pixel_scale;

                auto sss_sample = BurleySSS::AlbedoMIS::sample(sss_params, po, { rng_sample.x, rng_sample.y, rng_sample.z });
                bool in_shadow = is_in_shadow(sss_sample.position.x);
                if (!in_shadow)
                    radiance += sss_sample.reflectance / sss_sample.PDF.value();

                // Approximate energy lost by frontside light scattering out the backside.
                float3 backside_position = sss_sample.position;
                backside_position.z -= slab_thickness;
                auto sss_sample_energy_loss = BurleySSS::evaluate(sss_params, po, backside_position);
                if (!in_shadow)
                    energy_loss += sss_sample_energy_loss / sss_sample.PDF.value();
            }

            radiance = radiance / sample_count - energy_loss / sample_count;
            RGBA pixel = { radiance.x, radiance.y, radiance.z, 1.0f };

            output.set_pixel(pixel, Vector2ui(x, y));
        }
    }
}

inline void integrate(float slab_thickness) {

    const int width = 100;
    const int half_width = width / 2;

    Image random_walk_image = Image::create2D("RandomWalkSSS", PixelFormat::RGB_Float, false, Vector2ui(half_width, bssrdf_count));
    Image burley_image = Image::create2D("BurleySSS", PixelFormat::RGB_Float, false, Vector2ui(half_width, bssrdf_count));

    random_walk_integrate(random_walk_image, slab_thickness);
    burley_integrator(burley_image, slab_thickness);

    // Expand image to one with 20 pixel rows per SSS material.
    int bssrdf_height = 20;
    int height = bssrdf_count * bssrdf_height;
    Image output_image = Image::create2D("", PixelFormat::RGB24, true, Vector2ui(width, height));
    for (int y = 0; y < height; ++y) {
        int bssrdf_y = y / bssrdf_height;
        for (int x = 0; x < half_width; ++x) {
            RGBA rw_radiance = random_walk_image.get_pixel(Vector2ui(x, bssrdf_y));
            output_image.set_pixel(rw_radiance, Vector2ui(half_width - x - 1, y));

            RGBA b_radiance = burley_image.get_pixel(Vector2ui(x, bssrdf_y));
            output_image.set_pixel(b_radiance, Vector2ui(half_width + x, y));
        }
    }

    // Output image
    std::string image_path = "C:/Temp/SSS_integration_slab_thickness_" + std::to_string(slab_thickness) + ".png";
    StbImageWriter::write(output_image, image_path);
    printf("Output image to '%s'\n", image_path.c_str());
}

} // NS Integrate

#endif // _SUBSURFACE_SCATTER_TEST_BED_SLAB_INTEGRATION_H_