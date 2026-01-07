// Subsurface scattering testbed
// ------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#include <Bifrost/Assets/Image.h>
#include <Bifrost/Math/Distributions.h>
#include <Bifrost/Math/Intersect.h>
#include <Bifrost/Math/Plane.h>
#include <Bifrost/Math/Ray.h>
#include <Bifrost/Math/RNG.h>
#include <Bifrost/Math/Utils.h>

#include <OptiXRenderer/RNG.h>
#include <OptiXRenderer/Shading/BSDFs/BurleySSS.h>

#include <StbImageWriter/StbImageWriter.h>

using namespace Bifrost::Assets;
using namespace Bifrost::Math;

enum ScatteringMethod { RandomWalk, Burley };

struct MeasuredScatteringParameters {
    RGB scattering_coefficient;
    RGB absorption_coefficient;
    RGB diffuse_albedo; // TODO Compute from scattering and absorption. See A Practical Model for Subsurface Light Transport, Jensen et al., 2001

    // Get the attenuation coefficient, or extinction coefficient, which is the sum of scattering and absorption along the path.
    // PBRT v4, section 11.1.3
    // https://pbr-book.org/4ed/Volume_Scattering/Volume_Scattering_Processes#OutScatteringandAttenuation
    RGB get_attenuation_coefficient() const { return scattering_coefficient + absorption_coefficient; }

    // The mean free path is the reciprocal of the attenuation coefficient.
    // PBRT v4, section 11.1.3
    // https://pbr-book.org/4ed/Volume_Scattering/Volume_Scattering_Processes#OutScatteringandAttenuation
    RGB get_mean_free_path() const {
        RGB sigma_t = get_attenuation_coefficient();
        return RGB(1.0f / sigma_t.r, 1.0f / sigma_t.g, 1.0f / sigma_t.b);
    }

    RGB get_single_scattering_albedo() const { return scattering_coefficient / (scattering_coefficient + absorption_coefficient); }
};

struct Options {
    ScatteringMethod scattering_method = ScatteringMethod::RandomWalk;

    static Options parse(int argc, char** argv) {
        Options options;

        int argument = 1;
        while (argument < argc) {
            if (strcmp(argv[argument], "--random_walk") == 0 ||
                strcmp(argv[argument], "--random-walk") == 0 ||
                strcmp(argv[argument], "--randomwalk") == 0)
                options.scattering_method = ScatteringMethod::RandomWalk;
            else if (strcmp(argv[argument], "--burley") == 0)
                options.scattering_method = ScatteringMethod::Burley;

            ++argument;
        }

        return options;
    }

    static void print_usage() {
        printf("Subsurface scattering testbed usage:\n"
            "  --random_walk: Compute SSS using random walk (default).\n"
            "  --burley: Use Burley's SSS approximation.\n");
    }
};

int main(int argc, char** argv) {
    std::string command = argc >= 2 ? std::string(argv[1]) : "";
    if (command.compare("-h") == 0 || command.compare("--help") == 0) {
        Options::print_usage();
        return 0;
    }

    Options options = Options::parse(argc, argv);

    printf("Subsurface scattering testbed with scattering method: %s\n",
        options.scattering_method == ScatteringMethod::RandomWalk ? "Random Walk" : "Burley");

    Images::allocate(1u);

    // Parameters from A Practical Model for Subsurface Light Transport, Jensen et al., 2001
    const int bssrdf_count = 5;
    MeasuredScatteringParameters ketchup = { RGB(0.18f, 0.07f, 0.03f), RGB(0.061f, 0.97f, 1.45f), RGB(0.164f, 0.006f, 0.002f) };
    MeasuredScatteringParameters marble = { RGB(2.19f, 2.62f, 3.00f), RGB(0.0021f, 0.0041f, 0.0071f), RGB(0.830f, 0.791f, 0.753f) };
    MeasuredScatteringParameters potato = { RGB(0.68f, 0.70f, 0.55f), RGB(0.0024f, 0.0090f, 0.12f), RGB(0.764f, 0.613f, 0.213f) };
    MeasuredScatteringParameters skin1 = { RGB(0.74f, 0.88f, 1.01f), RGB(0.032f, 0.17f, 0.48f), RGB(0.436f, 0.227f, 0.131f) };
    MeasuredScatteringParameters whole_milk = { RGB(2.55f, 3.21f, 3.77f), RGB(0.0011f, 0.0024f, 0.014f), RGB(0.908f, 0.881f, 0.759f) };
    MeasuredScatteringParameters sss_params[bssrdf_count] = { potato, marble, whole_milk, skin1, ketchup };

    int width = 100, height = 100;
    int half_width = width / 2;
    Image image = Image::create2D("", PixelFormat::RGB24, true, Vector2ui(width, height));

    // Each pixel represents one mm in world space.
    // A 10 pixel wide column in the center of the image illuminates the plane consisting of the different materials.
    auto is_in_shadow = [=](float position_x) -> bool {
        return abs(position_x - half_width) > 5; // Pixels more than 5mm away from the center row are in shadow.
    };

    if (options.scattering_method == ScatteringMethod::RandomWalk) {
        image.set_name("RandomWalkSSS");

        Plane surface = Plane::from_point_normal(Vector3f::zero(), Vector3f(0, 0, 1));

        const int path_count = 4048;
        const int max_bounce_count = 250;

        #pragma omp parallel for schedule(dynamic, 2)
        for (int y = 0; y < height; y++) {
            // Scattering parameters
            int bssrdf_index = int(y / (height / float(bssrdf_count)));
            RGB sigma_t = sss_params[bssrdf_index].get_attenuation_coefficient();
            RGB single_scattering_albedo = sss_params[bssrdf_index].get_single_scattering_albedo();
            
            for (int x = 0; x < width; x++) {

                RGB radiance = { 0.0f, 0.0f, 0.0f };
                for (int c = 0; c < 3; ++c) {
                    for (int i = 0; i < path_count; i++) {
                        RNG::LinearCongruential rng = RNG::LinearCongruential(RNG::teschner_hash(x, y) ^ Bifrost::Math::reverse_bits(unsigned int(i)));

                        // Ray starting right past the surface at x,y and looking into the medium along negative z.
                        Ray ray = Ray(Vector3f(float(x), float(y), -1e-6f), Vector3f(0, 0, -1));

                        // Volumetric integration based on Ray Tracing Inhomogeneous Volumes, Ray Tracing Gems 1, chapter 28
                        float throughput = 1.0f;
                        for (int bounce_count = 0; throughput > 0.0001f; ++bounce_count) {
                            // Terminate the ray when max number of bounces is reached.
                            if (bounce_count == max_bounce_count) {
                                throughput = 0.0f;
                                continue;
                            }

                            // Sample distance to next scattering event
                            auto scattering_distance = Distributions::Exponential::sample_distance(sigma_t[c], rng.sample1f());

                            // Intersect with the scene
                            float distance_to_surface = intersect(ray, surface);
                            bool surface_behind_ray = distance_to_surface < 0.0f;
                            bool exits_medium = !surface_behind_ray && distance_to_surface < scattering_distance;

                            // Check if the ray leaves the medium and apply light if that's the case.
                            if (exits_medium) {
                                // Surface intersection before scattering event
                                Vector3f medium_exit_position = ray.position_at(distance_to_surface);

                                // Apply light and update radiance.
                                float l = is_in_shadow(medium_exit_position.x) ? 0.0f : 1.0f;
                                radiance[c] += throughput * l;

                                // Terminate the ray by setting the ray's throughput to zero.
                                throughput = 0.0f;
                            } else {
                                // Sample scattering direction from isotropic (spherical) distribution
                                auto direction_sample = Distributions::Sphere::sample(rng.sample2f());
                                // auto direction_sample_PDF = Distributions::Sphere::PDF();

                                // Adjust throughput by phase function P and sampling probabilities.
                                // They cancel out as long as we trace one color channel at a time.
                                // throughput *= direction_sample_PDF / direction_sample_PDF;

                                throughput *= single_scattering_albedo[c];

                                // Russian roulette based on ray throughput.
                                if (rng.sample1f() < throughput)
                                    throughput = 1.0f; // Should be throughput / RR_decision_probability, but that simplifies to throughput / throughput in the one channel case.
                                else
                                    throughput = 0.0f;

                                // Create new ray from scattering event.
                                Vector3f scattering_position = ray.position_at(scattering_distance);
                                ray = Ray(scattering_position, direction_sample);
                            }
                        }
                    }
                }

                radiance /= path_count;
                image.set_pixel(RGBA(radiance), Vector2ui(x, y));
            }
        }

    } else if (options.scattering_method == ScatteringMethod::Burley) {

        using namespace optix;
        using namespace OptiXRenderer;
        using namespace OptiXRenderer::Shading::BSDFs;

        image.set_name("BurleySSS");

        // Scale to make Burley's approximation fit the volumetric path tracer.
        // This most likely means that there is a bug somewhere, since this shouldn't be needed, but we'll fix it later.
        const float mean_free_path_fitting_scale = 1.5f;
        BurleySSS::Parameters bssrdf_params[bssrdf_count];
        for (int bssrdf_index = 0; bssrdf_index < bssrdf_count; ++bssrdf_index) {
            RGB albedo = sss_params[bssrdf_index].diffuse_albedo;
            RGB mfp = sss_params[bssrdf_index].get_mean_free_path() * mean_free_path_fitting_scale;
            bssrdf_params[bssrdf_index] = BurleySSS::Parameters::create({ albedo.r, albedo.g, albedo.b }, { mfp.r, mfp.g, mfp.b } );
        }

        const int sample_count = 4092;

        #pragma omp parallel for schedule(dynamic, 2)
        for (int y = 0; y < height; y++) {
            int bssrdf_index = int(y / (height / float(bssrdf_count)));
            BurleySSS::Parameters sss_params = bssrdf_params[bssrdf_index];
            for (int x = 0; x < width; x++) {
                int index_offset = OptiXRenderer::RNG::teschner_hash(x, y);

                float3 radiance = { 0.0f, 0.0f, 0.0f };
                for (int i = 0; i < sample_count; i++) {
                    auto rng = OptiXRenderer::RNG::PracticalScrambledSobol(index_offset + i, 0);
                    float3 rng_sample = make_float3(rng.sample4f());

                    auto sss_sample = BurleySSS::AlbedoMIS::sample(sss_params, make_float3(float(x), float(y), 0), rng_sample);
                    bool in_shadow = is_in_shadow(sss_sample.position.x);
                    if (!in_shadow)
                        radiance += sss_sample.reflectance / sss_sample.PDF.value();
                }

                radiance /= sample_count;
                RGBA pixel = { radiance.x, radiance.y, radiance.z, 1.0f };

                image.set_pixel(pixel, Vector2ui(x, y));
            }
        }
    }

    std::string image_path = "C:/Temp/" + image.get_name() + ".png";
    StbImageWriter::write(image, image_path);
    printf("Output image to '%s'\n", image_path.c_str());

    return 0;
}