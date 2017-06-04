// Cogwheel infinite area light.
// ---------------------------------------------------------------------------
// Copyright (C) 2015-2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#include <Cogwheel/Assets/InfiniteAreaLight.h>

#include <Cogwheel/Math/Distributions.h>
#include <Cogwheel/Math/Quaternion.h>
#include <Cogwheel/Math/RNG.h>

namespace Cogwheel {
namespace Assets {

// ------------------------------------------------------------------------------------------------
// Samplable, textured infinite area light.
// ------------------------------------------------------------------------------------------------
inline void InfiniteAreaLight::compute_PDF(TextureND latlong, float* PDF_result) {
    Image image = latlong.get_image();
    int width = image.get_width(), height = image.get_height();

    bool filter_pixels = latlong.get_magnification_filter() == Assets::MagnificationFilter::Linear;

    // Use a temporary PDF array if the PDFs should be filtered afterwards, otherwise use the result array.
    float* PDF = filter_pixels ? new float[width * height] : PDF_result;

    #pragma omp parallel for schedule(dynamic, 16)
    for (int y = 0; y < height; ++y) {
        // PBRT p. 728. Account for the non-uniform surface area of the pixels, e.g. the higher density near the poles.
        float sin_theta = sinf(Math::PI<float>() * (y + 0.5f) / float(height));

        float* PDF_row = PDF + y * width;
        for (int x = 0; x < width; ++x) {
            Math::RGB pixel = image.get_pixel(Math::Vector2ui(x, y)).rgb();
            PDF_row[x] = (pixel.r + pixel.g + pixel.b) * sin_theta;
        }
    }

    // If the texture is unfiltered, then the per pixel importance corresponds to the PDF.
    // If filtering is enabled, then we need to filter the PDF as well.
    // Generally this doesn't change much in terms of convergence, but it helps us to 
    // avoid artefacts in cases where a black pixel would have a PDF of 0,
    // but due to filtering the entire texel wouldn't actually be black.
    if (filter_pixels) {
        #pragma omp parallel for schedule(dynamic, 16)
        for (int y = 0; y < height; ++y) {
            // Blur per pixel importance to account for linear interpolation.
            // The pixel's own contribution is 20 / 32.
            // Neighbours on the side contribute by 2 / 32.
            // Neighbours in the corners contribute by 1 / 32.
            // Weights have been estimated based on linear interpolation.
            for (int x = 0; x < width; ++x) {
                float& pixel_PDF = PDF_result[x + y * width];
                pixel_PDF = 0.0f;

                { // Add contribution from left column.
                    int left_x = x - 1 < 0 ? (width - 1) : (x - 1); // Repeat mode.

                    int lower_left_index = left_x + std::max(0, y - 1) * width;
                    pixel_PDF += float(PDF[lower_left_index]);

                    int middle_left_index = left_x + y * width;
                    pixel_PDF += float(PDF[middle_left_index]) * 2.0f;

                    int upper_left_index = left_x + std::min(int(height) - 1, y + 1) * width;
                    pixel_PDF += float(PDF[upper_left_index]);
                }

                { // Add contribution from right column.
                    int right_x = x + 1 == width ? 0 : (x + 1); // Repeat mode.

                    int lower_right_index = right_x + std::max(0, y - 1) * width;
                    pixel_PDF += float(PDF[lower_right_index]);

                    int middle_right_index = right_x + y * width;
                    pixel_PDF += float(PDF[middle_right_index]) * 2.0f;

                    int upper_right_index = right_x + std::min(int(height) - 1, y + 1) * width;
                    pixel_PDF += float(PDF[upper_right_index]);
                }

                { // Add contribution from middle column.
                    int lower_middle_index = x + std::max(0, y - 1) * width;
                    pixel_PDF += float(PDF[lower_middle_index]) * 2.0f;

                    int upper_middle_index = x + std::min(int(height) - 1, y + 1) * width;
                    pixel_PDF += float(PDF[upper_middle_index]) * 2.0f;

                    // Center last as it has the highest weight.
                    pixel_PDF += float(PDF[x + y * width]) * 20.0f;
                }

                // Normalize.
                pixel_PDF /= 32.0f;
            }
        }

        delete[] PDF;
    }
}

// ------------------------------------------------------------------------------------------------
// Infinite area light utilities.
// ------------------------------------------------------------------------------------------------

namespace InfiniteAreaLightUtils {

template <typename T, typename F>
inline void Convolute(const InfiniteAreaLight& light, IBLConvolution<T>* begin, IBLConvolution<T>* end, F color_conversion) {

    using namespace Cogwheel::Math;
    using namespace Cogwheel::Math::Distributions;

    int max_sample_count = 0;
    for (IBLConvolution<T>* itr = begin; itr != end; ++itr)
        max_sample_count = max(max_sample_count, itr->sample_count);

    // Precompute light samples.
    std::vector<LightSample> light_samples = std::vector<LightSample>();
    light_samples.resize(max_sample_count * 4);
    #pragma omp parallel for schedule(dynamic, 16)
    for (int s = 0; s < light_samples.size(); ++s)
        light_samples[s] = light.sample(RNG::sample02(s));

    for (; begin != end; ++begin) {

        int width = begin->Width, height = begin->Height;
        float roughness = begin->Roughness;
        float alpha = roughness * roughness;

        // Handle nearly specular case.
        if (alpha < 0.00000000001f) {
            Textures::UID env_map_ID = light.get_texture_ID();
            #pragma omp parallel for schedule(dynamic, 16)
            for (int i = 0; i < width * height; ++i) {
                int x = i % width, y = i / width;
                begin->Pixels[x + y * width] = color_conversion(sample2D(env_map_ID, Vector2f((x + 0.5f) / width, (y + 0.5f) / height)).rgb());
            }
            continue;
        }

        std::vector<GGX::Sample> ggx_samples = std::vector<GGX::Sample>();
        ggx_samples.resize(begin->sample_count * 4);
        #pragma omp parallel for schedule(dynamic, 16)
        for (int s = 0; s < ggx_samples.size(); ++s)
            ggx_samples[s] = GGX::sample(alpha, RNG::sample02(s));

        #pragma omp parallel for schedule(dynamic, 16)
        for (int i = 0; i < width * height; ++i) {

            int x = i % width;
            int y = i / width;

            Vector2f up_uv = Vector2f((x + 0.5f) / width, (y + 0.5f) / height);
            Vector3f up_vector = latlong_texcoord_to_direction(up_uv);
            Quaternionf up_rotation = Quaternionf::look_in(up_vector);

            RGB radiance = RGB::black();

            int light_sample_count = begin->sample_count / 2;
            for (int s = 0; s < begin->sample_count / 2; ++s) {
                const LightSample& sample = light_samples[(s + RNG::hash(i)) % light_samples.size()];
                if (sample.PDF < 0.000000001f)
                    continue;

                Vector3f local_direction = normalize(inverse_unit(up_rotation) * sample.direction_to_light);
                float ggx_f = GGX::D(alpha, local_direction.z);
                if (isnan(ggx_f))
                    continue;

                float cos_theta = fmaxf(local_direction.z, 0.0f);
                float mis_weight = RNG::power_heuristic(sample.PDF, GGX::PDF(alpha, local_direction.z));
                radiance += sample.radiance * (mis_weight * ggx_f * cos_theta / sample.PDF);
            }

            int bsdf_sample_count = begin->sample_count - light_sample_count;
            for (int s = 0; s < bsdf_sample_count; ++s) {
                GGX::Sample sample = ggx_samples[(s + RNG::hash(i + 1013904223)) % ggx_samples.size()];
                if (sample.PDF < 0.000000001f)
                    continue;

                sample.direction = normalize(up_rotation * sample.direction);
                float mis_weight = RNG::power_heuristic(sample.PDF, light.PDF(sample.direction));
                radiance += light.evaluate(sample.direction) * mis_weight;
            }

            // Account for the samples being split evenly between BSDF and light.
            radiance *= 2.0f;
            begin->Pixels[x + y * width] = color_conversion(radiance / float(begin->sample_count));
        }
    }
}

} // NS InfiniteAreaLightUtils
} // NS Assets
} // NS Cogwheel