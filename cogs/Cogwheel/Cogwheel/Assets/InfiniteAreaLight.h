// Cogwheel infinite area light.
// ---------------------------------------------------------------------------
// Copyright (C) 2015-2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _COGWHEEL_ASSETS_INFINITE_AREA_LIGHT_H_
#define _COGWHEEL_ASSETS_INFINITE_AREA_LIGHT_H_

#include <Cogwheel/Assets/Texture.h>
#include <Cogwheel/Math/Distribution2D.h>

namespace Cogwheel {
namespace Assets {

// ------------------------------------------------------------------------------------------------
// A single sample from a light source.
// ------------------------------------------------------------------------------------------------
struct LightSample {
    Math::RGB radiance;
    float PDF;
    Math::Vector3f direction_to_light;
    float distance;
};

// ------------------------------------------------------------------------------------------------
// Samplable, textured infinite area light.
// ------------------------------------------------------------------------------------------------
class InfiniteAreaLight {
private:
    mutable TextureND m_latlong;
    Math::Distribution2D<double> m_distribution;

public:

    //*********************************************************************************************
    // Constructor.
    //*********************************************************************************************
    InfiniteAreaLight(Textures::UID latlong_ID)
        : m_latlong(latlong_ID)
        , m_distribution(compute_PDF(m_latlong), m_latlong.get_image().get_width(), m_latlong.get_image().get_height()) {
    }

    //*********************************************************************************************
    // Evaluate.
    //*********************************************************************************************

    Math::RGB evaluate(Math::Vector2f uv) const {
        return sample2D(m_latlong.get_ID(), uv).rgb();
    }

    Math::RGB evaluate(Math::Vector3f direction_to_light) const {
        Math::Vector2f uv = Math::direction_to_latlong_texcoord(direction_to_light);
        return evaluate(uv);
    }

    //*********************************************************************************************
    // Sampling.
    //*********************************************************************************************

    LightSample sample(Math::Vector2f random_sample) const {
        auto CDF_sample = m_distribution.sample_continuous(random_sample);

        LightSample sample;
        sample.direction_to_light = Math::latlong_texcoord_to_direction(CDF_sample.index);
        sample.distance = 1e30f;
        sample.radiance = sample2D(m_latlong.get_ID(), CDF_sample.index).rgb();
        float sin_theta = abs(sqrtf(1.0f - sample.direction_to_light.y * sample.direction_to_light.y));
        float PDF = float(CDF_sample.PDF) / (2.0f * Math::PI<float>() * Math::PI<float>() * sin_theta);
        sample.PDF = sin_theta == 0.0f ? 0.0f : PDF;
        return sample;
    }

    float PDF(Math::Vector3f direction_to_light) const {
        float sin_theta = abs(sqrtf(1.0f - direction_to_light.y * direction_to_light.y));
        Math::Vector2f uv = Math::direction_to_latlong_texcoord(direction_to_light);
        float distribution_PDF = float(m_distribution.PDF_continuous(uv));
        float PDF = distribution_PDF / (2.0f * Math::PI<float>() * Math::PI<float>() * sin_theta);
        return sin_theta == 0.0f ? 0.0f : PDF;
    }

    //*********************************************************************************************
    // Static utility functions.
    //*********************************************************************************************
    static float* compute_PDF(TextureND latlong) {
        float* PDF = new float[latlong.get_image().get_pixel_count()];
        compute_PDF(latlong, PDF);
        return PDF;
    }

    // Computes the array of per pixel PDFs for an infinite area light. 
    // The importance is based on the average radiance of a pixel.
    // The PDF input must contain room for at least pixelcount floats.
    static void compute_PDF(TextureND latlong, float* PDF_result) {
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
};

} // NS Assets
} // NS Cogwheel

#endif // _COGWHEEL_ASSETS_INFINITE_AREA_LIGHT_H_