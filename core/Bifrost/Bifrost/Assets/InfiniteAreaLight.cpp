// Bifrost infinite area light.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#include <Bifrost/Assets/InfiniteAreaLight.h>

#include <Bifrost/Math/Constants.h>
#include <Bifrost/Math/Distributions.h>

namespace Bifrost::Assets {

struct PdfResult {
    int width, height;
    float* PDF;
};

// Computes the array of per pixel PDFs for an infinite area light. 
// The importance is based on the average radiance of a pixel.
// The PDF input must contain room for at least as many elements as there are pixels.
inline PdfResult compute_PDF(Texture latlong) {
    Image image = latlong.get_image();

    PdfResult result;
    int width = result.width = image.get_width();
    int height = result.height = Math::max(image.get_height(), InfiniteAreaLight::MINIMUM_PDF_HEIGHT);
    result.PDF = new float[width * height];
    bool resample_height = height != image.get_height();

    bool filter_pixels = latlong.get_magnification_filter() == Assets::MagnificationFilter::Linear || resample_height;

    // Use a temporary PDF array if the PDFs should be filtered afterwards, otherwise use the result array.
    float* PDF = filter_pixels ? new float[width * height] : result.PDF;

    #pragma omp parallel for schedule(dynamic, 16)
    for (int y = 0; y < height; ++y) {
        // PBRT p. 728. Account for the non-uniform surface area of the pixels, i.e. the higher density near the poles.
        float sin_theta = sinf(Math::PI<float>() * (y + 0.5f) / float(height));

        float* PDF_row = PDF + y * width;
        if (resample_height) {
            // Sample the image for PDF values
            float v = (y + 0.5f) / height;
            for (int x = 0; x < width; ++x) {
                float u = (x + 0.5f) / width;
                Math::RGB pixel = sample2D(latlong, { u, v }).rgb();
                PDF_row[x] = (pixel.r + pixel.g + pixel.b) * sin_theta;
            }
        } else {
            // One to one correspondence between image and PDF size.
            for (int x = 0; x < width; ++x) {
                Math::RGB pixel = image.get_pixel(Math::Vector2ui(x, y)).rgb();
                PDF_row[x] = (pixel.r + pixel.g + pixel.b) * sin_theta;
            }
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
                float& pixel_PDF = result.PDF[x + y * width];
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

    return result;
}

inline Math::Distribution2D<float> compute_distribution(Texture latlong) {
    PdfResult PDF = compute_PDF(latlong);
    return Math::Distribution2D<float>(PDF.PDF, PDF.width, PDF.height);
}

InfiniteAreaLight::InfiniteAreaLight(Texture latlong)
    : m_latlong(latlong), m_distribution(compute_distribution(latlong)) { }

// ------------------------------------------------------------------------------------------------
// Infinite area light utilities.
// ------------------------------------------------------------------------------------------------

namespace InfiniteAreaLightUtils {

void reconstruct_solid_angle_PDF_sans_sin_theta(const InfiniteAreaLight& light, float* per_pixel_PDF) {
    int width = light.get_PDF_width(), height = light.get_PDF_height();

    float PDF_image_scaling = float(width * height);
    float PDF_normalization_term = 1.0f / (2.0f * Math::PI<float>() * Math::PI<float>());
    float PDF_scale = PDF_image_scaling * PDF_normalization_term;
    #pragma omp parallel for schedule(dynamic, 16)
    for (int y = 0; y < height; ++y) {
        float marginal_PDF = light.get_image_marginal_CDF()[y + 1] - light.get_image_marginal_CDF()[y];

        for (int x = 0; x < width; ++x) {
            const float* conditional_CDF_offset = light.get_image_conditional_CDF() + x + y * (width + 1);
            float conditional_PDF = conditional_CDF_offset[1] - conditional_CDF_offset[0];

            per_pixel_PDF[x + y * width] = marginal_PDF * conditional_PDF * PDF_scale;
        }
    }
}

} // NS InfiniteAreaLightUtils
} // NS Bifrost::Assets
