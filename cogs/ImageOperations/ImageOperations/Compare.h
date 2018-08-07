// Image comparison operations.
// ------------------------------------------------------------------------------------------------
// Copyright (C) 2015-2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. 
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _IMAGE_OPERATIONS_COMPARE_H_
#define _IMAGE_OPERATIONS_COMPARE_H_

#include <Cogwheel/Assets/Image.h>

namespace ImageOperations {
namespace Compare {

using namespace Cogwheel::Assets;
using namespace Cogwheel::Math;

// ------------------------------------------------------------------------------------------------
// Root mean square image diff.
// ------------------------------------------------------------------------------------------------
float rms(Image reference, Image target, Image diff = Image()) {

    assert(reference.get_width() > 0u && reference.get_height() > 0u);
    assert(reference.get_width() == target.get_width() && reference.get_height() == target.get_height());
    assert(!diff.exists() || reference.get_width() == diff.get_width() && reference.get_height() == diff.get_height());

    double mean_squared = 0.0f;
    for (unsigned int y = 0; y < reference.get_height(); ++y) {
        for (unsigned int x = 0; x < reference.get_width(); ++x) {
            RGB a = reference.get_pixel(Vector2ui(x, y)).rgb();
            RGB b = target.get_pixel(Vector2ui(x, y)).rgb();
            RGB error = RGB(abs(a.r - b.r), abs(a.g - b.g), abs(a.b - b.b));
            float l1 = luminance(error);
            mean_squared += l1 * l1;
            if (diff.exists())
                diff.set_pixel(RGBA(error), Vector2ui(x, y));
        }
    }

    return sqrt(float(mean_squared / reference.get_pixel_count()));
}

class Statistics {

    Vector3d summed_reference = {};
    Vector3d summed_reference_squared = {};

    Vector3d summed_target = {};
    Vector3d summed_target_squared = {};

    Vector3d summed_joint_expectation = {};

    double summed_weight = 0.0f;

public:

    Vector3d reference_mean() const { return summed_reference / summed_weight; }
    Vector3d reference_variance() const { return summed_reference_squared / summed_weight - reference_mean() *  reference_mean(); }

    Vector3d target_mean() const { return summed_target / summed_weight; }
    Vector3d target_variance() const { return summed_target_squared / summed_weight - target_mean() * target_mean(); }

    Vector3d covariance() const { return summed_joint_expectation / summed_weight - summed_reference * summed_target / (summed_weight * summed_weight); }

    void add(RGB reference_sample, RGB target_sample, double weight = 1.0) {
        auto RGB_to_vector3d = [](RGB rgb) -> Vector3d { return Vector3d(rgb.r, rgb.g, rgb.b); };

        Vector3d reference = RGB_to_vector3d(reference_sample);
        summed_reference += weight * reference;
        summed_reference_squared += weight * reference * reference;

        Vector3d target = RGB_to_vector3d(target_sample);
        summed_target += weight * target;
        summed_target_squared += weight * target * target;

        summed_joint_expectation += weight * reference * target;

        summed_weight += weight;
    }
};

// ------------------------------------------------------------------------------------------------
// Structural similarity index (SSIM).
// http://www.cns.nyu.edu/pub/lcv/wang03-reprint.pdf
// ------------------------------------------------------------------------------------------------
float ssim(Image reference_image, Image target_image) {

    assert(reference_image.get_width() > 0u && reference_image.get_height() > 0u);
    assert(reference_image.get_width() == target_image.get_width() && reference_image.get_height() == target_image.get_height());

    unsigned int width = reference_image.get_width(), height = reference_image.get_height();

    auto RGB_to_vector3d = [](RGB rgb) -> Vector3d { return Vector3d(rgb.r, rgb.g, rgb.b); };

    Statistics image_stats = {};
    for (unsigned int y = 0; y < height; ++y)
        for (unsigned int x = 0; x < width; ++x)
            image_stats.add(reference_image.get_pixel(Vector2ui(x, y)).rgb(),
                            target_image.get_pixel(Vector2ui(x, y)).rgb());

    Vector3d reference_mean = image_stats.reference_mean();
    Vector3d reference_variance = image_stats.reference_variance();
    Vector3d target_mean = image_stats.target_mean();
    Vector3d target_variance = image_stats.target_variance();
    Vector3d covariance = image_stats.covariance();

    // Compute SSIM, algorithm (13)
    double C1 = 0.01, C2 = 0.03;

    Vector3d ssim = (2.0 * reference_mean * target_mean + C1) * (2.0 * covariance + C2) /
        ((reference_mean * reference_mean + target_mean * target_mean + C1) * (reference_variance + target_variance+ C2));
    RGB ssim_rgb = { float(ssim.x), float(ssim.y), float(ssim.z) };

    return luminance(ssim_rgb);
}

// ------------------------------------------------------------------------------------------------
// Structural similarity index (SSIM).
// http://www.cns.nyu.edu/pub/lcv/wang03-reprint.pdf
// ------------------------------------------------------------------------------------------------
float mssim(Image reference_image, Image target_image, int support, Image diff_image = Image()) {

    assert(reference_image.get_width() > 0u && reference_image.get_height() > 0u);
    assert(reference_image.get_width() == target_image.get_width() && reference_image.get_height() == target_image.get_height());
    assert(!diff_image.exists() || reference_image.get_width() == diff_image.get_width() && reference_image.get_height() == diff_image.get_height());

    unsigned int width = reference_image.get_width(), height = reference_image.get_height();

    // Store all the pixel values in floats for faster lookup.
    RGB* reference = new RGB[width * height];
    RGB* target = new RGB[width * height];
    for (unsigned int y = 0; y < height; ++y)
        for (unsigned int x = 0; x < width; ++x) {
            reference[x + y * width] = reference_image.get_pixel(Vector2ui(x, y)).rgb();
            target[x + y * width] = target_image.get_pixel(Vector2ui(x, y)).rgb();
        }

    // Loop over all pixels and compute their SSIM values inside the kernel's support area.
    double mssim = 0.0;
    for (int i = 0; i < int(width * height); ++i) {
        int xx = i % width, yy = i / width;

        // Kernel ranges.
        int y_start = max(yy - support, 0);
        int y_end = min(yy + support, int(height));
        int x_start = max(xx - support, 0);
        int x_end = min(xx + support, int(width));

        Statistics image_stats = {};
        for (int y = y_start; y < y_end; ++y)
            for (int x = x_start; x < x_end; ++x) {
                float distance_squared = magnitude_squared(Vector2f(float(x - xx), float(y - yy)) / float(support));
                float weight_variance = 1.5f * 1.5f;
                float weight = exp(distance_squared / (2.0f * weight_variance)) / sqrtf(2.0f * PI<float>() * weight_variance);
                image_stats.add(reference[x + y * width], target[x + y * width], weight);
            }

        Vector3d reference_mean = image_stats.reference_mean();
        Vector3d reference_variance = image_stats.reference_variance();
        Vector3d target_mean = image_stats.target_mean();
        Vector3d target_variance = image_stats.target_variance();
        Vector3d covariance = image_stats.covariance();

        // Compute SSIM, algorithm (13)
        double C1 = 0.01, C2 = 0.03;

        Vector3d ssim = (2.0 * reference_mean * target_mean + C1) * (2.0 * covariance + C2) /
            ((reference_mean * reference_mean + target_mean * target_mean + C1) * (reference_variance + target_variance + C2));
        RGB ssim_rgb = { float(ssim.x), float(ssim.y), float(ssim.z) };

        mssim += luminance(ssim_rgb);

        if (diff_image.exists())
            diff_image.set_pixel(RGBA(1.0f - ssim_rgb.r, 1.0f - ssim_rgb.g, 1.0f - ssim_rgb.b, 1.0f), Vector2ui(xx, yy));
    }
    mssim /= width * height;

    delete[] reference;
    delete[] target;
    
    return float(mssim);
}

} // NS Compare
} // NS ImageOperations

#endif // _IMAGE_OPERATIONS_COMPARE_H_