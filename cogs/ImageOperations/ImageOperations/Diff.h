// Image differentiation operations.
// ------------------------------------------------------------------------------------------------
// Copyright (C) 2015-2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. 
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _IMAGE_OPERATIONS_DIFF_H_
#define _IMAGE_OPERATIONS_DIFF_H_

#include <Cogwheel/Assets/Image.h>

namespace ImageOperations {
namespace Diff {

using namespace Cogwheel::Assets;
using namespace Cogwheel::Math;

// ------------------------------------------------------------------------------------------------
// Mean (taxicab / L1) image diff.
// ------------------------------------------------------------------------------------------------
float mean(Image reference, Image target, Image diff = Image()) {

    assert(reference.get_width() > 0u && reference.get_height() > 0u);
    assert(reference.get_width() == target.get_width() && reference.get_height() == target.get_height());

    double mean = 0.0f;
    for (unsigned int y = 0; y < reference.get_height(); ++y) {
        for (unsigned int x = 0; x < reference.get_width(); ++x) {
            RGB a = reference.get_pixel(Vector2ui(x, y)).rgb();
            RGB b = target.get_pixel(Vector2ui(x, y)).rgb();
            RGB d = RGB(abs(a.r - b.r), abs(a.g - b.g), abs(a.b - b.b));
            float l1 = luma(d);
            mean += l1;
            if (diff.exists())
                diff.set_pixel(RGBA(d), Vector2ui(x, y));
        }
    }

    return float(mean / reference.get_pixel_count());
}

// ------------------------------------------------------------------------------------------------
// Root mean square image diff.
// ------------------------------------------------------------------------------------------------
float rms(Image reference, Image target, Image diff = Image()) {

    assert(reference.get_width() > 0u && reference.get_height() > 0u);
    assert(reference.get_width() == target.get_width() && reference.get_height() == target.get_height());

    double mean = 0.0f;
    for (unsigned int y = 0; y < reference.get_height(); ++y) {
        for (unsigned int x = 0; x < reference.get_width(); ++x) {
            RGB a = reference.get_pixel(Vector2ui(x, y)).rgb();
            RGB b = target.get_pixel(Vector2ui(x, y)).rgb();
            RGB error = RGB(abs(a.r - b.r), abs(a.g - b.g), abs(a.b - b.b));
            float l1 = luma(error);
            mean += l1 * l1;
            if (diff.exists())
                diff.set_pixel(RGBA(gammacorrect(error, 2.0f)), Vector2ui(x, y));
        }
    }

    return float(mean / (reference.get_pixel_count() * 3.0));
}

// ------------------------------------------------------------------------------------------------
// Structural similarity index (SSIM).
// http://www.cns.nyu.edu/pub/lcv/wang03-reprint.pdf
// ------------------------------------------------------------------------------------------------
float ssim(Image reference_image, Image target_image) {

    assert(reference_image.get_width() > 0u && reference_image.get_height() > 0u);
    assert(reference_image.get_width() == target_image.get_width() && reference_image.get_height() == target_image.get_height());

    unsigned int width = reference_image.get_width(), height = reference_image.get_height();

    auto RGB_to_vector3d = [](RGB rgb) -> Vector3d { return Vector3d(rgb.r, rgb.g, rgb.b); };

    // Compute the mean and store all the pixel values in floats for faster lookup. Algorithm (2)
    RGB* reference = new RGB[width * height];
    RGB* target = new RGB[width * height];
    Vector3d reference_mean = {};
    Vector3d target_mean = {};
    for (unsigned int y = 0; y < height; ++y) {
        for (unsigned int x = 0; x < width; ++x) {
            *reference = reference_image.get_pixel(Vector2ui(x, y)).rgb();
            *target = target_image.get_pixel(Vector2ui(x, y)).rgb();

            reference_mean += RGB_to_vector3d(*reference);
            target_mean += RGB_to_vector3d(*target);

            ++reference;
            ++target;
        }
    }
    reference_mean /= width * height;
    target_mean /= width * height;

    // Reset the RGB arrays.
    reference -= width * height;
    target -= width * height;

    // Compute the standard deviation and covariance of the image. Algorithm (4) and (11).
    Vector3d reference_std_dev = {};
    Vector3d target_std_dev = {};
    Vector3d covariance = {};
    for (unsigned int y = 0; y < height; ++y)
        for (unsigned int x = 0; x < width; ++x) {
            Vector3d ref_d = RGB_to_vector3d(reference[x + y * width]) - reference_mean;
            reference_std_dev += { ref_d.x * ref_d.x, ref_d.y * ref_d.y, ref_d.z * ref_d.z };

            Vector3d tar_d = RGB_to_vector3d(target[x + y * width]) - target_mean;
            target_std_dev += { tar_d.x * tar_d.x, tar_d.y * tar_d.y, tar_d.z * tar_d.z };

            covariance += {ref_d.x * tar_d.x, ref_d.y * tar_d.y, ref_d.z * tar_d.z };
        }
    reference_std_dev /= width * height;
    reference_std_dev = { sqrt(reference_std_dev.x), sqrt(reference_std_dev.y), sqrt(reference_std_dev.z) };

    target_std_dev /= width * height;
    target_std_dev = { sqrt(target_std_dev.x), sqrt(target_std_dev.y), sqrt(target_std_dev.z) };

    covariance /= width * height;

    delete[] reference;
    delete[] target;

    // Compute SSIM, algorithm (13)
    double C1 = 0.000001, C2 = 0.0000001;

    Vector3d ssim = (2 * reference_mean * target_mean + C1) * (2 * covariance + C2) /
        ((reference_mean * reference_mean + target_mean * target_mean + C1) * (reference_std_dev * reference_std_dev + target_std_dev * target_std_dev + C2));
    RGB ssim_rgb = { float(ssim.x), float(ssim.y), float(ssim.z) };

    return luma(ssim_rgb);
}

// ------------------------------------------------------------------------------------------------
// Structural similarity index (SSIM).
// http://www.cns.nyu.edu/pub/lcv/wang03-reprint.pdf
// ------------------------------------------------------------------------------------------------
float ssim(Image reference_image, Image target_image, int bandwidth, Image diff_image = Image()) {

    assert(reference_image.get_width() > 0u && reference_image.get_height() > 0u);
    assert(reference_image.get_width() == target_image.get_width() && reference_image.get_height() == target_image.get_height());
    assert(reference_image.get_width() == diff_image.get_width() && reference_image.get_height() == diff_image.get_height());

    unsigned int width = reference_image.get_width(), height = reference_image.get_height();

    // Store all the pixel values in floats for faster lookup.
    RGB* reference = new RGB[width * height];
    RGB* target = new RGB[width * height];
    for (unsigned int y = 0; y < height; ++y) {
        for (unsigned int x = 0; x < width; ++x) {
            reference[x + y * width] = reference_image.get_pixel(Vector2ui(x, y)).rgb();
            target[x + y * width] = target_image.get_pixel(Vector2ui(x, y)).rgb();
        }
    }

    // Loop over all pixels and compute their SSIM values inside the kernel bandwidth.
    double ssim = 0.0;
    // #pragma omp parallel for // TODO But requires atomic ssim updates.
    for (int i = 0; i < int(width * height); ++i) {
        int xx = i % width, yy = i / width;

        // Kernel ranges.
        int y_start = max(yy - bandwidth, 0);
        int y_end = min(yy + bandwidth, int(height));
        int x_start = max(xx - bandwidth, 0);
        int x_end = min(xx + bandwidth, int(width));
        int pixel_count = (x_end - x_start) * (y_end - y_start);

        // Compute the mean. Algorithm (2)
        RGB reference_mean = RGB::black();
        RGB target_mean = RGB::black();
        for (int y = y_start; y < y_end; ++y)
            for (int x = x_start; x < x_end; ++x) {
                reference_mean += reference[x + y * width];
                target_mean += target[x + y * width];
            }
        reference_mean /= float(pixel_count);
        target_mean /= float(pixel_count);

        // Compute the standard deviation and covariance of the image. Algorithm (4) and (11).
        RGB reference_std_dev = {};
        RGB target_std_dev = {};
        RGB covariance = {};
        for (int y = y_start; y < y_end; ++y)
            for (int x = x_start; x < x_end; ++x) {
                RGB ref_d = reference[x + y * width] - reference_mean;
                reference_std_dev += { ref_d.r * ref_d.r, ref_d.g * ref_d.g, ref_d.b * ref_d.b };

                RGB tar_d = target[x + y * width] - target_mean;
                target_std_dev += { tar_d.r * tar_d.r, tar_d.g * tar_d.g, tar_d.b * tar_d.b };

                covariance += {ref_d.r * tar_d.r, ref_d.g * tar_d.g, ref_d.b * tar_d.b };
            }
        reference_std_dev /= float(pixel_count);
        reference_std_dev = { sqrt(reference_std_dev.r), sqrt(reference_std_dev.g), sqrt(reference_std_dev.b) };

        target_std_dev /= float(pixel_count);
        target_std_dev = { sqrt(target_std_dev.r), sqrt(target_std_dev.g), sqrt(target_std_dev.b) };

        covariance /= float(pixel_count);

        // Compute SSIM, algorithm (13)
        float C1 = 0.000001f, C2 = 0.0000001f;

        RGB ssim_i = (2 * reference_mean * target_mean + C1) * (2 * covariance + C2) /
            ((reference_mean * reference_mean + target_mean * target_mean + C1) * (reference_std_dev * reference_std_dev + target_std_dev * target_std_dev + C2));
        
        ssim += luma(ssim_i);

        if (diff_image.exists())
            diff_image.set_pixel(RGBA(1.0f - ssim_i.r, 1.0f - ssim_i.g, 1.0f - ssim_i.b, 1.0f), Vector2ui(xx, yy));
    }
    ssim /= width * height;

    delete[] reference;
    delete[] target;
    
    return float(ssim);
}

} // NS Diff
} // NS ImageOperations

#endif // _IMAGE_OPERATIONS_DIFF_H_