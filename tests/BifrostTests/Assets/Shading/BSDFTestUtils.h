// Test utils for Bifrost's BSDFs.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _BIFROST_ASSETS_SHADING_BSDF_TEST_UTILS_H_
#define _BIFROST_ASSETS_SHADING_BSDF_TEST_UTILS_H_

#include <Expects.h>

#include <Bifrost/Assets/Shading/Constants.h>
#include <Bifrost/Assets/Shading/Fittings.h>
#include <Bifrost/Assets/Shading/Utils.h>
#include <Bifrost/Math/Distributions.h>
#include <Bifrost/Math/RNG.h>
#include <Bifrost/Math/Statistics.h>

#include <functional>

#include <gtest/gtest.h>

namespace Bifrost::Assets::Shading::BSDFTestUtils {

using namespace Bifrost::Math;

// Precompute the random numbers and make them available as a global constant,
// to make it easy to reuse across the BSDF sample test utils and avoid recomputing them multiple times.
static const RNG::PmjbRNG g_bsdf_rng(16384u);
inline Vector2f bsdf_rng_sample2f(int i) { return g_bsdf_rng.sample2f(i); }
inline Vector3f bsdf_rng_sample3f(int i, int max_sample_count) { return g_bsdf_rng.sample3f(i, max_sample_count); }

struct RhoResult {
    RGB reflectance;
    RGB std_dev;
    Vector3f mean_direction;

    // Normalize error wrt reflectance, so dark BSDFs don't automatically have a smaller error
    RGB normalized_std_dev() const { return std_dev / reflectance; }

    static RhoResult invalid() {
        RhoResult res;
        res.reflectance = res.std_dev = RGB(nanf(""));
        res.mean_direction = Vector3f(nanf(""));
        return res;
    }
};

template <typename BSDFModel>
inline RhoResult directional_hemispherical_reflectance_function(BSDFModel bsdf_model, Vector3f wo, unsigned int sample_count) {
    // Return an invalid result if more samples are requested than can be produced.
    if (g_bsdf_rng.m_max_sample_capacity < sample_count)
        return RhoResult::invalid();

    Statistics<double> reflectance_statistics[3] = { Statistics<double>(), Statistics<double>(), Statistics<double>() };
    Vector3d summed_directions = { 0.0, 0.0, 0.0 };
    for (unsigned int i = 0u; i < sample_count; ++i) {
        BSDFSample sample = bsdf_model.sample(wo, bsdf_rng_sample3f(i, sample_count));

        RGB reflectance = { 0, 0, 0 };
        if (sample.PDF.is_valid()) {
            reflectance = sample.reflectance * abs(sample.direction.z) / sample.PDF.value(); // f * ||cos_theta|| / pdf

            float direction_weight = sum(reflectance);
            summed_directions = { summed_directions.x + direction_weight * sample.direction.x,
                                  summed_directions.y + direction_weight * sample.direction.y,
                                  summed_directions.z + direction_weight * sample.direction.z };
        }

        reflectance_statistics[0].add(reflectance.r);
        reflectance_statistics[1].add(reflectance.g);
        reflectance_statistics[2].add(reflectance.b);
    }

    RGB mean_reflectance = { (float)reflectance_statistics[0].mean(),
                             (float)reflectance_statistics[1].mean(),
                             (float)reflectance_statistics[2].mean() };
    RGB reflectance_std_dev = { (float)reflectance_statistics[0].standard_deviation(),
                                (float)reflectance_statistics[1].standard_deviation(),
                                (float)reflectance_statistics[2].standard_deviation() };

    Vector3f direction = Vector3f(normalize(summed_directions));

    return { mean_reflectance, reflectance_std_dev, direction };
}

template <typename BSDFModel>
inline void BSDF_sampling_variance_test(BSDFModel bsdf_model, unsigned int sample_count, RGB expected_rho_std_dev, float epsilon = 0.01f) {
    RGB total_std_dev = { 0, 0, 0 };
    for (float cos_theta : {0.1f, 0.3f, 0.5f, 0.7f, 0.9f, 1.0f}) {
        Vector3f wo = w_from_cos_theta(cos_theta);
        auto rho = directional_hemispherical_reflectance_function(bsdf_model, wo, sample_count);
        RGB rho_std_dev = rho.normalized_std_dev();
        total_std_dev += rho_std_dev;
    }
    RGB average_std_dev = total_std_dev / 6;
    EXPECT_RGB_EQ_EPS(average_std_dev, expected_rho_std_dev, epsilon) << bsdf_model.to_string();
}

template <typename BSDFModel>
inline void BSDF_sampling_variance_test(BSDFModel bsdf_model, unsigned int sample_count, float expected_rho_std_dev, float epsilon = 0.01f) {
    BSDF_sampling_variance_test(bsdf_model, sample_count, RGB(expected_rho_std_dev), epsilon);
}

template <typename BSDFModel>
inline void helmholtz_reciprocity(BSDFModel bsdf_model, Vector3f wo, unsigned int sample_count) {
    for (unsigned int i = 0u; i < sample_count; ++i) {
        Vector3f rng_sample = bsdf_rng_sample3f(i, sample_count);
        BSDFSample sample = bsdf_model.sample(wo, rng_sample);

        if (sample.PDF.is_valid()) {
            RGB f = bsdf_model.evaluate(sample.direction, wo);
            EXPECT_RGB_EQ_EPS(sample.reflectance, f, 0.0001f) << bsdf_model.to_string();
        }
    }
}

template <typename BSDFModel>
inline void BSDF_consistency_test(BSDFModel bsdf_model, Vector3f wo, unsigned int sample_count) {
    for (unsigned int i = 0u; i < sample_count; ++i) {
        Vector3f rng_sample = bsdf_rng_sample3f(i, sample_count);
        BSDFSample sample = bsdf_model.sample(wo, rng_sample);

        if (sample.PDF.is_valid()) {
            EXPECT_GE(sample.reflectance.r, 0.0f) << bsdf_model.to_string() << ", cos_theta: " << wo.z;

            EXPECT_RGB_EQ_PCT(sample.reflectance, bsdf_model.evaluate(wo, sample.direction), 0.00002f) << bsdf_model.to_string() << ", cos_theta: " << wo.z;
            EXPECT_PDF_EQ_PCT(sample.PDF, bsdf_model.pdf(wo, sample.direction), 0.00002f) << bsdf_model.to_string() << ", cos_theta: " << wo.z;

            BSDFResponse response = bsdf_model.evaluate_with_PDF(wo, sample.direction);
            EXPECT_RGB_EQ_PCT(sample.reflectance, response.reflectance, 0.00002f) << bsdf_model.to_string() << ", cos_theta: " << wo.z;
            EXPECT_PDF_EQ_PCT(sample.PDF, response.PDF, 0.00002f) << bsdf_model.to_string() << ", cos_theta: " << wo.z;
        }
    }
}

// Sample BRDF over a sphere and validate that if the BRDF reflects light, then the PDF must be positive.
template <typename BSDFModel>
inline void PDF_positivity_test(BSDFModel bsdf_model, Vector3f wo, unsigned int sample_count) {
    for (unsigned int i = 0u; i < sample_count; ++i) {
        Math::Vector3f wi = Distributions::Sphere::sample_direction(bsdf_rng_sample2f(i));

        BSDFResponse sample = bsdf_model.evaluate_with_PDF(wo, wi);

        // Test that reflectance is never negative.
        EXPECT_GE(sample.reflectance.r, 0.0f) << bsdf_model.to_string();
        EXPECT_GE(sample.reflectance.g, 0.0f) << bsdf_model.to_string();
        EXPECT_GE(sample.reflectance.b, 0.0f) << bsdf_model.to_string();

        // Test that if the bsdf reflects light, then the PDF is positive.
        if (!is_black(sample.reflectance))
            EXPECT_GT(sample.PDF.value(), 0.0f) << bsdf_model.to_string() << ", cos_theta: " << wo.z;
    }
}

struct ThinSheetThroughput {
    RGB reflected;
    RGB transmitted;
};

inline ThinSheetThroughput integrate_over_thin_sheet(std::function<BSDFSample(Vector3f wo, Vector3f random_sample)> bsdf_model_sampler,
                                                     Vector3f wo, unsigned int path_count, unsigned int bounce_count = 8u) {
    Vector3d summed_reflection = { 0.0, 0.0, 0.0 };
    Vector3d summed_transmission = { 0.0, 0.0, 0.0 };
    for (unsigned int i = 0; i < path_count; ++i) {
        // Keep track of the ray state. The ray is either entering, bouncing inside the thin sheet, or exited.
        RGB throughput = { 1.0f, 1.0f, 1.0f };
        Vector3f ray_wo = wo;
        bool terminate_ray = false;
        bool escaped_ray_is_reflection = false;

        for (unsigned int bounce = 0; bounce < bounce_count && !terminate_ray; ++bounce) {

            // First bounce is from air to the sheet. All other bounces are from inside the sheet towards air.
            float hemisphere_sign = (bounce == 0) ? 1.0f : -1.0f;
            ray_wo.z = hemisphere_sign * abs(ray_wo.z);

            Vector4f rng_sample = RNG::PracticalScrambledSobol::sample4f(i, 0, bounce);
            BSDFSample bsdf_sample = bsdf_model_sampler(ray_wo, Vector3f(rng_sample.x, rng_sample.y, rng_sample.z));

            if (bsdf_sample.PDF.is_valid())
                throughput *= bsdf_sample.reflectance * abs(bsdf_sample.direction.z) / bsdf_sample.PDF.value(); // f * ||cos(theta)|| / pdf
            else {
                throughput = RGB::black();
                terminate_ray = true;
            }

            // Terminate the ray if the first interaction is a reflection or if the ray is inside the sheet and transmits
            bool is_inside = bounce > 0;
            bool transmission_out_of_sheet = is_inside && !same_hemisphere(bsdf_sample.direction, ray_wo);
            bool initial_reflection_event = bounce == 0 && bsdf_sample.direction.z >= 0.0f;
            if (initial_reflection_event || transmission_out_of_sheet)
                terminate_ray = true;

            ray_wo = bsdf_sample.direction;

            // As the ray is bouncing between the two surfaces of the sheet,
            // odd bounces escape as a reflection and even bounces as a transmission.
            escaped_ray_is_reflection = (bounce % 2) == 0;
        }

        if (escaped_ray_is_reflection) {
            summed_reflection.x += throughput.r;
            summed_reflection.y += throughput.g;
            summed_reflection.z += throughput.b;
        } else {
            summed_transmission.x += throughput.r;
            summed_transmission.y += throughput.g;
            summed_transmission.z += throughput.b;
        }
    }

    RGB reflected = RGB(summed_reflection.x, summed_reflection.y, summed_reflection.z) / float(path_count);
    RGB transmitted = RGB(summed_transmission.x, summed_transmission.y, summed_transmission.z) / float(path_count);

    return { reflected, transmitted };
}

// Compute the expected ratio of light reflected and transmitted of a smooth, thin, locally flat medium when viewed from the angle theta_o
inline ThinSheetThroughput smooth_thin_sheet_reflectance(float cos_theta_o, float medium_IOR, RGB transmission_tint) {
    // Medium cannot have lower IOR than air.
    if (medium_IOR <= air_ior)
        return { RGB(nanf("")), RGB(nanf("")) };

    float specularity = dielectric_specularity(air_ior, medium_IOR);
    RGB transmission_tint_per_side = { sqrt(transmission_tint.r), sqrt(transmission_tint.g), sqrt(transmission_tint.b) };

    float refracted_cos_theta;
    bool total_internal_reflection = !refract(refracted_cos_theta, -abs(cos_theta_o), medium_IOR / air_ior);
    if (total_internal_reflection)
        return { RGB::white(), RGB::black() };

    // The reflected and transmitted throughput of a thin sheet depends on the reflection and transmission at the initial intersection, R0 and T0,
    // and the reflection and transmission of the refract light bouncing inside the glass, Ri and Ti.
    // As every intersection after the first happens at the border of the glass, with air on the other side, they all have the same R and T values.
    float R0 = dielectric_schlick_fresnel(specularity, cos_theta_o, medium_IOR / air_ior);
    RGB T0 = transmission_tint_per_side * (1 - R0);
    float Ri = schlick_fresnel(specularity, abs(refracted_cos_theta));
    RGB Ti = transmission_tint_per_side * (1 - Ri);

    // The expected amount of light reflected, Re, is given by the amount of light reflected by the first intersection and
    // all the transmitted light that does an odd number of reflections inside the glass before transmitting and exiting on the same side as entered.
    // Re = R0 + T0 * Ri * Ti + T0 * Ri^3 * Ti + T0 * Ri^5 * Ti + ...
    //    = R + T0 * Ti * (Ri + Ri^3 + Ri^5 + ...)
    //    = R + T0 * Ti * Ri * (1 + Ri^2 + Ri^4 + ...)
    //    = R + T0 * Ti * Ri * 1 / (1 - Ri^2) <-- Use the geometric power series to get an expression for the infinite series.
    // Similarly, the expected amount of transmitted light, Te, is given by the light that transmits at the first intersection,
    // performs an even number of reflections inside the glass, and then exits with a final transmission event.
    // Te = T0 * Ti + T0 * Ri^2 * Ti + T0 * Ri^4 * Ti + T0 * Ri^6 * Ti + ...
    //    = T0 * Ti * (1 + Ri^2 + Ri^4 + Ri^6 + ...)
    //    = T0 * Ti * 1 / (1 - Ri^2) <-- Use the geometric power series to get an expression for the infinite series.
    RGB reflected = R0 + (T0 * Ti * Ri) / (1 - Ri * Ri);
    RGB transmitted = (T0 * Ti) / (1 - Ri * Ri);

    return { reflected, transmitted };
}

// Approximate thin sheet reflectance.
// The algorithm is based on the exact solution for smooth thin sheets found in BSDFTestUtils::smooth_thin_sheet_reflectance.
// For rough thin sheets the solution is approximated by replacing the exact smooth reflectance with the precomputed rough reflectance.
inline ThinSheetThroughput approx_thin_sheet_reflectance(float abs_cos_theta, float roughness, float ior_i_over_o, RGB transmission_tint) {
    // Compute the representative refracted cos(theta) that the opposite side of the sheet is observed from.
    float refracted_cos_theta;
    bool total_internal_reflection = !refract(refracted_cos_theta, -abs_cos_theta, ior_i_over_o);
    if (total_internal_reflection)
        return { RGB::white(), RGB::black() };

    // Compute the reflected rho for each different type of intersection and the combined rho for the transmission.
    auto rho0 = Bifrost::Assets::Shading::Rho::sample_dielectric_GGX(abs_cos_theta, roughness, ior_i_over_o);
    float R0 = rho0.reflected_rho / rho0.total_rho; // Compensate for energy-loss by dividing by total rho
    float T0 = 1 - R0;
    // NOTE For the transmission rho the relative IOR should be inverted, as the ray is coming from the backside.
    // But for some reason that doesn't give the correct result when comparing with smooth surfaces, so we'll leave it like it is.
    auto rhoi = Bifrost::Assets::Shading::Rho::sample_dielectric_GGX(abs(refracted_cos_theta), roughness, ior_i_over_o);
    float Ri = rhoi.reflected_rho / rhoi.total_rho; // Compensate for energy-loss by dividing by total rho
    float Ti = 1 - Ri;
    RGB T0Ti = transmission_tint * T0 * Ti;

    // Implementation that reuses computations for speed.
    // See smooth_thin_sheet_reflectance for the derivation of the terms.
    RGB transmitted = T0Ti / (1 - Ri * Ri);
    RGB reflected = R0 + transmitted * Ri;

    return { reflected, transmitted };
}

inline Vector3f w_from_cos_theta(float cos_theta) {
    return { sqrt(1 - pow2(cos_theta)), 0.0f, cos_theta };
}

} // NS Bifrost::Assets::Shading::BSDFTestUtils

#endif // _BIFROST_ASSETS_SHADING_BSDF_TEST_UTILS_H_