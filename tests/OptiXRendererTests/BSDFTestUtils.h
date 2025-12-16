// Test utils for OptiXRenderer's BSDFs.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _OPTIXRENDERER_BSDF_TEST_UTILS_H_
#define _OPTIXRENDERER_BSDF_TEST_UTILS_H_

#include <Utils.h>

#include <Bifrost/Math/RNG.h>
#include <Bifrost/Math/Statistics.h>

#include <OptiXRenderer/Distributions.h>
#include <OptiXRenderer/Utils.h>

#include <gtest/gtest.h>

namespace OptiXRenderer {
namespace BSDFTestUtils {

using namespace optix;

// Precompute the random numbers and make them available as a global constant,
// to make it easy to reuse across the BSDF sample test utils and avoid recomputing them multiple times.
static const Bifrost::Math::RNG::PmjbRNG g_bsdf_rng(16384u);
inline float2 bsdf_rng_sample2f(int i) { auto s = g_bsdf_rng.sample2f(i); return { s.x, s.y }; }
inline float3 bsdf_rng_sample3f(int i, int max_sample_count) { auto s = g_bsdf_rng.sample3f(i, max_sample_count); return { s.x, s.y, s.z }; }

struct RhoResult {
    float3 reflectance;
    float3 std_dev;
    float3 mean_direction;

    // Normalize error wrt reflectance, so dark BSDFs don't automatically have a smaller error
    float3 normalized_std_dev() const { return std_dev / reflectance; }

    static RhoResult invalid() {
        RhoResult res;
        res.reflectance = res.std_dev = res.mean_direction = make_float3(nanf(""));
        return res;
    }
};

template <typename BSDFModel>
inline RhoResult directional_hemispherical_reflectance_function(BSDFModel bsdf_model, float3 wo, unsigned int sample_count) {
    using namespace Bifrost::Math;
    using namespace optix;

    // Return an invalid result if more samples are requested than can be produced.
    if (g_bsdf_rng.m_max_sample_capacity < sample_count)
        return RhoResult::invalid();

    Statistics<double> reflectance_statistics[3] = { Statistics<double>(), Statistics<double>(), Statistics<double>() };
    double3 summed_directions = { 0.0, 0.0, 0.0 };
    for (unsigned int i = 0u; i < sample_count; ++i) {
        BSDFSample sample = bsdf_model.sample(wo, bsdf_rng_sample3f(i, sample_count));

        float3 reflectance = { 0, 0, 0 };
        if (sample.PDF.is_valid()) {
            reflectance = sample.reflectance * abs(sample.direction.z) / sample.PDF.value(); // f * ||cos_theta|| / pdf

            float direction_weight = sum(reflectance);
            summed_directions = { summed_directions.x + direction_weight * sample.direction.x,
                                  summed_directions.y + direction_weight * sample.direction.y,
                                  summed_directions.z + direction_weight * sample.direction.z };
        }

        reflectance_statistics[0].add(reflectance.x);
        reflectance_statistics[1].add(reflectance.y);
        reflectance_statistics[2].add(reflectance.z);
    }

    float3 mean_reflectance = { (float)reflectance_statistics[0].mean(),
                                (float)reflectance_statistics[1].mean(),
                                (float)reflectance_statistics[2].mean() };
    float3 reflectance_std_dev = { (float)reflectance_statistics[0].standard_deviation(),
                                   (float)reflectance_statistics[1].standard_deviation(),
                                   (float)reflectance_statistics[2].standard_deviation() };

    float3 direction = normalize(make_float3(summed_directions));

    return { mean_reflectance, reflectance_std_dev, direction };
}

template <typename BSDFModel>
inline void BSDF_sampling_variance_test(BSDFModel bsdf_model, unsigned int sample_count, optix::float3 expected_rho_std_dev, float epsilon = 0.01f) {
    optix::float3 total_std_dev = { 0, 0, 0 };
    for (float cos_theta : {0.1f, 0.3f, 0.5f, 0.7f, 0.9f, 1.0f}) {
        float3 wo = w_from_cos_theta(cos_theta);
        auto rho = directional_hemispherical_reflectance_function(bsdf_model, wo, sample_count);
        optix::float3 rho_std_dev = rho.normalized_std_dev();
        total_std_dev += rho_std_dev;
    }
    optix::float3 average_std_dev = total_std_dev / 6;
    EXPECT_FLOAT3_EQ_EPS(average_std_dev, expected_rho_std_dev, epsilon) << bsdf_model.to_string();
}

template <typename BSDFModel>
inline void BSDF_sampling_variance_test(BSDFModel bsdf_model, unsigned int sample_count, float expected_rho_std_dev, float epsilon = 0.01f) {
    BSDF_sampling_variance_test(bsdf_model, sample_count, optix::make_float3(expected_rho_std_dev), epsilon);
}

template <typename BSDFModel>
inline void helmholtz_reciprocity(BSDFModel bsdf_model, float3 wo, unsigned int sample_count) {
    for (unsigned int i = 0u; i < sample_count; ++i) {
        float3 rng_sample = bsdf_rng_sample3f(i, sample_count);
        BSDFSample sample = bsdf_model.sample(wo, rng_sample);

        if (sample.PDF.is_valid()) {
            float3 f = bsdf_model.evaluate(sample.direction, wo);
            EXPECT_COLOR_EQ_EPS(sample.reflectance, f, 0.0001f) << bsdf_model.to_string();
        }
    }
}

template <typename BSDFModel>
inline void BSDF_consistency_test(BSDFModel bsdf_model, float3 wo, unsigned int sample_count) {
    for (unsigned int i = 0u; i < sample_count; ++i) {
        float3 rng_sample = bsdf_rng_sample3f(i, sample_count);
        BSDFSample sample = bsdf_model.sample(wo, rng_sample);

        if (sample.PDF.is_valid()) {
            EXPECT_GE(sample.reflectance.x, 0.0f) << bsdf_model.to_string() << ", cos_theta: " << wo.z;

            EXPECT_PDF_EQ_PCT(sample.PDF, bsdf_model.pdf(wo, sample.direction), 0.00002f) << bsdf_model.to_string() << ", cos_theta: " << wo.z;
            EXPECT_COLOR_EQ_PCT(sample.reflectance, bsdf_model.evaluate(wo, sample.direction), 0.00002f) << bsdf_model.to_string() << ", cos_theta: " << wo.z;

            BSDFResponse response = bsdf_model.evaluate_with_PDF(wo, sample.direction);
            EXPECT_COLOR_EQ_PCT(sample.reflectance, response.reflectance, 0.00002f) << bsdf_model.to_string() << ", cos_theta: " << wo.z;
            EXPECT_PDF_EQ_PCT(sample.PDF, response.PDF, 0.00002f) << bsdf_model.to_string() << ", cos_theta: " << wo.z;
        }
    }
}

// Sample BRDF over a sphere and validate that if the BRDF reflects light, then the PDF must be positive.
template <typename BSDFModel>
inline void PDF_positivity_test(BSDFModel bsdf_model, optix::float3 wo, unsigned int sample_count) {
    using namespace optix;

    for (unsigned int i = 0u; i < sample_count; ++i) {
        auto wi = Distributions::UniformSphere::sample(bsdf_rng_sample2f(i)).direction;

        BSDFResponse sample = bsdf_model.evaluate_with_PDF(wo, wi);

        // Test that reflectance is never negative.
        EXPECT_GE(sample.reflectance.x, 0.0f) << bsdf_model.to_string();
        EXPECT_GE(sample.reflectance.y, 0.0f) << bsdf_model.to_string();
        EXPECT_GE(sample.reflectance.z, 0.0f) << bsdf_model.to_string();

        // Test that if the bsdf reflects light, then the PDF is positive.
        if (!is_black(sample.reflectance))
            EXPECT_GT(sample.PDF.value(), 0.0f) << bsdf_model.to_string() << ", cos_theta: " << wo.z;
    }
}

struct ThinSheetThroughput {
    optix::float3 reflected;
    optix::float3 transmitted;
};

inline ThinSheetThroughput integrate_over_thin_sheet(std::function<BSDFSample(optix::float3 wo, optix::float3 random_sample)> bsdf_model_sampler,
                                                     optix::float3 wo, unsigned int path_count, unsigned int bounce_count = 8u) {
    using namespace optix;

    double3 summed_reflection = { 0.0, 0.0, 0.0 };
    double3 summed_transmission = { 0.0, 0.0, 0.0 };
    for (unsigned int i = 0; i < path_count; ++i) {
        // Keep track of the ray state. The ray is either entering, bouncing inside the thin sheet, or exited.
        float3 throughput = { 1.0f, 1.0f, 1.0f };
        float3 ray_wo = wo;
        bool terminate_ray = false;
        bool escaped_ray_is_reflection = false;

        for (unsigned int bounce = 0; bounce < bounce_count && !terminate_ray; ++bounce) {

            // First bounce is from air to the sheet. All other bounces are from inside the sheet towards air.
            float hemisphere_sign = (bounce == 0) ? 1.0f : -1.0f;
            ray_wo.z = hemisphere_sign * abs(ray_wo.z);

            float4 rng_sample = RNG::PracticalScrambledSobol::sample4f(i, 0, bounce);
            BSDFSample bsdf_sample = bsdf_model_sampler(ray_wo, make_float3(rng_sample));

            if (bsdf_sample.PDF.is_valid())
                throughput *= bsdf_sample.reflectance * abs(bsdf_sample.direction.z) / bsdf_sample.PDF.value(); // f * ||cos(theta)|| / pdf
            else {
                throughput = make_float3(0.0f);
                terminate_ray = true;
            }

            // Terminate the ray if the first interaction is a reflection or if the ray is inside the sheet and transmits
            bool is_inside = bounce > 0;
            bool transmission_out_of_sheet = is_inside && sign(bsdf_sample.direction.z) != sign(ray_wo.z);
            bool initial_reflection_event = bounce == 0 && bsdf_sample.direction.z >= 0.0f;
            if (initial_reflection_event || transmission_out_of_sheet)
                terminate_ray = true;

            ray_wo = bsdf_sample.direction;

            // As the ray is bouncing between the two surfaces of the sheet,
            // odd bounces escape as a reflection and even bounces as a transmission.
            escaped_ray_is_reflection = (bounce % 2) == 0;
        }

        if (escaped_ray_is_reflection) {
            summed_reflection.x += throughput.x;
            summed_reflection.y += throughput.y;
            summed_reflection.z += throughput.z;
        } else {
            summed_transmission.x += throughput.x;
            summed_transmission.y += throughput.y;
            summed_transmission.z += throughput.z;
        }
    }

    float3 reflected = make_float3(summed_reflection) / float(path_count);
    float3 transmitted = make_float3(summed_transmission) / float(path_count);

    return { reflected, transmitted };
}

// Compute the expected ratio of light reflected and transmitted of a smooth, thin, locally flat medium when viewed from the angle theta_o
inline ThinSheetThroughput smooth_thin_sheet_reflectance(float cos_theta_o, float medium_IOR, optix::float3 transmission_tint) {
    // Medium cannot have lower IOR than air.
    if (medium_IOR <= AIR_IOR)
        return { optix::make_float3(nanf("")), optix::make_float3(nanf("")) };

    float specularity = dielectric_specularity(AIR_IOR, medium_IOR);
    optix::float3 transmission_tint_per_side = sqrt3(transmission_tint);

    float refracted_cos_theta;
    bool total_internal_reflection = !refract(refracted_cos_theta, -abs(cos_theta_o), medium_IOR / AIR_IOR);
    if (total_internal_reflection)
        return { optix::make_float3(1), optix::make_float3(0) };

    // The reflected and transmitted throughput of a thin sheet depends on the reflection and transmission at the initial intersection, R0 and T0,
    // and the reflection and transmission of the refract light bouncing inside the glass, Ri and Ti.
    // As every intersection after the first happens at the border of the glass, with air on the other side, they all have the same R and T values.
    float R0 = dielectric_schlick_fresnel(specularity, cos_theta_o, medium_IOR / AIR_IOR);
    optix::float3 T0 = (1 - R0) * transmission_tint_per_side;
    float Ri = schlick_fresnel(specularity, abs(refracted_cos_theta));
    optix::float3 Ti = (1 - Ri) * transmission_tint_per_side;

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
    optix::float3 reflected = R0 + (Ri * T0 * Ti) / (1 - Ri * Ri);
    optix::float3 transmitted = (T0 * Ti) / (1 - Ri * Ri);

    return { reflected, transmitted };
}

inline float3 w_from_cos_theta(float cos_theta) {
    return { sqrt(1 - pow2(cos_theta)), 0.0f, cos_theta };
}

} // NS BSDFTestUtils
} // NS OptiXRenderer

#endif // _OPTIXRENDERER_BSDF_TEST_UTILS_H_