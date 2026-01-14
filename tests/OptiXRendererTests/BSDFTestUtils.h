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

struct PmjbRNG {
    unsigned int m_max_sample_capacity;
    Bifrost::Math::Vector2f* m_samples;

    PmjbRNG(unsigned int max_sample_capacity) {
        m_max_sample_capacity = max_sample_capacity;
        m_samples = new Bifrost::Math::Vector2f[max_sample_capacity];
        Bifrost::Math::RNG::fill_progressive_multijittered_bluenoise_samples(m_samples, m_samples + max_sample_capacity);
    }
    PmjbRNG(PmjbRNG& other) = delete;
    PmjbRNG(PmjbRNG&& other) = default;

    PmjbRNG& operator=(PmjbRNG& rhs) = delete;
    PmjbRNG& operator=(PmjbRNG&& rhs) = default;

    ~PmjbRNG() { delete[] m_samples; }

    float2 sample_2f(int i) const { return make_float2(m_samples[i].x, m_samples[i].y); }
    float3 sample_3f(int i, int max_sample_count) const { return make_float3(sample_2f(i), (i + 0.5f) / max_sample_count); }
};

// Precompute the random numbers and make them available as a global constant,
// to make it easy to reuse across the BSDF sample test utils and avoid recomputing them multiple times.
static const PmjbRNG g_rng = PmjbRNG(16384u);

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
RhoResult directional_hemispherical_reflectance_function(BSDFModel bsdf_model, float3 wo, unsigned int sample_count) {
    using namespace Bifrost::Math;
    using namespace optix;

    // Return an invalid result if more samples are requested than can be produced.
    if (g_rng.m_max_sample_capacity < sample_count)
        return RhoResult::invalid();

    Statistics<double> reflectance_statistics[3] = { Statistics<double>(), Statistics<double>(), Statistics<double>() };
    double3 summed_directions = { 0.0, 0.0, 0.0 };
    for (unsigned int i = 0u; i < sample_count; ++i) {
        BSDFSample sample = bsdf_model.sample(wo, g_rng.sample_3f(i, sample_count));

        float3 reflectance = { 0, 0, 0 };
        if (sample.PDF.is_valid()) {
            reflectance = sample.reflectance * abs(sample.direction.z) / sample.PDF.value(); // f * ||cos_theta|| / pdf

            float direction_weight = sum(sample.reflectance) / sample.PDF.value();
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

    float3 direction = { float(summed_directions.x), float(summed_directions.y), float(summed_directions.z) };

    return { mean_reflectance, reflectance_std_dev, normalize(direction) };
}

template <typename BSDFModel>
void BSDF_sampling_variance_test(BSDFModel bsdf_model, unsigned int sample_count, optix::float3 expected_rho_std_dev, float epsilon = 0.01f) {
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
void BSDF_sampling_variance_test(BSDFModel bsdf_model, unsigned int sample_count, float expected_rho_std_dev, float epsilon = 0.01f) {
    BSDF_sampling_variance_test(bsdf_model, sample_count, optix::make_float3(expected_rho_std_dev), epsilon);
}

template <typename BSDFModel>
void helmholtz_reciprocity(BSDFModel bsdf_model, float3 wo, unsigned int sample_count) {
    for (unsigned int i = 0u; i < sample_count; ++i) {
        float3 rng_sample = g_rng.sample_3f(i, sample_count);
        BSDFSample sample = bsdf_model.sample(wo, rng_sample);

        if (sample.PDF.is_valid()) {
            float3 f = bsdf_model.evaluate(sample.direction, wo);
            EXPECT_COLOR_EQ_EPS(sample.reflectance, f, 0.0001f) << bsdf_model.to_string();
        }
    }
}

template <typename BSDFModel>
void BSDF_consistency_test(BSDFModel bsdf_model, float3 wo, unsigned int sample_count) {
    for (unsigned int i = 0u; i < sample_count; ++i) {
        float3 rng_sample = g_rng.sample_3f(i, sample_count);
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
void PDF_positivity_test(BSDFModel bsdf_model, optix::float3 wo, unsigned int sample_count) {
    using namespace optix;

    for (unsigned int i = 0u; i < sample_count; ++i) {
        auto wi = Distributions::UniformSphere::sample(g_rng.sample_2f(i)).direction;

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

float3 w_from_cos_theta(float cos_theta) {
    return { sqrt(1 - pow2(cos_theta)), 0.0f, cos_theta };
}

} // NS BSDFTestUtils
} // NS OptiXRenderer

#endif // _OPTIXRENDERER_BSDF_TEST_UTILS_H_