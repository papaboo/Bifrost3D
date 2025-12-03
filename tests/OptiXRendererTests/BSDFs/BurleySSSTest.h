// Test OptiXRenderer's Burley BSSRDF.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _OPTIXRENDERER_BSDFS_BURLEY_SSS_TEST_H_
#define _OPTIXRENDERER_BSDFS_BURLEY_SSS_TEST_H_

#include <BSDFTestUtils.h>

#include <OptiXRenderer/Shading/BSDFs/BurleySSS.h>

#include <gtest/gtest.h>

namespace OptiXRenderer {

class BurleySSSWrapper {
public:
    enum class SamplingStrategy { SampleMostScattering, ImportanceSampleChannels };

    Shading::BSDFs::BurleySSS::Parameters m_parameters;
    SamplingStrategy m_sampling_strategy = SamplingStrategy::ImportanceSampleChannels;

    BurleySSSWrapper(optix::float3 albedo, optix::float3 mean_free_path)
        : m_parameters(Shading::BSDFs::BurleySSS::Parameters::create(albedo, mean_free_path)) {}

    void set_sampling_strategy(SamplingStrategy sampling_strategy) {
        m_sampling_strategy = sampling_strategy;
    }

    // Standard subsurface scattering parameters.
    // https://rmanwiki-25.pixar.com/space/REN25/20416088/Subsurface+Scattering+Parameters
    static BurleySSSWrapper ketchup() { return BurleySSSWrapper({ 0.164f, 0.006f, 0.002f }, { 4.76f, 0.58f, 0.39f }); }
    static BurleySSSWrapper marble() { return BurleySSSWrapper({ 0.830f, 0.791f, 0.753f }, { 8.51f, 5.57f, 3.95f }); }
    static BurleySSSWrapper potato() { return BurleySSSWrapper({ 0.764f, 0.613f, 0.213f }, { 14.27f, 7.23f, 2.04f }); }
    static BurleySSSWrapper skin1() { return BurleySSSWrapper({ 0.436f, 0.227f, 0.131f }, { 3.67f, 1.37f, 0.68f }); }
    static BurleySSSWrapper whole_milk() { return BurleySSSWrapper({ 0.908f, 0.881f, 0.759f }, { 10.90f, 6.58f, 2.51f }); }

    optix::float3 evaluate(optix::float3 po, optix::float3 pi) const {
        return Shading::BSDFs::BurleySSS::evaluate(m_parameters, po, pi);
    }

    PDF pdf(optix::float3 po, optix::float3 pi) const {
        if (m_sampling_strategy == SamplingStrategy::SampleMostScattering)
            return Shading::BSDFs::BurleySSS::SampleMostScattering::pdf(m_parameters, po, pi);
        else
            return Shading::BSDFs::BurleySSS::ImportanceSampleChannels::pdf(m_parameters, po, pi);
    }

    BSDFResponse evaluate_with_PDF(optix::float3 po, optix::float3 pi) const {
        if (m_sampling_strategy == SamplingStrategy::SampleMostScattering)
            return Shading::BSDFs::BurleySSS::SampleMostScattering::evaluate_with_PDF(m_parameters, po, pi);
        else
            return Shading::BSDFs::BurleySSS::ImportanceSampleChannels::evaluate_with_PDF(m_parameters, po, pi);
    }

    BSSRDFSample sample(optix::float3 po, optix::float3 random_sample) const {
        if (m_sampling_strategy == SamplingStrategy::SampleMostScattering)
            return Shading::BSDFs::BurleySSS::SampleMostScattering::sample(m_parameters, po, random_sample);
        else
            return Shading::BSDFs::BurleySSS::ImportanceSampleChannels::sample(m_parameters, po, random_sample);
    }

    std::string to_string() const {
        std::ostringstream out;
        out << "Burley: albedo: " << m_parameters.albedo << ", scattering distance: " << m_parameters.scattering_distance;
        return out.str();
    }
};

template <typename BSSRDFModel>
BSDFTestUtils::RhoResult directional_hemispherical_reflectance_function(BSSRDFModel bssrdf_model, unsigned int sample_count) {
    using namespace Bifrost::Math;
    using namespace optix;

    // Return an invalid result if more samples are requested than can be produced.
    if (BSDFTestUtils::g_rng.m_max_sample_capacity < sample_count)
        return BSDFTestUtils::RhoResult::invalid();

    const float3 po = { 1.0f, -2.0f, 4.0f };

    Statistics<double> reflectance_statistics[3] = { Statistics<double>(), Statistics<double>(), Statistics<double>() };
    double3 summed_directions = { 0.0, 0.0, 0.0 };
    for (unsigned int i = 0u; i < sample_count; ++i) {
        BSSRDFSample sample = bssrdf_model.sample(po, BSDFTestUtils::g_rng.sample_3f(i, sample_count));

        float3 reflectance = { 0, 0, 0 };
        if (sample.PDF.is_valid()) {
            reflectance = sample.reflectance / sample.PDF.value();

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

    float3 direction = { float(summed_directions.x), float(summed_directions.y), float(summed_directions.z) };

    return { mean_reflectance, reflectance_std_dev, normalize(direction) };
}

/*
GTEST_TEST(BurleySSS, power_conservation) {
    optix::float3 wo = optix::normalize(optix::make_float3(1.0f, 1.0f, 1.0f));
    for (float roughness : {0.0f, 0.2f, 0.4f, 0.6f, 0.8f, 1.0f}) {
        BurleyWrapper burley = BurleyWrapper(roughness);
        auto res = BSDFTestUtils::directional_hemispherical_reflectance_function(burley, wo, 1024u);
        EXPECT_FLOAT3_LE(res.reflectance, 1.00045f) << burley.to_string();
    }
}

GTEST_TEST(BurleySSS, Helmholtz_reciprocity) {
    optix::float3 wo = optix::normalize(optix::make_float3(1.0f, 1.0f, 1.0f));
    for (float roughness : {0.0f, 0.2f, 0.4f, 0.6f, 0.8f, 1.0f}) {
        BurleyWrapper burley = BurleyWrapper(roughness);
        BSDFTestUtils::helmholtz_reciprocity(burley, wo, 16u);
    }
}
*/

GTEST_TEST(BurleySSS, function_consistency) {
    using namespace optix;

    const int sample_count = 16;

    float3 albedo = { 0.3f, 0.9f, 0.5f };
    float3 mean_free_path = { 1, 3, 7 };
    float3 po = { 1.0f, -2.0f, 4.0f };
    BurleySSSWrapper burley = BurleySSSWrapper(albedo, mean_free_path);
    for (auto sampling_strategy : { BurleySSSWrapper::SamplingStrategy::SampleMostScattering, BurleySSSWrapper::SamplingStrategy::ImportanceSampleChannels })
        burley.set_sampling_strategy(sampling_strategy);
        for (unsigned int i = 0u; i < sample_count; ++i) {
            float3 rng_sample = BSDFTestUtils::g_rng.sample_3f(i, sample_count);
            BSSRDFSample sample = burley.sample(po, rng_sample);

            if (sample.PDF.is_valid()) {
                EXPECT_GE(sample.reflectance.x, 0.0f) << burley.to_string();

                EXPECT_PDF_EQ_PCT(sample.PDF, burley.pdf(po, sample.pi), 0.00002f) << burley.to_string();
                EXPECT_COLOR_EQ_PCT(sample.reflectance, burley.evaluate(po, sample.pi), 0.00002f) << burley.to_string();

                BSDFResponse response = burley.evaluate_with_PDF(po, sample.pi);
                EXPECT_COLOR_EQ_PCT(sample.reflectance, response.reflectance, 0.00002f) << burley.to_string();
                EXPECT_PDF_EQ_PCT(sample.PDF, response.PDF, 0.00002f) << burley.to_string();
            }
        }
}

GTEST_TEST(BurleySSS, sampling_standard_deviation) {
    using namespace optix;

    const int sample_count = 1024;

    // Uniform albedo and mean free path to avoid variance from importance sampling different albedos and mean free path
    for (float albedo : { 0.1f, 0.5f })
        for (float mean_free_path : { 0.2f, 0.5f, 1.0f}) {

            BurleySSSWrapper burley = BurleySSSWrapper(make_float3(albedo), make_float3(mean_free_path));
            for (auto sampling_strategy : { BurleySSSWrapper::SamplingStrategy::SampleMostScattering, BurleySSSWrapper::SamplingStrategy::ImportanceSampleChannels }) {
                burley.set_sampling_strategy(sampling_strategy);

                auto rho = directional_hemispherical_reflectance_function(burley, 1024);

                // Reflectance converges to albedo
                EXPECT_FLOAT_EQ_EPS(albedo, rho.reflectance.x, 0.0001f) << burley.to_string();

                // Sampling should have low standard deviation.
                float3 normalized_std_dev = rho.normalized_std_dev();
                EXPECT_LT(normalized_std_dev.x, 0.0001) << burley.to_string();
            }
        }
}



// TODO Sample according to albedo

// TODO Handle scattering_distance tending towards 0 gracefully, i.e. revert to albedo

// TODO Invalid samples when albedo or scattering distance is 'high'. Happens for standard values, but does it happen often?

} // NS OptiXRenderer

#endif // _OPTIXRENDERER_BSDFS_BURLEY_SSS_TEST_H_