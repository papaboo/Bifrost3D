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
    enum class SamplingStrategy { SampleMostScattering, AlbedoMIS, KarisApproximation };

    static constexpr SamplingStrategy sampling_strategies[3] = { SamplingStrategy::SampleMostScattering, SamplingStrategy::AlbedoMIS, SamplingStrategy::KarisApproximation };

    Shading::BSDFs::BurleySSS::Parameters m_parameters;
    SamplingStrategy m_sampling_strategy = SamplingStrategy::AlbedoMIS;

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

    optix::float3 get_albedo() const { return m_parameters.albedo; }

    optix::float3 evaluate(optix::float3 po, optix::float3 pi) const {
        return Shading::BSDFs::BurleySSS::evaluate(m_parameters, po, pi);
    }

    PDF pdf(optix::float3 po, optix::float3 pi) const {
        if (m_sampling_strategy == SamplingStrategy::SampleMostScattering)
            return Shading::BSDFs::BurleySSS::SampleMostScattering::pdf(m_parameters, po, pi);
        else if (m_sampling_strategy == SamplingStrategy::KarisApproximation)
            return Shading::BSDFs::BurleySSS::ApproximateSampling::pdf(m_parameters, po, pi);
        else
            return Shading::BSDFs::BurleySSS::AlbedoMIS::pdf(m_parameters, po, pi);
    }

    BSDFResponse evaluate_with_PDF(optix::float3 po, optix::float3 pi) const {
        if (m_sampling_strategy == SamplingStrategy::SampleMostScattering)
            return Shading::BSDFs::BurleySSS::SampleMostScattering::evaluate_with_PDF(m_parameters, po, pi);
        else if (m_sampling_strategy == SamplingStrategy::KarisApproximation)
            return Shading::BSDFs::BurleySSS::ApproximateSampling::evaluate_with_PDF(m_parameters, po, pi);
        else
            return Shading::BSDFs::BurleySSS::AlbedoMIS::evaluate_with_PDF(m_parameters, po, pi);
    }

    SeparableBSSRDFPositionSample sample(optix::float3 po, optix::float3 random_sample) const {
        if (m_sampling_strategy == SamplingStrategy::SampleMostScattering)
            return Shading::BSDFs::BurleySSS::SampleMostScattering::sample(m_parameters, po, random_sample);
        else if (m_sampling_strategy == SamplingStrategy::KarisApproximation)
            return Shading::BSDFs::BurleySSS::ApproximateSampling::sample(m_parameters, po, random_sample);
        else
            return Shading::BSDFs::BurleySSS::AlbedoMIS::sample(m_parameters, po, random_sample);
    }

    std::string to_string() const {
        std::ostringstream out;
        out << "Burley: albedo: " << m_parameters.albedo << ", diffuse mean free path: " << m_parameters.diffuse_mean_free_path;
        return out.str();
    }
};

template <typename BSSRDFModel>
BSDFTestUtils::RhoResult directional_hemispherical_reflectance_function(BSSRDFModel bssrdf_model, unsigned int sample_count) {
    using namespace Bifrost::Math;
    using namespace optix;

    // Return an invalid result if more samples are requested than can be produced.
    if (BSDFTestUtils::g_bsdf_rng.m_max_sample_capacity < sample_count)
        return BSDFTestUtils::RhoResult::invalid();

    const float3 po = { 1.0f, -2.0f, 4.0f };

    Statistics<double> reflectance_statistics[3] = { Statistics<double>(), Statistics<double>(), Statistics<double>() };
    for (unsigned int i = 0u; i < sample_count; ++i) {
        float3 rng = BSDFTestUtils::bsdf_rng_sample3f(i, sample_count);
        auto sample = bssrdf_model.sample(po, rng);

        // All sampled PDFs are valid, so the validity check is avoided in order to not bias the result by ignoring samples with low PDF.
        float3 reflectance = sample.reflectance / sample.PDF.value();

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

    return { mean_reflectance, reflectance_std_dev, make_float3(0, 0, 1) };
}

GTEST_TEST(BurleySSS, power_conservation) {
    optix::float3 white = optix::make_float3(1.0f);
    for (float mean_free_path : { 0.2f, 1.0f, 5.0f, 15.0f }) {
        BurleySSSWrapper bssrdf = BurleySSSWrapper(white, optix::make_float3(mean_free_path));
        auto reflectance = directional_hemispherical_reflectance_function(bssrdf, 4096).reflectance;
        EXPECT_FLOAT3_EQ_EPS(reflectance, white, 0.00045f) << bssrdf.to_string();
    }
}

GTEST_TEST(BurleySSS, reciprocity) {
    using namespace optix;

    auto bssrdf = BurleySSSWrapper::ketchup();
    float3 po = { 0.0f, 0.0f, 0.0f };
    for (float3 pi : { make_float3(0.1f, -0.2f, 0.3f), make_float3(0.1f, -0.2f, -0.3f), make_float3(-0.1f, 1.2f, 0.6f)}) {
        float3 scattering0 = bssrdf.evaluate(po, pi);
        float3 scattering1 = bssrdf.evaluate(pi, po);
        EXPECT_FLOAT3_EQ(scattering0, scattering1);
    }
}

GTEST_TEST(BurleySSS, function_consistency) {
    using namespace optix;

    const int sample_count = 16;

    float3 albedo = { 0.3f, 0.9f, 0.5f };
    float3 mean_free_path = { 1, 3, 7 };
    float3 po = { 1.0f, -2.0f, 4.0f };
    BurleySSSWrapper burley = BurleySSSWrapper(albedo, mean_free_path);
    for (auto sampling_strategy : BurleySSSWrapper::sampling_strategies)
        burley.set_sampling_strategy(sampling_strategy);
        for (unsigned int i = 0u; i < sample_count; ++i) {
            float3 rng_sample = BSDFTestUtils::bsdf_rng_sample3f(i, sample_count);
            auto sample = burley.sample(po, rng_sample);

            if (sample.PDF.is_valid()) {
                EXPECT_GE(sample.reflectance.x, 0.0f) << burley.to_string();

                EXPECT_PDF_EQ_PCT(sample.PDF, burley.pdf(po, sample.position), 0.00002f) << burley.to_string();
                EXPECT_COLOR_EQ_PCT(sample.reflectance, burley.evaluate(po, sample.position), 0.00002f) << burley.to_string();

                BSDFResponse response = burley.evaluate_with_PDF(po, sample.position);
                EXPECT_COLOR_EQ_PCT(sample.reflectance, response.reflectance, 0.00002f) << burley.to_string();
                EXPECT_PDF_EQ_PCT(sample.PDF, response.PDF, 0.00002f) << burley.to_string();
            }
        }
}

GTEST_TEST(BurleySSS, sampling_correctness) {
    using namespace optix;

    const int sample_count = 1024;

    for (auto sampling_strategy : BurleySSSWrapper::sampling_strategies) {

        float summed_std_dev = 0.0f;
        for (auto bssrdf : { BurleySSSWrapper::ketchup(), BurleySSSWrapper::skin1(), BurleySSSWrapper::marble() }) {
            bssrdf.set_sampling_strategy(sampling_strategy);

            auto rho = directional_hemispherical_reflectance_function(bssrdf, 4096);

            // Reflectance converges to albedo
            bool is_approximated_sampling = sampling_strategy == BurleySSSWrapper::SamplingStrategy::KarisApproximation;
            float epsilon = is_approximated_sampling ? 0.02f : 0.01f;
            EXPECT_COLOR_EQ_PCT(rho.reflectance, bssrdf.get_albedo(), epsilon);

            float variance = average(pow2(rho.std_dev));
            auto std_dev = sqrt(variance);
            summed_std_dev += std_dev;
        }
        float average_std_dev = summed_std_dev / 3.0f;

        // Sampling should have expected standard deviation across all materials.
        if (sampling_strategy == BurleySSSWrapper::SamplingStrategy::SampleMostScattering)
            EXPECT_LT(average_std_dev, 0.1745f) << " for most scattering sampling";
        else if (sampling_strategy == BurleySSSWrapper::SamplingStrategy::AlbedoMIS)
            EXPECT_LT(average_std_dev, 0.1334f) << " for albedo MIS sampling";
        else
            EXPECT_LT(average_std_dev, 0.1731f) << " for Karis' sampling approximation";
    }
}

} // NS OptiXRenderer

#endif // _OPTIXRENDERER_BSDFS_BURLEY_SSS_TEST_H_