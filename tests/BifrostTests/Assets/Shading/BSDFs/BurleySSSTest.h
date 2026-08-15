// Test Bifrost's Burley BSSRDF.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _BIFROST_ASSETS_SHADING_BSDFS_BURLEY_SSS_TEST_H_
#define _BIFROST_ASSETS_SHADING_BSDFS_BURLEY_SSS_TEST_H_

#include <Assets/Shading/BSDFTestUtils.h>
#include <Expects.h>

#include <Bifrost/Assets/Shading/BSDFs/BurleySSS.h>
#include <Bifrost/Assets/Media.h>

#include <gtest/gtest.h>

namespace Bifrost::Assets::Shading::BSDFs {

class BurleySSSWrapper {
public:
    enum class SamplingStrategy { SampleMostScattering, AlbedoMIS, KarisApproximation };

    static constexpr SamplingStrategy sampling_strategies[3] = { SamplingStrategy::SampleMostScattering, SamplingStrategy::AlbedoMIS, SamplingStrategy::KarisApproximation };

    Shading::BSDFs::BurleySSS::Parameters m_parameters;
    SamplingStrategy m_sampling_strategy = SamplingStrategy::AlbedoMIS;

    BurleySSSWrapper(Math::RGB albedo, Math::Vector3f mean_free_path)
        : m_parameters(Shading::BSDFs::BurleySSS::Parameters::create(albedo, mean_free_path)) {}

    BurleySSSWrapper(Media::ArtisticScatteringParameters params)
        : m_parameters(Shading::BSDFs::BurleySSS::Parameters::create(params.diffuse_albedo, params.mean_free_path)) {}

    void set_sampling_strategy(SamplingStrategy sampling_strategy) {
        m_sampling_strategy = sampling_strategy;
    }

    Math::RGB evaluate(Math::Vector3f po, Math::Vector3f pi) const {
        return Shading::BSDFs::BurleySSS::evaluate(m_parameters, po, pi);
    }

    Math::MonteCarlo::PDF pdf(Math::Vector3f po, Math::Vector3f pi) const {
        if (m_sampling_strategy == SamplingStrategy::SampleMostScattering)
            return Shading::BSDFs::BurleySSS::SampleMostScattering::pdf(m_parameters, po, pi);
        else if (m_sampling_strategy == SamplingStrategy::KarisApproximation)
            return Shading::BSDFs::BurleySSS::ApproximateSampling::pdf(m_parameters, po, pi);
        else
            return Shading::BSDFs::BurleySSS::AlbedoMIS::pdf(m_parameters, po, pi);
    }

    BSDFResponse evaluate_with_PDF(Math::Vector3f po, Math::Vector3f pi) const {
        if (m_sampling_strategy == SamplingStrategy::SampleMostScattering)
            return Shading::BSDFs::BurleySSS::SampleMostScattering::evaluate_with_PDF(m_parameters, po, pi);
        else if (m_sampling_strategy == SamplingStrategy::KarisApproximation)
            return Shading::BSDFs::BurleySSS::ApproximateSampling::evaluate_with_PDF(m_parameters, po, pi);
        else
            return Shading::BSDFs::BurleySSS::AlbedoMIS::evaluate_with_PDF(m_parameters, po, pi);
    }

    SeparableBSSRDFPositionSample sample(Math::Vector3f po, Math::Vector3f random_sample) const {
        if (m_sampling_strategy == SamplingStrategy::SampleMostScattering)
            return Shading::BSDFs::BurleySSS::SampleMostScattering::sample(m_parameters, po, random_sample);
        else if (m_sampling_strategy == SamplingStrategy::KarisApproximation)
            return Shading::BSDFs::BurleySSS::ApproximateSampling::sample(m_parameters, po, random_sample);
        else
            return Shading::BSDFs::BurleySSS::AlbedoMIS::sample(m_parameters, po, random_sample);
    }

    std::string to_string() const {
        std::ostringstream out;
        out << "Burley: diffuse albedo: " << m_parameters.diffuse_albedo << ", diffuse mean free path: " << m_parameters.diffuse_mean_free_path;
        return out.str();
    }
};

template <typename BSSRDFModel>
BSDFTestUtils::RhoResult directional_hemispherical_reflectance_function(BSSRDFModel bssrdf_model, unsigned int sample_count) {
    using namespace Bifrost::Math;

    // Return an invalid result if more samples are requested than can be produced.
    if (BSDFTestUtils::g_bsdf_rng.m_max_sample_capacity < sample_count)
        return BSDFTestUtils::RhoResult::invalid();

    const Vector3f po = { 1.0f, -2.0f, 4.0f };

    Statistics<double> reflectance_statistics[3] = { Statistics<double>(), Statistics<double>(), Statistics<double>() };
    for (unsigned int i = 0u; i < sample_count; ++i) {
        Vector3f rng = BSDFTestUtils::bsdf_rng_sample3f(i, sample_count);
        auto sample = bssrdf_model.sample(po, rng);

        // All sampled PDFs are valid, so the validity check is avoided in order to not bias the result by ignoring samples with low PDF.
        RGB reflectance = sample.reflectance / sample.PDF.value();

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

    return { mean_reflectance, reflectance_std_dev, Vector3f(0, 0, 1) };
}

GTEST_TEST(Assets_Shading_BSDFs_BurleySSS, power_conservation) {
    Math::RGB white = Math::RGB::white();
    for (float mean_free_path : { 0.2f, 1.0f, 5.0f, 15.0f }) {
        BurleySSSWrapper bssrdf = BurleySSSWrapper(white, Math::Vector3f(mean_free_path));
        auto reflectance = directional_hemispherical_reflectance_function(bssrdf, 4096).reflectance;
        EXPECT_RGB_EQ_EPS(reflectance, white, 0.00045f) << bssrdf.to_string();
    }
}

GTEST_TEST(Assets_Shading_BSDFs_BurleySSS, reciprocity) {
    auto bssrdf = BurleySSSWrapper(Media::ArtisticScatteringParameters::ketchup());
    Math::Vector3f po = { 0.0f, 0.0f, 0.0f };
    for (Math::Vector3f pi : { Math::Vector3f(0.1f, -0.2f, 0.3f), Math::Vector3f(0.1f, -0.2f, -0.3f), Math::Vector3f(-0.1f, 1.2f, 0.6f)}) {
        Math::RGB scattering0 = bssrdf.evaluate(po, pi);
        Math::RGB scattering1 = bssrdf.evaluate(pi, po);
        EXPECT_RGB_EQ(scattering0, scattering1);
    }
}

GTEST_TEST(Assets_Shading_BSDFs_BurleySSS, function_consistency) {
    const int sample_count = 16;

    Math::RGB albedo = { 0.3f, 0.9f, 0.5f };
    Math::Vector3f mean_free_path = { 1, 3, 7 };
    Math::Vector3f po = { 1.0f, -2.0f, 4.0f };
    BurleySSSWrapper burley = BurleySSSWrapper(albedo, mean_free_path);
    for (auto sampling_strategy : BurleySSSWrapper::sampling_strategies)
        burley.set_sampling_strategy(sampling_strategy);
        for (unsigned int i = 0u; i < sample_count; ++i) {
            Math::Vector3f rng_sample = BSDFTestUtils::bsdf_rng_sample3f(i, sample_count);
            auto sample = burley.sample(po, rng_sample);

            if (sample.PDF.is_valid()) {
                EXPECT_GE(sample.reflectance.r, 0.0f) << burley.to_string();

                EXPECT_PDF_EQ_PCT(sample.PDF, burley.pdf(po, sample.position), 0.00002f) << burley.to_string();
                EXPECT_RGB_EQ_PCT(sample.reflectance, burley.evaluate(po, sample.position), 0.00002f) << burley.to_string();

                BSDFResponse response = burley.evaluate_with_PDF(po, sample.position);
                EXPECT_RGB_EQ_PCT(sample.reflectance, response.reflectance, 0.00002f) << burley.to_string();
                EXPECT_PDF_EQ_PCT(sample.PDF, response.PDF, 0.00002f) << burley.to_string();
            }
        }
}

GTEST_TEST(Assets_Shading_BSDFs_BurleySSS, sampling_correctness) {
    const int sample_count = 1024;

    for (auto sampling_strategy : BurleySSSWrapper::sampling_strategies) {

        float summed_std_dev = 0.0f;
        for (auto sss_params : { Media::ArtisticScatteringParameters::ketchup(), Media::ArtisticScatteringParameters::skin1(), Media::ArtisticScatteringParameters::marble() }) {
            auto bssrdf = BurleySSSWrapper(sss_params);
            bssrdf.set_sampling_strategy(sampling_strategy);

            auto rho = directional_hemispherical_reflectance_function(bssrdf, 4096);

            // Reflectance converges to albedo
            bool is_approximated_sampling = sampling_strategy == BurleySSSWrapper::SamplingStrategy::KarisApproximation;
            float epsilon = is_approximated_sampling ? 0.02f : 0.01f;
            EXPECT_RGB_EQ_PCT(rho.reflectance, sss_params.diffuse_albedo, epsilon);

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

} // NS Bifrost::Assets::Shading::BSDFs

#endif // _BIFROST_ASSETS_SHADING_BSDFS_BURLEY_SSS_TEST_H_