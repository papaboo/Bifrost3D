// Test utils for Bifrost light sources.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _BIFROST_ASSETS_SHADING_LIGHTSOURCES_LIGHT_TEST_UTILS_H_
#define _BIFROST_ASSETS_SHADING_LIGHTSOURCES_LIGHT_TEST_UTILS_H_

#include <Assets/Shading//BSDFTestUtils.h>

#include <Bifrost/Assets/Shading/Utils.h>

namespace Bifrost::Assets::Shading::LightSources::LightTestUtils {

template <typename LightSource>
inline void light_consistency_test(LightSource light_source, Math::Vector3f lit_position, unsigned int sample_count) {
    for (unsigned int i = 0u; i < sample_count; ++i) {
        Math::Vector2f rng_sample = BSDFTestUtils::bsdf_rng_sample2f(i);
        LightSample sample = light_source.sample_radiance(lit_position, rng_sample);

        if (!sample.PDF.is_valid()) {
            EXPECT_RGB_EQ(Math::RGB::black(), sample.radiance) << light_source.to_string();
        } else if (sample.PDF.is_delta_dirac()) {
            EXPECT_GT(sample.radiance.r, 0.0f) << light_source.to_string();

            // Delta light can be sampled, but not evaluated
            auto actual_PDF = light_source.pdf(lit_position, sample.direction_to_light);
            EXPECT_TRUE(actual_PDF.is_delta_dirac()) << light_source.to_string();
            EXPECT_FALSE(actual_PDF.is_valid()) << light_source.to_string();

            LightResponse response = light_source.evaluate_with_PDF(lit_position, sample.direction_to_light);
            EXPECT_EQ(actual_PDF, response.PDF) << light_source.to_string();
            EXPECT_RGB_EQ(Math::RGB::black(), response.radiance) << light_source.to_string();

        } else if (sample.PDF.is_valid()) {
            EXPECT_GE(sample.radiance.r, 0.0f) << light_source.to_string();

            EXPECT_PDF_EQ_PCT(sample.PDF, light_source.pdf(lit_position, sample.direction_to_light), 0.00002f) << light_source.to_string();

            LightResponse response = light_source.evaluate_with_PDF(lit_position, sample.direction_to_light);
            EXPECT_RGB_EQ_PCT(sample.radiance, response.radiance, 0.00002f) << light_source.to_string();
            EXPECT_PDF_EQ_PCT(sample.PDF, response.PDF, 0.00002f) << light_source.to_string();
        }
    }
}

} // NS Bifrost::Assets::Shading::LightSources::LightTestUtils

#endif // _BIFROST_ASSETS_SHADING_LIGHTSOURCES_LIGHT_TEST_UTILS_H_