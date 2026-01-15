// Test Bifrost media parameters and functionality.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _BIFROST_ASSETS_MEDIA_TEST_H_
#define _BIFROST_ASSETS_MEDIA_TEST_H_

#include <Bifrost/Assets/Media.h>

#include <Expects.h>

namespace Bifrost::Assets::Media {

struct MatchingTestParameters {
    MeasuredScatteringParameters measured_parameters;
    ArtisticScatteringParameters artistic_parameters;

    static MatchingTestParameters apple() { return { MeasuredScatteringParameters::apple(), ArtisticScatteringParameters::apple() }; }
    static MatchingTestParameters ketchup() { return { MeasuredScatteringParameters::ketchup(), ArtisticScatteringParameters::ketchup() }; }
    static MatchingTestParameters marble() { return { MeasuredScatteringParameters::marble(), ArtisticScatteringParameters::marble() }; }
    static MatchingTestParameters skin() { return { MeasuredScatteringParameters::skin1(), ArtisticScatteringParameters::skin1() }; }
    static MatchingTestParameters potato() { return { MeasuredScatteringParameters::potato(), ArtisticScatteringParameters::potato() }; }
};

GTEST_TEST(Assets_Media, convert_measured_scattering_parameters_to_artistic) {
    MatchingTestParameters matching_test_parameters[3] = { MatchingTestParameters::apple(), MatchingTestParameters::ketchup(), MatchingTestParameters::potato() };

    for (int p = 0; p < 3; ++p) {
        auto expected_artistic = matching_test_parameters[p].artistic_parameters;
        auto actual_artistic = ArtisticScatteringParameters::from_measured_parameters(matching_test_parameters[p].measured_parameters);

        EXPECT_RGB_EQ_EPS(expected_artistic.diffuse_albedo, actual_artistic.diffuse_albedo, 0.002f);
        EXPECT_RGB_EQ_EPS(expected_artistic.mean_free_path, actual_artistic.mean_free_path, 0.00001f);
    }
}

GTEST_TEST(Assets_Media, convert_artistic_scattering_parameters_to_measured) {
    MatchingTestParameters matching_test_parameters[2] = { MatchingTestParameters::apple(), MatchingTestParameters::potato() };

    for (int p = 0; p < 2; ++p) {
        auto expected_measured_parameters = matching_test_parameters[p].measured_parameters;
        auto actual_measured_parameters = MeasuredScatteringParameters::from_artistic_parameters(matching_test_parameters[p].artistic_parameters);

        // Test against the parameters that Chiang's conversion algorithm was fitted against.
        EXPECT_RGB_EQ_EPS(expected_measured_parameters.get_single_scattering_albedo(), actual_measured_parameters.get_single_scattering_albedo(), 0.2f);
        EXPECT_RGB_EQ_EPS(expected_measured_parameters.get_mean_free_path(), actual_measured_parameters.get_mean_free_path(), 0.00001f);
    }
}

} // NS Bifrost::Assets::Media

#endif // _BIFROST_ASSETS_MEDIA_TEST_H_
