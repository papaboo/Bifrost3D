// Test image statistics.
// ---------------------------------------------------------------------------
// Copyright (C) 2015-2017, Bifrost. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _IMAGE_OPERATIONS_STATISTICS_TEST_H_
#define _IMAGE_OPERATIONS_STATISTICS_TEST_H_

#include <ImageOperations/Compare.h>
#include <../BifrostTests/Expects.h>

namespace ImageOperations {
namespace Compare {

RGB to_rgb(Vector3d v) { return RGB(float(v.x), float(v.y), float(v.z)); }

GTEST_TEST(ImageOperations_Statistics, Stats) {

    std::vector<int> v1s = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
    std::vector<int> v2s = { 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18 };

    Statistics stats;
    for (int i = 0; i <= 16; ++i)
        stats.add(RGB(float(v1s[i])), RGB(float(v2s[i])));

    float mean_1 = 8.0f;
    float mean_2 = 10.0f;
    EXPECT_RGB_EQ(RGB(mean_1), to_rgb(stats.reference_mean()));
    EXPECT_RGB_EQ(RGB(mean_2), to_rgb(stats.target_mean()));

    double variance_1 = 0.0f;
    for (int i = 0; i < v1s.size(); ++i)
        variance_1 += (v1s[i] - mean_1) * (v1s[i] - mean_1);
    variance_1 /= v1s.size();

    double variance_2 = 0.0f;
    for (int i = 0; i < v2s.size(); ++i)
        variance_2 += (v2s[i] - mean_2) * (v2s[i] - mean_2);
    variance_2 /= v2s.size();

    EXPECT_EQ(variance_1, variance_2);
    EXPECT_RGB_EQ(RGB(float(variance_1)), to_rgb(stats.reference_variance()));
    EXPECT_RGB_EQ(RGB(float(variance_2)), to_rgb(stats.target_variance()));

    double covariance = 0.0f;
    for (int i = 0; i < v1s.size(); ++i)
        covariance += (v1s[i] - mean_1) * (v2s[i] - mean_2);
    covariance /= v1s.size();

    EXPECT_RGB_EQ(RGB(float(covariance)), to_rgb(stats.covariance()));
}

GTEST_TEST(ImageOperations_Statistics, Weighting) {
    Statistics verbose_stats = {};
    verbose_stats.add(RGB(2), RGB(3));
    verbose_stats.add(RGB(5), RGB(4));
    verbose_stats.add(RGB(5), RGB(4));

    Statistics weighted_stats = {};
    weighted_stats.add(RGB(2), RGB(3));
    weighted_stats.add(RGB(5), RGB(4), 2);

    EXPECT_RGB_EQ(to_rgb(verbose_stats.reference_mean()), to_rgb(weighted_stats.reference_mean()));
    EXPECT_RGB_EQ(to_rgb(verbose_stats.reference_variance()), to_rgb(weighted_stats.reference_variance()));
    EXPECT_RGB_EQ(to_rgb(verbose_stats.target_mean()), to_rgb(weighted_stats.target_mean()));
    EXPECT_RGB_EQ(to_rgb(verbose_stats.target_variance()), to_rgb(weighted_stats.target_variance()));
    EXPECT_RGB_EQ(to_rgb(verbose_stats.covariance()), to_rgb(weighted_stats.covariance()));

    Statistics low_weighted_stats = {};
    low_weighted_stats.add(RGB(2), RGB(3), 0.5);
    low_weighted_stats.add(RGB(5), RGB(4), 1);

    EXPECT_RGB_EQ(to_rgb(verbose_stats.reference_mean()), to_rgb(low_weighted_stats.reference_mean()));
    EXPECT_RGB_EQ(to_rgb(verbose_stats.reference_variance()), to_rgb(low_weighted_stats.reference_variance()));
    EXPECT_RGB_EQ(to_rgb(verbose_stats.target_mean()), to_rgb(low_weighted_stats.target_mean()));
    EXPECT_RGB_EQ(to_rgb(verbose_stats.target_variance()), to_rgb(low_weighted_stats.target_variance()));
    EXPECT_RGB_EQ(to_rgb(verbose_stats.covariance()), to_rgb(low_weighted_stats.covariance()));
}

} // NS Compare
} // NS ImageOperations

#endif // _IMAGE_OPERATIONS_STATISTICS_TEST_H_