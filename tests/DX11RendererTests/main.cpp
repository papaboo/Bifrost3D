// DX11Renderer unit tests.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#include <gtest/gtest.h>

#include <BloomTest.h>
#include <ExposureHistogramTest.h>
#include <LogAverageLuminanceTest.h>
#include <PrefixSumTest.h>

// NOTE
// To run a subset of test cases use a filter, e.g '--gtest_filter=*ExposureHistogram*'.
int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
