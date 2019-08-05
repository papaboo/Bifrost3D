// OptiXRenderer unit tests.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#include <gtest/gtest.h>

#include <BSDFs/BurleyTest.h>
#include <BSDFs/GGXTest.h>
#include <BSDFs/LambertTest.h>
#include <BSDFs/OrenNayarTest.h>

#include <LightSources/SphereLightTest.h>

#include <ShadingModels/DefaultShadingTest.h>

#include <MiscTest.h>

// NOTE
// To run a subset of test cases use a filter, e.g '--gtest_filter=*BSDFs*'.
int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
