// OptiXRenderer unit tests.
// ---------------------------------------------------------------------------
// Copyright (C) 2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#include <gtest/gtest.h>

#include <BSDFs/GGXTest.h>
#include <BSDFs/LambertTest.h>
#include <BSDFs/OrenNayarTest.h>

#include <LightSources/SphereLightTest.h>

#include <ShadingModels/DefaultShadingTest.h>

#include <EncodedNormalTest.h>

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
