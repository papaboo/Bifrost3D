// Test OptiXRenderer's rough glass shading model.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _OPTIXRENDERER_SHADING_MODELS_UTILS_TEST_H_
#define _OPTIXRENDERER_SHADING_MODELS_UTILS_TEST_H_

#include <Utils.h>

#include <Bifrost/Assets/Shading/Fittings.h>

#include <OptiXRenderer/Shading/ShadingModels/Utils.h>

#include <gtest/gtest.h>

namespace OptiXRenderer {

GTEST_TEST(ShadingModelUtils, GGX_rho_texture_size) {
    int expected_angle_sample_count = Bifrost::Assets::Shading::Rho::GGX_angle_sample_count;
    int actual_angle_sample_count = Shading::ShadingModels::SpecularRho::angle_sample_count;
    EXPECT_EQ(actual_angle_sample_count, expected_angle_sample_count);

    int expected_roughness_sample_count = Bifrost::Assets::Shading::Rho::GGX_roughness_sample_count;
    int actual_roughness_sample_count = Shading::ShadingModels::SpecularRho::roughness_sample_count;
    EXPECT_EQ(actual_roughness_sample_count, expected_roughness_sample_count);
}

} // NS OptiXRenderer

#endif // _OPTIXRENDERER_SHADING_MODELS_UTILS_TEST_H_