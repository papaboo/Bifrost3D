// Test asset flags in the OptiXRenderer.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _OPTIXRENDERER_ASSETS_FLAGS_TEST_H_
#define _OPTIXRENDERER_ASSETS_FLAGS_TEST_H_

#include <Bifrost/Assets/Material.h>

#include <OptiXRenderer/Types.h>

#include <gtest/gtest.h>

namespace OptiXRenderer {

GTEST_TEST(Assets, material_flags_has_same_layout_as_Bifrost) {
    EXPECT_EQ((int)Bifrost::Assets::MaterialFlag::ThinWalled, OptiXRenderer::Material::Flags::ThinWalled);
    EXPECT_EQ((int)Bifrost::Assets::MaterialFlag::Cutout, OptiXRenderer::Material::Flags::Cutout);
}

GTEST_TEST(Assets, shading_models_has_same_layout_as_Bifrost) {
    EXPECT_EQ((int)Bifrost::Assets::ShadingModel::Default, OptiXRenderer::Material::ShadingModel::Default);
    EXPECT_EQ((int)Bifrost::Assets::ShadingModel::Diffuse, OptiXRenderer::Material::ShadingModel::Diffuse);
    EXPECT_EQ((int)Bifrost::Assets::ShadingModel::Transmissive, OptiXRenderer::Material::ShadingModel::Transmissive);
    EXPECT_EQ((int)Bifrost::Assets::ShadingModel::Count, 3);
}

} // NS OptiXRenderer

#endif // _OPTIXRENDERER_ASSETS_FLAGS_TEST_H_