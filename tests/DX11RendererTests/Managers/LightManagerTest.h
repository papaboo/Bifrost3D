// DX11Renderer light manager test.
// ------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _DX11RENDERER_MESH_MANAGER_LIGHT_H_
#define _DX11RENDERER_MESH_MANAGER_LIGHT_H_

#include <gtest/gtest.h>
#include <Utils.h>

#include <DX11Renderer/Managers/LightManager.h>

#include <Bifrost/Scene/LightSource.h>

namespace DX11Renderer::Managers {

class LightManagerFixture : public ::testing::Test {
protected:
    ODevice1 device;
    ODeviceContext1 context;

    // Per-test set-up and tear-down logic.
    virtual void SetUp() {
        Bifrost::Scene::SceneNodes::allocate(8u);
        Bifrost::Scene::LightSources::allocate(8u);
        device = create_test_device();
        context = get_immidiate_context1(device);
    }
    virtual void TearDown() {
        Bifrost::Scene::LightSources::deallocate();
        Bifrost::Scene::SceneNodes::deallocate();
    }
};

TEST_F(LightManagerFixture, created_light_is_added) {
    LightManager light_manager(device);

    auto node = Bifrost::Scene::SceneNode("Dummy");
    auto light = Bifrost::Scene::SphereLight(node, Bifrost::Math::RGB(10), 1);

    light_manager.handle_updates(context);

    // Test that the light was created
    EXPECT_EQ(1, light_manager.active_light_count());
}

TEST_F(LightManagerFixture, destroyed_light_is_cleared) {
    LightManager light_manager(device);

    auto node = Bifrost::Scene::SceneNode("Dummy");
    auto light = Bifrost::Scene::SphereLight(node, Bifrost::Math::RGB(10), 1);

    light_manager.handle_updates(context);

    // Test that the light was created
    EXPECT_EQ(1, light_manager.active_light_count());

    light.destroy();
    node.destroy();

    light_manager.handle_updates(context);

    // Test that the light is cleared
    EXPECT_EQ(0, light_manager.active_light_count());
}

TEST_F(LightManagerFixture, created_and_destroyed_light_is_ignored) {
    LightManager light_manager(device);

    auto node = Bifrost::Scene::SceneNode("Dummy");
    auto light = Bifrost::Scene::SphereLight(node, Bifrost::Math::RGB(10), 1);
    light.destroy();
    node.destroy();

    light_manager.handle_updates(context);

    EXPECT_EQ(0, light_manager.active_light_count());
}

} // NS DX11Renderer::Managers

#endif // _DX11RENDERER_MESH_MANAGER_LIGHT_H_