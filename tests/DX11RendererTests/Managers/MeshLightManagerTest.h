// DX11Renderer mesh light manager test.
// ------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _DX11RENDERER_MESH_LIGHT_MANAGER_TEST_H_
#define _DX11RENDERER_MESH_LIGHT_MANAGER_TEST_H_

#include <gtest/gtest.h>
#include <Utils.h>

#include <DX11Renderer/Managers/MeshLightManager.h>

#include <Bifrost/Assets/Material.h>
#include <Bifrost/Assets/Mesh.h>
#include <Bifrost/Assets/MeshModel.h>
#include <Bifrost/Scene/SceneNode.h>

namespace DX11Renderer::Managers {

class MeshLightManagerFixture : public ::testing::Test {
protected:
    ODevice1 device;

    // Per-test set-up and tear-down logic.
    virtual void SetUp() {
        Bifrost::Assets::Materials::allocate(8u);
        Bifrost::Assets::Meshes::allocate(8u);
        Bifrost::Assets::MeshModels::allocate(8u);
        Bifrost::Scene::SceneNodes::allocate(8u);

        device = create_test_device();
    }
    virtual void TearDown() {
        Bifrost::Assets::Materials::deallocate();
        Bifrost::Assets::Meshes::deallocate();
        Bifrost::Assets::MeshModels::deallocate();
        Bifrost::Scene::SceneNodes::deallocate();
    }

    inline void handle_and_reset_updates(MeshLightManager& light_manager) {
        light_manager.handle_updates(device);
        Bifrost::Assets::Materials::reset_change_notifications();
        Bifrost::Assets::Meshes::reset_change_notifications();
        Bifrost::Assets::MeshModels::reset_change_notifications();
        Bifrost::Scene::SceneNodes::reset_change_notifications();
    }

    inline Bifrost::Assets::Material create_non_emissive_material() {
        return Bifrost::Assets::Material::create_dielectric("Plastic", Bifrost::Math::RGB(0.1f, 0.1f, 0.4f), 0.1f);
    }

    inline Bifrost::Assets::Material create_emissive_material() {
        return Bifrost::Assets::Material::create_emissive("Emissive", Bifrost::Math::RGB(1.00f));
    }

    inline Bifrost::Assets::MeshModel create_mesh_model(Bifrost::Assets::Material material) {
        Bifrost::Scene::SceneNode node = Bifrost::Scene::SceneNode("node");
        Bifrost::Assets::Mesh mesh = create_triangle("Geometry");
        return Bifrost::Assets::MeshModel(node, mesh, material);
    }

    inline Bifrost::Assets::MeshModel create_non_emissive_model() { return create_mesh_model(create_non_emissive_material()); }
    inline Bifrost::Assets::MeshModel create_emissive_model() { return create_mesh_model(create_emissive_material()); }
};

TEST_F(MeshLightManagerFixture, create_emissive_model) {
    auto mesh_light_manager = MeshLightManager();
    auto model = create_emissive_model();
    handle_and_reset_updates(mesh_light_manager);

    // Assert that there is one emissive triangle in the manager.
    EXPECT_EQ(1u, mesh_light_manager.get_emissive_triangle_count());
    EXPECT_TRUE(mesh_light_manager.has_mesh_light(model.get_ID()));
}

TEST_F(MeshLightManagerFixture, non_emissive_model_yields_no_emissive_lights) {
    auto mesh_light_manager = MeshLightManager();
    auto model = create_non_emissive_model();
    handle_and_reset_updates(mesh_light_manager);

    // Assert that there is no emissive meshes in the manager.
    EXPECT_EQ(0u, mesh_light_manager.get_emissive_triangle_count());
    EXPECT_FALSE(mesh_light_manager.has_mesh_light(model.get_ID()));
}

TEST_F(MeshLightManagerFixture, created_and_destroyed_model_is_ignored) {
    auto mesh_light_manager = MeshLightManager();
    auto model = create_emissive_model();
    model.destroy();
    handle_and_reset_updates(mesh_light_manager);

    // Assert that there is no emissive triangle in the manager.
    EXPECT_EQ(0u, mesh_light_manager.get_emissive_triangle_count());
    EXPECT_FALSE(mesh_light_manager.has_mesh_light(model.get_ID()));
}

TEST_F(MeshLightManagerFixture, destroyed_model_is_cleared) {
    auto mesh_light_manager = MeshLightManager();
    auto model = create_emissive_model();
    handle_and_reset_updates(mesh_light_manager);

    // Assert that there is one emissive triangle in the manager.
    EXPECT_EQ(1u, mesh_light_manager.get_emissive_triangle_count());
    EXPECT_TRUE(mesh_light_manager.has_mesh_light(model.get_ID()));

    model.destroy();
    handle_and_reset_updates(mesh_light_manager);

    // Assert that there is no emissive triangle in the manager.
    EXPECT_EQ(0u, mesh_light_manager.get_emissive_triangle_count());
    EXPECT_FALSE(mesh_light_manager.has_mesh_light(model.get_ID()));
}

TEST_F(MeshLightManagerFixture, change_to_non_emissive_material_removes_light) {
    auto mesh_light_manager = MeshLightManager();
    auto model = create_emissive_model();
    handle_and_reset_updates(mesh_light_manager);

    // Assert that there is one emissive triangle in the manager.
    EXPECT_EQ(1u, mesh_light_manager.get_emissive_triangle_count());
    EXPECT_TRUE(mesh_light_manager.has_mesh_light(model.get_ID()));

    auto non_emissive_material = create_non_emissive_material();
    model.set_material(non_emissive_material);
    handle_and_reset_updates(mesh_light_manager);

    // Assert that there is no emissive triangle in the manager.
    EXPECT_EQ(0u, mesh_light_manager.get_emissive_triangle_count());
    EXPECT_FALSE(mesh_light_manager.has_mesh_light(model.get_ID()));
}

TEST_F(MeshLightManagerFixture, material_destruction_removes_light) {
    auto mesh_light_manager = MeshLightManager();
    auto model = create_emissive_model();
    handle_and_reset_updates(mesh_light_manager);

    // Assert that there is one emissive triangle in the manager.
    EXPECT_EQ(1u, mesh_light_manager.get_emissive_triangle_count());
    EXPECT_TRUE(mesh_light_manager.has_mesh_light(model.get_ID()));

    model.get_material().destroy();
    handle_and_reset_updates(mesh_light_manager);

    // Assert that there is no emissive triangle in the manager.
    EXPECT_EQ(0u, mesh_light_manager.get_emissive_triangle_count());
    EXPECT_FALSE(mesh_light_manager.has_mesh_light(model.get_ID()));
}

TEST_F(MeshLightManagerFixture, set_emission_on_material_adds_light) {
    auto mesh_light_manager = MeshLightManager();
    auto model = create_non_emissive_model();
    handle_and_reset_updates(mesh_light_manager);

    // Assert that there is no emissive models in the manager.
    EXPECT_EQ(0u, mesh_light_manager.get_emissive_triangle_count());

    model.get_material().set_emission(Bifrost::Math::RGB(1.0f));
    handle_and_reset_updates(mesh_light_manager);

    // Assert that there is one emissive triangle in the manager.
    EXPECT_EQ(1u, mesh_light_manager.get_emissive_triangle_count());
    EXPECT_TRUE(mesh_light_manager.has_mesh_light(model.get_ID()));
}

TEST_F(MeshLightManagerFixture, zero_emission_on_material_removes_light) {
    auto mesh_light_manager = MeshLightManager();
    auto model = create_emissive_model();
    handle_and_reset_updates(mesh_light_manager);

    // Assert that there is one emissive triangle in the manager.
    EXPECT_EQ(1u, mesh_light_manager.get_emissive_triangle_count());

    model.get_material().set_emission(Bifrost::Math::RGB(0.0f));
    handle_and_reset_updates(mesh_light_manager);

    // Assert that there is no emissive triangle in the manager.
    EXPECT_EQ(0u, mesh_light_manager.get_emissive_triangle_count());
    EXPECT_FALSE(mesh_light_manager.has_mesh_light(model.get_ID()));
}

TEST_F(MeshLightManagerFixture, mesh_destruction_removes_light) {
    auto mesh_light_manager = MeshLightManager();
    auto model = create_emissive_model();
    handle_and_reset_updates(mesh_light_manager);

    // Assert that there is one emissive triangle in the manager.
    EXPECT_EQ(1u, mesh_light_manager.get_emissive_triangle_count());
    EXPECT_TRUE(mesh_light_manager.has_mesh_light(model.get_ID()));

    model.get_mesh().destroy();
    handle_and_reset_updates(mesh_light_manager);

    // Assert that there is no emissive triangle in the manager.
    EXPECT_EQ(0u, mesh_light_manager.get_emissive_triangle_count());
    EXPECT_FALSE(mesh_light_manager.has_mesh_light(model.get_ID()));
}

TEST_F(MeshLightManagerFixture, model_with_emissive_vertices_and_zero_material_emission_not_added) {
    Bifrost::Scene::SceneNode node = Bifrost::Scene::SceneNode("node");
    Bifrost::Assets::Mesh mesh = create_triangle("Geometry");
    auto non_emissive_material = create_non_emissive_material();
    auto model = Bifrost::Assets::MeshModel(node, mesh, non_emissive_material);

    auto mesh_light_manager = MeshLightManager();
    handle_and_reset_updates(mesh_light_manager);

    // Assert that there is no emissive triangle in the manager.
    EXPECT_EQ(0u, mesh_light_manager.get_emissive_triangle_count());
    EXPECT_FALSE(mesh_light_manager.has_mesh_light(model.get_ID()));
}

} // NS DX11Renderer

#endif // _DX11RENDERER_MESH_LIGHT_MANAGER_TEST_H_