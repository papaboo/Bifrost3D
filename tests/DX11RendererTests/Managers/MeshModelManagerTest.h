// DX11Renderer mesh model manager test.
// ------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _DX11RENDERER_MESH_MODEL_MANAGER_TEST_H_
#define _DX11RENDERER_MESH_MODEL_MANAGER_TEST_H_

#include <gtest/gtest.h>
#include <Utils.h>

#include <DX11Renderer/Managers/MeshModelManager.h>

#include <Bifrost/Assets/Material.h>
#include <Bifrost/Assets/Mesh.h>
#include <Bifrost/Assets/MeshModel.h>
#include <Bifrost/Scene/SceneNode.h>

namespace DX11Renderer::Managers {

class MeshModelManagerFixture : public ::testing::Test {
protected:
    // Per-test set-up and tear-down logic.
    virtual void SetUp() {
        Bifrost::Assets::Materials::allocate(8u);
        Bifrost::Assets::Meshes::allocate(8u);
        Bifrost::Assets::MeshModels::allocate(8u);
        Bifrost::Scene::SceneNodes::allocate(8u);
    }
    virtual void TearDown() {
        Bifrost::Assets::Materials::deallocate();
        Bifrost::Assets::Meshes::deallocate();
        Bifrost::Assets::MeshModels::deallocate();
        Bifrost::Scene::SceneNodes::deallocate();
    }
};

inline void handle_and_reset_updates(MeshModelManager& model_manager) {
    model_manager.handle_updates();
    Bifrost::Assets::Materials::reset_change_notifications();
    Bifrost::Assets::Meshes::reset_change_notifications();
    Bifrost::Assets::MeshModels::reset_change_notifications();
    Bifrost::Scene::SceneNodes::reset_change_notifications();
}

inline Bifrost::Assets::Material create_plastic_material() {
    return Bifrost::Assets::Material::create_dielectric("Plastic", Bifrost::Math::RGB(0.1f, 0.1f, 0.4f), 0.1f);
}

inline Bifrost::Assets::Material create_glass_material() {
    return Bifrost::Assets::Material::create_transmissive("Glass", Bifrost::Math::RGB(0.95f), 0.0f);
}

inline Bifrost::Assets::MeshModel create_small_mesh_model() {
    Bifrost::Scene::SceneNode node = Bifrost::Scene::SceneNode("node");
    Bifrost::Assets::Mesh mesh = create_triangle("Geometry");
    Bifrost::Assets::Material material = create_plastic_material();
    return Bifrost::Assets::MeshModel(node, mesh, material);
}

TEST_F(MeshModelManagerFixture, invalid_model_at_first_index) {
    auto mesh_model_manager = MeshModelManager();

    Dx11Model model = mesh_model_manager.get_model(0);
    EXPECT_EQ(0, model.transform_ID.get_index());
    EXPECT_EQ(0, model.material_ID.get_index());
    EXPECT_EQ(0, model.mesh_ID.get_index());
    EXPECT_EQ(0, model.model_ID.get_index());
}

TEST_F(MeshModelManagerFixture, create_model) {
    auto mesh_model_manager = MeshModelManager();

    auto model = create_small_mesh_model();

    handle_and_reset_updates(mesh_model_manager);

    // Assert that there is one mesh models in the manager.
    EXPECT_EQ(mesh_model_manager.begin() + 1, mesh_model_manager.end());

    Dx11Model dx11_model = mesh_model_manager.get_model(model.get_ID());
    EXPECT_EQ(model.get_scene_node(), dx11_model.transform_ID);
    EXPECT_EQ(model.get_material(), dx11_model.material_ID);
    EXPECT_EQ(model.get_mesh(), dx11_model.mesh_ID);
    EXPECT_EQ(model, dx11_model.model_ID);
}

TEST_F(MeshModelManagerFixture, created_and_destroyed_model_is_ignored) {
    auto mesh_model_manager = MeshModelManager();

    auto mesh_model = create_small_mesh_model();
    unsigned int mesh_model_index = mesh_model.get_ID();
    mesh_model.destroy();

    handle_and_reset_updates(mesh_model_manager);

    // Assert that there are no mesh models in the manager.
    EXPECT_EQ(mesh_model_manager.begin(), mesh_model_manager.end());
}

TEST_F(MeshModelManagerFixture, destroyed_model_is_cleared) {
    auto mesh_model_manager = MeshModelManager();

    auto model = create_small_mesh_model();
    unsigned int model_index = model.get_ID();

    handle_and_reset_updates(mesh_model_manager);

    // Test that the mesh was created
    EXPECT_EQ(mesh_model_manager.begin() + 1, mesh_model_manager.end());
    Dx11Model dx11_model = mesh_model_manager.get_model(model.get_ID());
    EXPECT_EQ(model.get_material(), dx11_model.material_ID);

    model.destroy();
    handle_and_reset_updates(mesh_model_manager);

    // Test that the mesh is cleared
    EXPECT_EQ(mesh_model_manager.begin(), mesh_model_manager.end());
    dx11_model = mesh_model_manager.get_model(model.get_ID());
    EXPECT_EQ(0, dx11_model.transform_ID.get_index());
    EXPECT_EQ(0, dx11_model.material_ID.get_index());
    EXPECT_EQ(0, dx11_model.mesh_ID.get_index());
    EXPECT_EQ(0, dx11_model.model_ID.get_index());
}

TEST_F(MeshModelManagerFixture, handle_material_changed) {
    auto mesh_model_manager = MeshModelManager();

    auto model = create_small_mesh_model();
    auto initial_material = model.get_material();

    handle_and_reset_updates(mesh_model_manager);

    // Test that the mesh was created with an opaque material
    EXPECT_EQ(mesh_model_manager.cbegin_opaque_models() + 1, mesh_model_manager.cbegin_transparent_models()); // One opaque material
    EXPECT_EQ(mesh_model_manager.cbegin_transparent_models(), mesh_model_manager.end()); // No transparent materials
    Dx11Model dx11_model = mesh_model_manager.get_model(model.get_ID());
    EXPECT_EQ(initial_material, dx11_model.material_ID);

    // Update with a transmissive glass material
    auto glass_material = create_glass_material();
    model.set_material(glass_material);
    handle_and_reset_updates(mesh_model_manager);

    // Test that the material is updated
    EXPECT_EQ(mesh_model_manager.cbegin_opaque_models(), mesh_model_manager.cbegin_transparent_models()); // No opaque materials
    EXPECT_EQ(mesh_model_manager.cbegin_transparent_models() + 1, mesh_model_manager.end()); // One transparent material
    dx11_model = mesh_model_manager.get_model(model.get_ID());
    EXPECT_EQ(glass_material, dx11_model.material_ID);
}

} // NS DX11Renderer

#endif // _DX11RENDERER_MESH_MODEL_MANAGER_TEST_H_