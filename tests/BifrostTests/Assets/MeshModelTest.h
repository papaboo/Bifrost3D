// Test Bifrost mesh models.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _BIFROST_ASSETS_MESH_MODEL_TEST_H_
#define _BIFROST_ASSETS_MESH_MODEL_TEST_H_

#include <Bifrost/Assets/MeshModel.h>

#include <gtest/gtest.h>

namespace Bifrost {
namespace Assets {

class Assets_MeshModels : public ::testing::Test {
protected:
    // Per-test set-up and tear-down logic.
    virtual void SetUp() {
        MeshModels::allocate(8u);
        Scene::SceneNodes::allocate(1u);
        Meshes::allocate(1u);
        Materials::allocate(1u);
    }
    virtual void TearDown() {
        MeshModels::deallocate();
        Scene::SceneNodes::deallocate();
        Meshes::deallocate();
        Materials::deallocate();
    }
};

TEST_F(Assets_MeshModels, resizing) {
    // Test that capacity can be increased.
    unsigned int largerCapacity = MeshModels::capacity() + 4u;
    MeshModels::reserve(largerCapacity);
    EXPECT_GE(MeshModels::capacity(), largerCapacity);

    // Test that capacity won't be decreased.
    MeshModels::reserve(5u);
    EXPECT_GE(MeshModels::capacity(), largerCapacity);

    MeshModels::deallocate();
    EXPECT_LT(MeshModels::capacity(), largerCapacity);
}

TEST_F(Assets_MeshModels, sentinel_mesh) {
    MeshModel sentinel = MeshModelID();

    EXPECT_FALSE(sentinel.exists());
    EXPECT_EQ(sentinel.get_scene_node(), Scene::SceneNode::invalid());
    EXPECT_EQ(sentinel.get_mesh(), Mesh::invalid());
}

TEST_F(Assets_MeshModels, create) {
    Scene::SceneNode node = Scene::SceneNode("TestNode");
    Mesh mesh = Mesh("TestMesh", 32u, 16u);
    Materials::Data data = {};
    Material material = Material("TestMaterial", data);

    MeshModel model = MeshModel(node, mesh, material);

    EXPECT_TRUE(model.exists());
    EXPECT_EQ(model.get_scene_node(), node);
    EXPECT_EQ(model.get_mesh(), mesh);
    EXPECT_EQ(model.get_material(), material);

    // Test model created notification.
    Core::Iterable<MeshModels::ChangedIterator> changed_models = MeshModels::get_changed_models();
    EXPECT_EQ(changed_models.end() - changed_models.begin(), 1);
    EXPECT_EQ(model, *changed_models.begin());
    EXPECT_EQ(model.get_changes(), MeshModels::Change::Created);
}

TEST_F(Assets_MeshModels, destroy) {
    Scene::SceneNode node = Scene::SceneNode("TestNode");
    Mesh mesh = Mesh("TestMesh", 32u, 16u);
    Materials::Data data = {};
    Material material = Material("TestMaterial", data);

    MeshModel model = MeshModel(node, mesh, material);
    EXPECT_TRUE(model.exists());

    MeshModels::reset_change_notifications();

    model.destroy();
    EXPECT_FALSE(model.exists());

    // Test model destroyed notification.
    Core::Iterable<MeshModels::ChangedIterator> changed_models = MeshModels::get_changed_models();
    EXPECT_EQ(changed_models.end() - changed_models.begin(), 1);
    EXPECT_EQ(model, *changed_models.begin());
    EXPECT_EQ(model.get_changes(), MeshModels::Change::Destroyed);
}

TEST_F(Assets_MeshModels, create_and_destroy_notifications) {
    Scene::SceneNode node = Scene::SceneNode("TestNode");
    Mesh mesh = Mesh("TestMesh", 32u, 16u);
    Materials::Data data = {};
    Material material = Material("TestMaterial", data);

    MeshModel model0 = MeshModel(node, mesh, material);
    MeshModel model1 = MeshModel(node, mesh, material);
    EXPECT_TRUE(model0.exists());
    EXPECT_TRUE(model1.exists());

    { // Test model create notifications.
        Core::Iterable<MeshModels::ChangedIterator> changed_models = MeshModels::get_changed_models();
        EXPECT_EQ(changed_models.end() - changed_models.begin(), 2);

        bool model0_created = false;
        bool model1_created = false;
        for (const MeshModel model : changed_models) {
            if (model == model0 && model.get_changes() == MeshModels::Change::Created)
                model0_created = true;
            if (model == model1 && model.get_changes() == MeshModels::Change::Created)
                model1_created = true;
        }

        EXPECT_TRUE(model0_created);
        EXPECT_TRUE(model1_created);
    }

    MeshModels::reset_change_notifications();

    { // Test destroy.
        model0.destroy();
        EXPECT_FALSE(model0.exists());

        Core::Iterable<MeshModels::ChangedIterator> changed_models = MeshModels::get_changed_models();
        EXPECT_EQ(changed_models.end() - changed_models.begin(), 1);

        bool model0_destroyed = false;
        bool other_change = false;
        for (const MeshModel model : changed_models) {
            if (model == model0 && model.get_changes() == MeshModels::Change::Destroyed)
                model0_destroyed = true;
            else
                other_change = true;
        }

        EXPECT_TRUE(model0_destroyed);
        EXPECT_FALSE(other_change);
    }

    MeshModels::reset_change_notifications();

    { // Test that destroyed model cannot be destroyed again.
        EXPECT_FALSE(model0.exists());

        model0.destroy();
        EXPECT_FALSE(model0.exists());
        EXPECT_TRUE(MeshModels::get_changed_models().is_empty());
    }
}

} // NS Assets
} // NS Bifrost

#endif // _BIFROST_ASSETS_MESH_MODEL_TEST_H_
