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
    MeshModelID sentinel_ID = MeshModelID::invalid_UID();

    EXPECT_FALSE(MeshModels::has(sentinel_ID));
    EXPECT_EQ(MeshModels::get_scene_node_ID(sentinel_ID), Scene::SceneNodeID::invalid_UID());
    EXPECT_EQ(MeshModels::get_mesh_ID(sentinel_ID), MeshID::invalid_UID());
}

TEST_F(Assets_MeshModels, create) {
    Scene::SceneNodeID node_ID = Scene::SceneNodes::create("TestNode");
    MeshID mesh_ID = Meshes::create("TestMesh", 32u, 16u);
    Materials::Data data = {};
    MaterialID material_ID = Materials::create("TestMaterial", data);

    MeshModelID model_ID = MeshModels::create(node_ID, mesh_ID, material_ID);

    EXPECT_TRUE(MeshModels::has(model_ID));
    EXPECT_EQ(MeshModels::get_scene_node_ID(model_ID), node_ID);
    EXPECT_EQ(MeshModels::get_mesh_ID(model_ID), mesh_ID);
    EXPECT_EQ(MeshModels::get_material_ID(model_ID), material_ID);

    // Test model created notification.
    Core::Iterable<MeshModels::ChangedIterator> changed_models = MeshModels::get_changed_models();
    EXPECT_EQ(changed_models.end() - changed_models.begin(), 1);
    EXPECT_EQ(*changed_models.begin(), model_ID);
    EXPECT_EQ(MeshModels::get_changes(model_ID), MeshModels::Change::Created);
}

TEST_F(Assets_MeshModels, destroy) {
    Scene::SceneNodeID node_ID = Scene::SceneNodes::create("TestNode");
    MeshID mesh_ID = Meshes::create("TestMesh", 32u, 16u);
    Materials::Data data = {};
    MaterialID material_ID = Materials::create("TestMaterial", data);

    MeshModelID model_ID = MeshModels::create(node_ID, mesh_ID, material_ID);
    EXPECT_TRUE(MeshModels::has(model_ID));

    MeshModels::reset_change_notifications();

    MeshModels::destroy(model_ID);
    EXPECT_FALSE(MeshModels::has(model_ID));

    // Test model destroyed notification.
    Core::Iterable<MeshModels::ChangedIterator> changed_models = MeshModels::get_changed_models();
    EXPECT_EQ(changed_models.end() - changed_models.begin(), 1);
    EXPECT_EQ(*changed_models.begin(), model_ID);
    EXPECT_EQ(MeshModels::get_changes(model_ID), MeshModels::Change::Destroyed);
}

TEST_F(Assets_MeshModels, create_and_destroy_notifications) {
    Scene::SceneNodeID node_ID = Scene::SceneNodes::create("TestNode");
    MeshID mesh_ID = Meshes::create("TestMesh", 32u, 16u);
    Materials::Data data = {};
    MaterialID material_ID = Materials::create("TestMaterial", data);

    MeshModelID model_ID0 = MeshModels::create(node_ID, mesh_ID, material_ID);
    MeshModelID model_ID1 = MeshModels::create(node_ID, mesh_ID, material_ID);
    EXPECT_TRUE(MeshModels::has(model_ID0));
    EXPECT_TRUE(MeshModels::has(model_ID1));

    { // Test model create notifications.
        Core::Iterable<MeshModels::ChangedIterator> changed_models = MeshModels::get_changed_models();
        EXPECT_EQ(changed_models.end() - changed_models.begin(), 2);

        bool model0_created = false;
        bool model1_created = false;
        for (const MeshModelID model_ID : changed_models) {
            if (model_ID == model_ID0 && MeshModels::get_changes(model_ID) == MeshModels::Change::Created)
                model0_created = true;
            if (model_ID == model_ID1 && MeshModels::get_changes(model_ID) == MeshModels::Change::Created)
                model1_created = true;
        }

        EXPECT_TRUE(model0_created);
        EXPECT_TRUE(model1_created);
    }

    MeshModels::reset_change_notifications();

    { // Test destroy.
        MeshModels::destroy(model_ID0);
        EXPECT_FALSE(MeshModels::has(model_ID0));

        Core::Iterable<MeshModels::ChangedIterator> changed_models = MeshModels::get_changed_models();
        EXPECT_EQ(changed_models.end() - changed_models.begin(), 1);

        bool model0_destroyed = false;
        bool other_change = false;
        for (const MeshModelID model_ID : changed_models) {
            if (model_ID == model_ID0 && MeshModels::get_changes(model_ID) == MeshModels::Change::Destroyed)
                model0_destroyed = true;
            else
                other_change = true;
        }

        EXPECT_TRUE(model0_destroyed);
        EXPECT_FALSE(other_change);
    }

    MeshModels::reset_change_notifications();

    { // Test that destroyed model cannot be destroyed again.
        EXPECT_FALSE(MeshModels::has(model_ID0));

        MeshModels::destroy(model_ID0);
        EXPECT_FALSE(MeshModels::has(model_ID0));
        EXPECT_TRUE(MeshModels::get_changed_models().is_empty());
    }
}

} // NS Assets
} // NS Bifrost

#endif // _BIFROST_ASSETS_MESH_MODEL_TEST_H_
