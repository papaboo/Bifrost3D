// Test Cogwheel mesh models.
// ---------------------------------------------------------------------------
// Copyright (C) 2015-2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _COGWHEEL_ASSETS_MESH_MODEL_TEST_H_
#define _COGWHEEL_ASSETS_MESH_MODEL_TEST_H_

#include <Cogwheel/Assets/MeshModel.h>

#include <gtest/gtest.h>

namespace Cogwheel {
namespace Assets {

class Assets_MeshModels : public ::testing::Test {
protected:
    // Per-test set-up and tear-down logic.
    virtual void SetUp() {
        MeshModels::allocate(8u);
    }
    virtual void TearDown() {
        MeshModels::deallocate();
    }
};

TEST_F(Assets_MeshModels, resizing) {
    // Test that capacity can be increased.
    unsigned int largerCapacity = Meshes::capacity() + 4u;
    MeshModels::reserve(largerCapacity);
    EXPECT_GE(MeshModels::capacity(), largerCapacity);

    // Test that capacity won't be decreased.
    MeshModels::reserve(5u);
    EXPECT_GE(MeshModels::capacity(), largerCapacity);

    MeshModels::deallocate();
    EXPECT_LT(MeshModels::capacity(), largerCapacity);
}

TEST_F(Assets_MeshModels, sentinel_mesh) {
    MeshModels::UID sentinel_ID = MeshModels::UID::invalid_UID();

    EXPECT_FALSE(MeshModels::has(sentinel_ID));
    EXPECT_EQ(MeshModels::get_model(sentinel_ID).m_scene_node_ID, Scene::SceneNodes::UID::invalid_UID());
    EXPECT_EQ(MeshModels::get_scene_node_ID(sentinel_ID), Scene::SceneNodes::UID::invalid_UID());
    EXPECT_EQ(MeshModels::get_model(sentinel_ID).m_mesh_ID, Meshes::UID::invalid_UID());
    EXPECT_EQ(MeshModels::get_mesh_ID(sentinel_ID), Meshes::UID::invalid_UID());
}

TEST_F(Assets_MeshModels, create) {
    Scene::SceneNodes::allocate(1u);
    Scene::SceneNodes::UID node_ID = Scene::SceneNodes::create("TestNode");
    Meshes::allocate(1u);
    Meshes::UID mesh_ID = Meshes::create("TestMesh", 32u, 16u);
    
    MeshModels::UID model_ID = MeshModels::create(node_ID, mesh_ID);

    EXPECT_TRUE(MeshModels::has(model_ID));
    EXPECT_EQ(MeshModels::get_model(model_ID).m_scene_node_ID, node_ID);
    EXPECT_EQ(MeshModels::get_scene_node_ID(model_ID), node_ID);
    EXPECT_EQ(MeshModels::get_model(model_ID).m_mesh_ID, mesh_ID);
    EXPECT_EQ(MeshModels::get_mesh_ID(model_ID), mesh_ID);

    // Test model created notification.
    Core::Iterable<MeshModels::model_created_iterator> created_models = MeshModels::get_created_models();
    EXPECT_EQ(created_models.end() - created_models.begin(), 1);
    EXPECT_EQ(*created_models.begin(), node_ID);

    Meshes::deallocate();
    Scene::SceneNodes::deallocate();
}

TEST_F(Assets_MeshModels, destroy) {
    MeshModels::UID model_ID = MeshModels::create(Scene::SceneNodes::UID::invalid_UID(), Meshes::UID::invalid_UID());
    EXPECT_TRUE(MeshModels::has(model_ID));

    MeshModels::clear_change_notifications();

    MeshModels::destroy(model_ID);
    EXPECT_FALSE(MeshModels::has(model_ID));

    // Test model destroyed notification.
    Core::Iterable<MeshModels::model_destroyed_iterator> destroyed_models = MeshModels::get_destroyed_models();
    EXPECT_EQ(destroyed_models.end() - destroyed_models.begin(), 1);
    EXPECT_EQ(*destroyed_models.begin(), model_ID);

    MeshModels::deallocate();
}


TEST_F(Assets_MeshModels, create_and_destroy_notifications) {
    MeshModels::UID model_ID0 = MeshModels::create(Scene::SceneNodes::UID::invalid_UID(), Meshes::UID::invalid_UID());
    MeshModels::UID model_ID1 = MeshModels::create(Scene::SceneNodes::UID::invalid_UID(), Meshes::UID::invalid_UID());
    EXPECT_TRUE(MeshModels::has(model_ID0));
    EXPECT_TRUE(MeshModels::has(model_ID1));

    { // Test model create notifications.
        Core::Iterable<MeshModels::model_created_iterator> created_models = MeshModels::get_created_models();
        EXPECT_EQ(created_models.end() - created_models.begin(), 2);
        Core::Iterable<MeshModels::model_destroyed_iterator> destroyed_models = MeshModels::get_destroyed_models();
        EXPECT_EQ(destroyed_models.end() - destroyed_models.begin(), 0);

        bool model0_created = false;
        bool model1_created = false;
        for (const MeshModels::UID model_ID : created_models) {
            if (model_ID == model_ID0)
                model0_created = true;
            if (model_ID == model_ID1)
                model1_created = true;
        }

        EXPECT_TRUE(model0_created);
        EXPECT_TRUE(model1_created);
    }

    MeshModels::clear_change_notifications();

    { // Test destroy.
        MeshModels::destroy(model_ID0);
        EXPECT_FALSE(MeshModels::has(model_ID0));

        Core::Iterable<MeshModels::model_created_iterator> created_models = MeshModels::get_created_models();
        EXPECT_EQ(created_models.end() - created_models.begin(), 0);
        Core::Iterable<MeshModels::model_destroyed_iterator> destroyed_models = MeshModels::get_destroyed_models();
        EXPECT_EQ(destroyed_models.end() - destroyed_models.begin(), 1);

        bool model0_destroyed = false;
        bool model1_destroyed = false;
        for (const MeshModels::UID model_ID : destroyed_models) {
            if (model_ID == model_ID0)
                model0_destroyed = true;
            if (model_ID == model_ID1)
                model1_destroyed = true;
        }

        EXPECT_TRUE(model0_destroyed);
        EXPECT_FALSE(model1_destroyed);
    }

    MeshModels::clear_change_notifications();

    { // Test that destroyed model cannot be destroyed again.
        MeshModels::destroy(model_ID0);
        EXPECT_FALSE(MeshModels::has(model_ID0));

        Core::Iterable<MeshModels::model_created_iterator> created_models = MeshModels::get_created_models();
        EXPECT_EQ(created_models.end() - created_models.begin(), 0);
        Core::Iterable<MeshModels::model_destroyed_iterator> destroyed_models = MeshModels::get_destroyed_models();
        EXPECT_EQ(destroyed_models.end() - destroyed_models.begin(), 0);
    }
}

} // NS Assets
} // NS Cogwheel

#endif // _COGWHEEL_ASSETS_MESH_MODEL_TEST_H_