// Test Cogwheel meshes.
// ---------------------------------------------------------------------------
// Copyright (C) 2015-2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _COGWHEEL_ASSETS_MESH_TEST_H_
#define _COGWHEEL_ASSETS_MESH_TEST_H_

#include <Cogwheel/Assets/Mesh.h>

#include <gtest/gtest.h>

namespace Cogwheel {
namespace Assets {

class Assets_Mesh : public ::testing::Test {
protected:
    // Per-test set-up and tear-down logic.
    virtual void SetUp() {
        Meshes::allocate(8u);
    }
    virtual void TearDown() {
        Meshes::deallocate();
    }
};

TEST_F(Assets_Mesh, resizing) {
    // Test that capacity can be increased.
    unsigned int largerCapacity = Meshes::capacity() + 4u;
    Meshes::reserve(largerCapacity);
    EXPECT_GE(Meshes::capacity(), largerCapacity);

    // Test that capacity won't be decreased.
    Meshes::reserve(5u);
    EXPECT_GE(Meshes::capacity(), largerCapacity);

    Meshes::deallocate();
    EXPECT_LT(Meshes::capacity(), largerCapacity);
}

TEST_F(Assets_Mesh, sentinel_mesh) {
    Meshes::UID sentinel_ID = Meshes::UID::invalid_UID();

    EXPECT_FALSE(Meshes::has(sentinel_ID));
    EXPECT_EQ(Meshes::get_mesh(sentinel_ID).indices_count, 0);
    EXPECT_EQ(Meshes::get_mesh(sentinel_ID).indices, nullptr);
    EXPECT_EQ(Meshes::get_mesh(sentinel_ID).vertex_count, 0u);
    EXPECT_EQ(Meshes::get_mesh(sentinel_ID).positions, nullptr);
    EXPECT_EQ(Meshes::get_mesh(sentinel_ID).normals, nullptr);
    EXPECT_EQ(Meshes::get_mesh(sentinel_ID).texcoords, nullptr);
}

TEST_F(Assets_Mesh, create) {
    Meshes::UID mesh_ID = Meshes::create("TestMesh", 32u, 16u);

    EXPECT_TRUE(Meshes::has(mesh_ID));
    EXPECT_EQ(Meshes::get_mesh(mesh_ID).indices_count, 32);
    EXPECT_NE(Meshes::get_mesh(mesh_ID).indices, nullptr);
    EXPECT_EQ(Meshes::get_mesh(mesh_ID).vertex_count, 16u);
    EXPECT_NE(Meshes::get_mesh(mesh_ID).positions, nullptr);
    EXPECT_NE(Meshes::get_mesh(mesh_ID).normals, nullptr);
    EXPECT_NE(Meshes::get_mesh(mesh_ID).texcoords, nullptr);

    // Test mesh created notification.
    Core::Iterable<Meshes::mesh_created_iterator> created_meshes = Meshes::get_created_meshes();
    EXPECT_EQ(created_meshes.end() - created_meshes.begin(), 1);
    EXPECT_EQ(*created_meshes.begin(), mesh_ID);
}

TEST_F(Assets_Mesh, create_only_positions) {
    Meshes::UID mesh_ID = Meshes::create("TestMesh", 32u, 16u, MeshFlags::Position);

    EXPECT_TRUE(Meshes::has(mesh_ID));
    EXPECT_EQ(Meshes::get_mesh(mesh_ID).indices_count, 32);
    EXPECT_NE(Meshes::get_mesh(mesh_ID).indices, nullptr);
    EXPECT_EQ(Meshes::get_mesh(mesh_ID).vertex_count, 16u);
    EXPECT_NE(Meshes::get_mesh(mesh_ID).positions, nullptr);
    EXPECT_EQ(Meshes::get_mesh(mesh_ID).normals, nullptr);
    EXPECT_EQ(Meshes::get_mesh(mesh_ID).texcoords, nullptr);

    // Test mesh created notification.
    Core::Iterable<Meshes::mesh_created_iterator> created_meshes = Meshes::get_created_meshes();
    EXPECT_EQ(created_meshes.end() - created_meshes.begin(), 1);
    EXPECT_EQ(*created_meshes.begin(), mesh_ID);
}

TEST_F(Assets_Mesh, destroy) {
    Meshes::UID mesh_ID = Meshes::create("TestMesh", 32u, 16u);
    EXPECT_TRUE(Meshes::has(mesh_ID));

    Meshes::reset_change_notifications();

    Meshes::destroy(mesh_ID);
    EXPECT_FALSE(Meshes::has(mesh_ID));

    // Test mesh destroyed notification.
    Core::Iterable<Meshes::mesh_destroyed_iterator> destroyed_meshes = Meshes::get_destroyed_meshes();
    EXPECT_EQ(destroyed_meshes.end() - destroyed_meshes.begin(), 1);
    EXPECT_EQ(*destroyed_meshes.begin(), mesh_ID);
}

TEST_F(Assets_Mesh, create_and_destroy_notifications) {
    Meshes::UID mesh_ID0 = Meshes::create("TestMesh0", 32u, 16u);
    Meshes::UID mesh_ID1 = Meshes::create("TestMesh1", 32u, 16u);
    EXPECT_TRUE(Meshes::has(mesh_ID0));
    EXPECT_TRUE(Meshes::has(mesh_ID1));

    { // Test mesh create notifications.
        Core::Iterable<Meshes::mesh_created_iterator> created_meshes = Meshes::get_created_meshes();
        EXPECT_EQ(created_meshes.end() - created_meshes.begin(), 2);
        Core::Iterable<Meshes::mesh_destroyed_iterator> destroyed_meshes = Meshes::get_destroyed_meshes();
        EXPECT_EQ(destroyed_meshes.end() - destroyed_meshes.begin(), 0);

        bool mesh0_created = false;
        bool mesh1_created = false;
        for (const Meshes::UID mesh_ID : created_meshes) {
            if (mesh_ID == mesh_ID0)
                mesh0_created = true;
            if (mesh_ID == mesh_ID1)
                mesh1_created = true;
        }

        EXPECT_TRUE(mesh0_created);
        EXPECT_TRUE(mesh1_created);
    }

    Meshes::reset_change_notifications();

    { // Test destroy.
        Meshes::destroy(mesh_ID0);
        EXPECT_FALSE(Meshes::has(mesh_ID0));

        Core::Iterable<Meshes::mesh_created_iterator> created_meshes = Meshes::get_created_meshes();
        EXPECT_EQ(created_meshes.end() - created_meshes.begin(), 0);
        Core::Iterable<Meshes::mesh_destroyed_iterator> destroyed_meshes = Meshes::get_destroyed_meshes();
        EXPECT_EQ(destroyed_meshes.end() - destroyed_meshes.begin(), 1);

        bool mesh0_destroyed = false;
        bool mesh1_destroyed = false;
        for (const Meshes::UID mesh_ID : destroyed_meshes) {
            if (mesh_ID == mesh_ID0)
                mesh0_destroyed = true;
            if (mesh_ID == mesh_ID1)
                mesh1_destroyed = true;
        }

        EXPECT_TRUE(mesh0_destroyed);
        EXPECT_FALSE(mesh1_destroyed);
    }

    Meshes::reset_change_notifications();

    { // Test that destroyed mesh cannot be destroyed again.
        Meshes::destroy(mesh_ID0);
        EXPECT_FALSE(Meshes::has(mesh_ID0));

        Core::Iterable<Meshes::mesh_created_iterator> created_meshes = Meshes::get_created_meshes();
        EXPECT_EQ(created_meshes.end() - created_meshes.begin(), 0);
        Core::Iterable<Meshes::mesh_destroyed_iterator> destroyed_meshes = Meshes::get_destroyed_meshes();
        EXPECT_EQ(destroyed_meshes.end() - destroyed_meshes.begin(), 0);
    }
}

} // NS Assets
} // NS Cogwheel

#endif // _COGWHEEL_ASSETS_MESH_TEST_H_