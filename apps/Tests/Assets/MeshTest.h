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
    Meshes::UID mesh_ID = Meshes::create("TestMesh", 32, 16u);

    EXPECT_TRUE(Meshes::has(mesh_ID));
    EXPECT_EQ(Meshes::get_mesh(mesh_ID).indices_count, 32);
    EXPECT_NE(Meshes::get_mesh(mesh_ID).indices, nullptr);
    EXPECT_EQ(Meshes::get_mesh(mesh_ID).vertex_count, 16u);
    EXPECT_NE(Meshes::get_mesh(mesh_ID).positions, nullptr);
    EXPECT_NE(Meshes::get_mesh(mesh_ID).normals, nullptr);
    EXPECT_NE(Meshes::get_mesh(mesh_ID).texcoords, nullptr);
}

} // NS Assets
} // NS Cogwheel

#endif // _COGWHEEL_ASSETS_MESH_TEST_H_