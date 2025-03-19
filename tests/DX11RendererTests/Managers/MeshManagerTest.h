// DX11Renderer mesh manager test.
// ------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _DX11RENDERER_MESH_MANAGER_TEST_H_
#define _DX11RENDERER_MESH_MANAGER_TEST_H_

#include <gtest/gtest.h>
#include <Utils.h>

#include <DX11Renderer/Managers/MeshManager.h>

#include <Bifrost/Assets/Mesh.h>

namespace DX11Renderer::Managers {

class MeshManagerFixture : public ::testing::Test {
protected:
    ODevice1 device;

    // Per-test set-up and tear-down logic.
    virtual void SetUp() {
        Bifrost::Assets::Meshes::allocate(8u);
        device = create_test_device();
    }
    virtual void TearDown() {
        Bifrost::Assets::Meshes::deallocate();
    }
};

TEST_F(MeshManagerFixture, create_dx_mesh_representation) {
    using namespace Bifrost::Assets;

    MeshManager mesh_manager(device);
    const OBuffer& null_buffer = mesh_manager.get_null_buffer();

    // Create mesh with just geometry, mesh with texcoords and mesh with colors
    Mesh geometry_mesh = create_triangle("Geometry");
    Mesh texcoords_mesh = create_triangle("Texcoords", MeshFlag::Texcoord);
    Mesh tints_mesh = create_triangle("Tints", MeshFlag::TintAndRoughness);

    mesh_manager.handle_updates(device);

    // Verify that the meshes were converted to their DX representation
    for (Mesh mesh : { geometry_mesh, texcoords_mesh, tints_mesh }) {
        Dx11Mesh dx_mesh = mesh_manager.get_mesh(mesh.get_ID());

        EXPECT_EQ(mesh.get_bounds().minimum.x, dx_mesh.bounds.min.x);
        EXPECT_EQ(mesh.get_bounds().minimum.y, dx_mesh.bounds.min.y);
        EXPECT_EQ(mesh.get_bounds().minimum.z, dx_mesh.bounds.min.z);
        EXPECT_EQ(mesh.get_bounds().maximum.x, dx_mesh.bounds.max.x);
        EXPECT_EQ(mesh.get_bounds().maximum.y, dx_mesh.bounds.max.y);
        EXPECT_EQ(mesh.get_bounds().maximum.z, dx_mesh.bounds.max.z);

        EXPECT_EQ(mesh.get_index_count(), dx_mesh.index_count);
        EXPECT_EQ(mesh.get_vertex_count(), dx_mesh.vertex_count);

        EXPECT_NE(null_buffer, dx_mesh.indices);

        // Test that empty buffers are set to the null buffer.
        if (mesh.get_texcoords() == nullptr)
            EXPECT_EQ(null_buffer, *dx_mesh.texcoords_address());
        if (mesh.get_tint_and_roughness() == nullptr)
            EXPECT_EQ(null_buffer, *dx_mesh.tint_and_roughness_address());
    }
}

TEST_F(MeshManagerFixture, created_and_destroyed_mesh_is_ignored) {
    MeshManager mesh_manager(device);

    auto mesh = create_triangle("Geometry");
    unsigned int mesh_index = mesh.get_ID();
    mesh.destroy();

    mesh_manager.handle_updates(device);

    Dx11Mesh dx_mesh = mesh_manager.get_mesh(mesh_index);
    EXPECT_EQ(dx_mesh.vertex_count, 0);
    EXPECT_EQ(dx_mesh.index_count, 0);
    EXPECT_EQ(dx_mesh.vertex_buffer_count, 0);
    EXPECT_EQ(nullptr, *dx_mesh.geometry_address());
}

TEST_F(MeshManagerFixture, destroyed_mesh_is_cleared) {
    MeshManager mesh_manager(device);

    auto mesh = create_triangle("Geometry");
    unsigned int mesh_index = mesh.get_ID();

    mesh_manager.handle_updates(device);

    // Test that the mesh was created
    Dx11Mesh dx_mesh = mesh_manager.get_mesh(mesh_index);
    EXPECT_EQ(dx_mesh.vertex_count, mesh.get_vertex_count());
    EXPECT_EQ(dx_mesh.index_count, mesh.get_index_count());

    mesh.destroy();
    mesh_manager.handle_updates(device);

    // Test that the mesh is cleared
    dx_mesh = mesh_manager.get_mesh(mesh_index);
    EXPECT_EQ(dx_mesh.vertex_count, 0);
    EXPECT_EQ(dx_mesh.index_count, 0);
    EXPECT_EQ(dx_mesh.vertex_buffer_count, 0);
    EXPECT_EQ(nullptr, *dx_mesh.geometry_address());
}

} // NS DX11Renderer::Managers

#endif // _DX11RENDERER_MESH_MANAGER_TEST_H_