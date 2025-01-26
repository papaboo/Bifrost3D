// Test Bifrost meshes.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _BIFROST_ASSETS_MESH_TEST_H_
#define _BIFROST_ASSETS_MESH_TEST_H_

#include <Bifrost/Assets/Mesh.h>
#include <Bifrost/Assets/MeshCreation.h>

#include <gtest/gtest.h>

namespace Bifrost {
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
    MeshID sentinel_ID = MeshID::invalid_UID();

    EXPECT_FALSE(Meshes::has(sentinel_ID));
    EXPECT_EQ(0u, Meshes::get_primitive_count(sentinel_ID));
    EXPECT_EQ(nullptr, Meshes::get_primitives(sentinel_ID));
    EXPECT_EQ(0u, Meshes::get_vertex_count(sentinel_ID));
    EXPECT_EQ(nullptr, Meshes::get_positions(sentinel_ID));
    EXPECT_EQ(nullptr, Meshes::get_normals(sentinel_ID));
    EXPECT_EQ(nullptr, Meshes::get_texcoords(sentinel_ID));
    Math::AABB bounds = Meshes::get_bounds(sentinel_ID);
    EXPECT_TRUE(isnan(bounds.minimum.x) && isnan(bounds.minimum.y) && isnan(bounds.minimum.z) &&
                isnan(bounds.maximum.x) && isnan(bounds.maximum.y) && isnan(bounds.maximum.z));
}

TEST_F(Assets_Mesh, create) {
    MeshID mesh_ID = Meshes::create("TestMesh", 32u, 16u);

    EXPECT_TRUE(Meshes::has(mesh_ID));
    EXPECT_EQ(32u, Meshes::get_primitive_count(mesh_ID));
    EXPECT_NE(nullptr, Meshes::get_primitives(mesh_ID));
    EXPECT_EQ(16u, Meshes::get_vertex_count(mesh_ID));
    EXPECT_NE(nullptr, Meshes::get_positions(mesh_ID));
    EXPECT_NE(nullptr, Meshes::get_normals(mesh_ID));
    EXPECT_NE(nullptr, Meshes::get_texcoords(mesh_ID));
    EXPECT_INVALID_AABB(Meshes::get_bounds(mesh_ID));

    // Test mesh created notification.
    Core::Iterable<Meshes::ChangedIterator> changed_meshes = Meshes::get_changed_meshes();
    EXPECT_EQ(1, changed_meshes.end() - changed_meshes.begin());
    EXPECT_EQ(mesh_ID, *changed_meshes.begin());
    EXPECT_EQ(Meshes::Change::Created, Meshes::get_changes(mesh_ID));
}

TEST_F(Assets_Mesh, create_only_positions) {
    MeshID mesh_ID = Meshes::create("TestMesh", 32u, 16u, MeshFlag::Position);

    EXPECT_TRUE(Meshes::has(mesh_ID));
    EXPECT_EQ(32u, Meshes::get_primitive_count(mesh_ID));
    EXPECT_NE(nullptr, Meshes::get_primitives(mesh_ID));
    EXPECT_EQ(16u, Meshes::get_vertex_count(mesh_ID));
    EXPECT_NE(nullptr, Meshes::get_positions(mesh_ID));
    EXPECT_EQ(nullptr, Meshes::get_normals(mesh_ID));
    EXPECT_EQ(nullptr, Meshes::get_texcoords(mesh_ID));

    // Test mesh created notification.
    Core::Iterable<Meshes::ChangedIterator> changed_meshes = Meshes::get_changed_meshes();
    EXPECT_EQ(1, changed_meshes.end() - changed_meshes.begin());
    EXPECT_EQ(mesh_ID, *changed_meshes.begin());
    EXPECT_EQ(Meshes::Change::Created, Meshes::get_changes(mesh_ID));
}

TEST_F(Assets_Mesh, destroy) {
    MeshID mesh_ID = Meshes::create("TestMesh", 32u, 16u);
    EXPECT_TRUE(Meshes::has(mesh_ID));

    Meshes::reset_change_notifications();

    Meshes::destroy(mesh_ID);
    EXPECT_FALSE(Meshes::has(mesh_ID));

    // Test mesh destroyed notification.
    Core::Iterable<Meshes::ChangedIterator> changed_meshes = Meshes::get_changed_meshes();
    EXPECT_EQ(1, changed_meshes.end() - changed_meshes.begin());
    EXPECT_EQ(mesh_ID, *changed_meshes.begin());
    EXPECT_EQ(Meshes::Change::Destroyed, Meshes::get_changes(mesh_ID));
}

TEST_F(Assets_Mesh, create_and_destroy_notifications) {
    MeshID mesh_ID0 = Meshes::create("TestMesh0", 32u, 16u);
    MeshID mesh_ID1 = Meshes::create("TestMesh1", 32u, 16u);
    EXPECT_TRUE(Meshes::has(mesh_ID0));
    EXPECT_TRUE(Meshes::has(mesh_ID1));

    { // Test mesh create notifications.
        Core::Iterable<Meshes::ChangedIterator> changed_meshes = Meshes::get_changed_meshes();
        EXPECT_EQ(2u, changed_meshes.end() - changed_meshes.begin());

        bool mesh0_created = false;
        bool mesh1_created = false;
        bool other_changes = false;
        for (const MeshID mesh_ID : changed_meshes) {
            bool mesh_created = Meshes::get_changes(mesh_ID) == Meshes::Change::Created;
            if (mesh_ID == mesh_ID0 && mesh_created)
                mesh0_created = true;
            else if (mesh_ID == mesh_ID1 && mesh_created)
                mesh1_created = true;
            else
                other_changes = true;
        }

        EXPECT_TRUE(mesh0_created);
        EXPECT_TRUE(mesh1_created);
        EXPECT_FALSE(other_changes);
    }

    Meshes::reset_change_notifications();

    { // Test destroy.
        Meshes::destroy(mesh_ID0);
        EXPECT_FALSE(Meshes::has(mesh_ID0));

        Core::Iterable<Meshes::ChangedIterator> changed_meshes = Meshes::get_changed_meshes();
        EXPECT_EQ(1, changed_meshes.end() - changed_meshes.begin());

        bool mesh0_destroyed = false;
        bool other_changes = false;
        for (const MeshID mesh_ID : changed_meshes) {
            if (mesh_ID == mesh_ID0 && Meshes::get_changes(mesh_ID) == Meshes::Change::Destroyed)
                mesh0_destroyed = true;
            else
                other_changes = true;
        }

        EXPECT_TRUE(mesh0_destroyed);
        EXPECT_FALSE(other_changes);
    }

    Meshes::reset_change_notifications();

    { // Test that destroyed mesh cannot be destroyed again.
        EXPECT_FALSE(Meshes::has(mesh_ID0));
        
        Meshes::destroy(mesh_ID0);
        EXPECT_FALSE(Meshes::has(mesh_ID0));
        EXPECT_TRUE(Meshes::get_changed_meshes().is_empty());
    }
}

TEST_F(Assets_Mesh, normals_correspond_to_winding_order) {
    MeshID plane_ID = MeshCreation::plane(3);
    EXPECT_EQ(0, MeshTests::normals_correspond_to_winding_order(plane_ID));
    EXPECT_EQ(0, MeshTests::count_degenerate_primitives(plane_ID, 0.000001f));

    MeshID cube_ID = MeshCreation::cube(3);
    EXPECT_EQ(0, MeshTests::normals_correspond_to_winding_order(cube_ID));
    EXPECT_EQ(0, MeshTests::count_degenerate_primitives(cube_ID, 0.000001f));

    MeshID cylinder_ID = MeshCreation::cylinder(3, 3);
    EXPECT_EQ(0, MeshTests::normals_correspond_to_winding_order(cylinder_ID));
    EXPECT_EQ(0, MeshTests::count_degenerate_primitives(cylinder_ID, 0.000001f));

    MeshID revolved_sphere_ID = MeshCreation::revolved_sphere(3, 3);
    EXPECT_EQ(0, MeshTests::normals_correspond_to_winding_order(revolved_sphere_ID));
    EXPECT_EQ(0, MeshTests::count_degenerate_primitives(revolved_sphere_ID, 0.000001f));
}

TEST_F(Assets_Mesh, expand_index_buffer) {
    Mesh mesh = MeshCreation::cube(2);
    unsigned int expanded_vertex_count = mesh.get_primitive_count() * 3;
    Math::Vector3f* expanded_positions = MeshUtils::expand_indexed_buffer(mesh.get_primitives(), mesh.get_primitive_count(), mesh.get_positions());

    for (unsigned int p = 0; p < mesh.get_primitive_count(); ++p) {
        Math::Vector3ui face = mesh.get_primitives()[p];
        for (int i = 0; i < 3; ++i) {
            Math::Vector3f indexed_position = mesh.get_positions()[face[i]];
            Math::Vector3f expanded_position = expanded_positions[p * 3 + i];
            EXPECT_EQ(indexed_position, expanded_position);
        }
    }
}

TEST_F(Assets_Mesh, hard_normal_computation) {
    using namespace Math;

    // Test that the computed hard normals match the cubes normals, which are hard as well.
    Mesh cube = MeshCreation::cube(2);
    unsigned int expanded_vertex_count = cube.get_primitive_count() * 3;
    Vector3f* positions = MeshUtils::expand_indexed_buffer(cube.get_primitives(), cube.get_primitive_count(), cube.get_positions());
    Vector3f* hard_normals = new Vector3f[expanded_vertex_count];
    MeshUtils::compute_hard_normals(positions, positions + expanded_vertex_count, hard_normals);

    for (unsigned int p = 0; p < cube.get_primitive_count(); ++p) {
        Math::Vector3ui face = cube.get_primitives()[p];
        for (int i = 0; i < 3; ++i) {
            Math::Vector3f cube_normal = cube.get_normals()[face[i]];
            Math::Vector3f hard_normal = hard_normals[p * 3 + i];
            EXPECT_NORMAL_EQ(cube_normal, hard_normal, 0.000001);
        }
    }
}

TEST_F(Assets_Mesh, merge_duplicate_vertices) {
    using namespace Math;

    Mesh two_tris = Meshes::create("two_tris", 2, 6, MeshFlag::Position);
    Vector3f* positions = two_tris.get_positions();
    positions[0] = Vector3f(0, 0, 0);
    positions[1] = positions[3] = Vector3f(1, 0, 0);
    positions[2] = positions[4] = Vector3f(0, 1, 0);
    positions[5] = Vector3f(1, 1, 0);
    Vector3ui* triangles = two_tris.get_primitives();
    triangles[0] = Vector3ui(0, 1, 2);
    triangles[1] = Vector3ui(3, 4, 5);

    // Assert that the number of indices went down to 4.
    Mesh two_merged_tris = MeshUtils::merge_duplicate_vertices(two_tris);
    EXPECT_EQ(4, two_merged_tris.get_vertex_count());

    // Assert that the unique vertices have been kept
    Vector3f* merged_positions = two_merged_tris.get_positions();
    EXPECT_EQ(Vector3f(0, 0, 0), merged_positions[0]);
    EXPECT_EQ(Vector3f(1, 0, 0), merged_positions[1]);
    EXPECT_EQ(Vector3f(0, 1, 0), merged_positions[2]);
    EXPECT_EQ(Vector3f(1, 1, 0), merged_positions[3]);

    // Assert that the new facets reference the correct vertices.
    // Here we make an assumption about the layout of the vertices after merging,
    // which is irrelevant to the functionality, but makes the test simpler.
    Vector3ui* merged_triangles = two_merged_tris.get_primitives();
    EXPECT_EQ(Vector3ui(0, 1, 2), merged_triangles[0]);
    EXPECT_EQ(Vector3ui(1, 2, 3), merged_triangles[1]);
}

TEST_F(Assets_Mesh, merge_duplicate_vertices_returns_copy_if_no_duplicates) {
    using namespace Math;

    Mesh expected_mesh = Meshes::create("two_tris", 2, 6, MeshFlag::Position);
    Vector3f* positions = expected_mesh.get_positions();
    positions[0] = Vector3f(0, 0, 0);
    positions[1] = Vector3f(1, 0, 0);
    positions[2] = Vector3f(0, 1, 0);
    positions[3] = Vector3f(2, 0, 0);
    positions[4] = Vector3f(0, 2, 0);
    positions[5] = Vector3f(2, 2, 0);
    Vector3ui* triangles = expected_mesh.get_primitives();
    triangles[0] = Vector3ui(0, 1, 2);
    triangles[1] = Vector3ui(3, 4, 5);

    // Assert that the meshes are equal
    Mesh actual_mesh = MeshUtils::merge_duplicate_vertices(expected_mesh);
    
    // Assert vertex equality
    EXPECT_EQ(expected_mesh.get_vertex_count(), actual_mesh.get_vertex_count());
    for (unsigned int v = 0; v < expected_mesh.get_vertex_count(); ++v)
        EXPECT_EQ(expected_mesh.get_positions()[v], actual_mesh.get_positions()[v]);

    // Assert primitive equality
    EXPECT_EQ(expected_mesh.get_primitive_count(), actual_mesh.get_primitive_count());
    for (unsigned int p = 0; p < expected_mesh.get_primitive_count(); ++p)
        EXPECT_EQ(expected_mesh.get_primitives()[p], actual_mesh.get_primitives()[p]);
}

TEST_F(Assets_Mesh, merge_duplicate_vertices_with_different_positions_if_positions_are_ignored) {
    using namespace Math;

    // Single triangle with duplicate normals at vertex 1 and 2.
    Mesh single_tri = Meshes::create("single_tri", 1, 3, { MeshFlag::Position, MeshFlag::Normal });
    Vector3f* positions = single_tri.get_positions();
    Vector3f* normals = single_tri.get_normals();
    positions[0] = Vector3f(0, 0, 0); normals[0] = Vector3f(1, 0, 0);
    positions[1] = Vector3f(1, 0, 0); normals[1] = Vector3f(0, 1, 0);
    positions[2] = Vector3f(0, 1, 0); normals[2] = Vector3f(0, 1, 0);
    Vector3ui* triangles = single_tri.get_primitives();
    triangles[0] = Vector3ui(0, 1, 2);

    // Merge vertices with duplicate normals and ignore differing positions.
    Mesh degenerate_tri = MeshUtils::merge_duplicate_vertices(single_tri, MeshFlag::Normal);

    // Assert that the number of indices went down to 2.
    EXPECT_EQ(2, degenerate_tri.get_vertex_count());

    // Assert that the new facet reference the correct vertices.
    Vector3ui degenerate_triangle = degenerate_tri.get_primitives()[0];
    EXPECT_EQ(Vector3ui(0, 1, 1), degenerate_triangle);
}

TEST_F(Assets_Mesh, merge_torus_duplicate_vertices) {
    using namespace Math;

    unsigned int torus_detail = 5;
    float minor_radius = 0.1f;
    Mesh original_mesh = MeshCreation::torus(torus_detail, torus_detail, minor_radius, { MeshFlag::Position, MeshFlag::Normal });
    Mesh merged_mesh = MeshUtils::merge_duplicate_vertices(original_mesh, { MeshFlag::Position, MeshFlag::Normal });

    // Assert that the merged mesh has fewer triangles and that the primitive counts are equal.
    EXPECT_EQ(36, original_mesh.get_vertex_count());
    EXPECT_EQ(25, merged_mesh.get_vertex_count());
    EXPECT_EQ(original_mesh.get_primitive_count(), merged_mesh.get_primitive_count());

    // Assert that triangles in the meshes define the same vertices.
    unsigned int primitive_count = original_mesh.get_primitive_count();
    Vector3ui* original_primitives = original_mesh.get_primitives();
    Vector3ui* merged_primitives = merged_mesh.get_primitives();
    for (unsigned int p = 0; p < primitive_count; ++p) {
        Vector3ui original_primitive = original_primitives[p];
        Vector3ui merged_primitive = merged_primitives[p];

        for (unsigned int v = 0; v < 3; ++v) {
            unsigned int original_vertex_index = original_primitive[v];
            unsigned int merged_vertex_index = merged_primitive[v];

            Vector3f original_position = original_mesh.get_positions()[original_vertex_index];
            Vector3f merged_position = merged_mesh.get_positions()[merged_vertex_index];
            EXPECT_EQ(original_position, merged_position);

            Vector3f original_normal = original_mesh.get_normals()[original_vertex_index];
            Vector3f merged_normal = merged_mesh.get_normals()[merged_vertex_index];
            EXPECT_EQ(original_normal, merged_normal);
        }
    }
}

} // NS Assets
} // NS Bifrost

#endif // _BIFROST_ASSETS_MESH_TEST_H_
