// Cogwheel mesh creation utilities.
// ---------------------------------------------------------------------------
// Copyright (C) 2015-2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#include <Cogwheel/Assets/MeshCreation.h>

using namespace Cogwheel::Math;

namespace Cogwheel {
namespace Assets {
namespace MeshCreation {

Meshes::UID plane(unsigned int quads_pr_edge) {
    if (quads_pr_edge == 0)
        return Meshes::UID::invalid_UID();

    unsigned int size = quads_pr_edge + 1;
    unsigned int quad_count = quads_pr_edge * quads_pr_edge;
    unsigned int indices_count = quad_count * 2;
    unsigned int vertex_count = size * size;
    
    Meshes::UID mesh_ID = Meshes::create("Plane", indices_count, vertex_count);
    Mesh& mesh = Meshes::get_mesh(mesh_ID);

    float tc_normalizer = 1.0f / quads_pr_edge;
    for (unsigned int z = 0; z < size; ++z) {
        for (unsigned int x = 0; x < size; ++x) {
            mesh.m_positions[z * size + x] = Vector3f(x - quads_pr_edge * 0.5f, 0.0f, z - quads_pr_edge * 0.5f);
            mesh.m_normals[z * size + x] = Vector3f(0.0f, 1.0f, 0.0f);
            mesh.m_texcoords[z * size + x] = Vector2f(x * tc_normalizer, z * tc_normalizer);
        }
    }

    for (unsigned int z = 0; z < quads_pr_edge; ++z) {
        for (unsigned int x = 0; x < quads_pr_edge; ++x) {
            Vector3ui* indices = mesh.m_indices + (z * quads_pr_edge + x) * 2;
            unsigned int base_index = x + z * size;
            indices[0] = Vector3ui(base_index, base_index + size, base_index + 1);
            indices[1] = Vector3ui(base_index + 1, base_index + size, base_index + size + 1);
        }
    }

    Meshes::compute_bounds(mesh_ID);

    return mesh_ID;
}

Meshes::UID cube(unsigned int quads_pr_edge) {
    if (quads_pr_edge == 0)
        return Meshes::UID::invalid_UID();

    unsigned int sides = 6;

    unsigned int verts_pr_edge = quads_pr_edge + 1;
    float scale = 1.0f / quads_pr_edge;
    float halfsize = 0.5f; // verts_pr_edge * 0.5f;
    unsigned int quad_count = quads_pr_edge * quads_pr_edge * sides;
    unsigned int indices_count = quad_count * 2;
    unsigned int verts_pr_side = verts_pr_edge * verts_pr_edge;
    unsigned int vertex_count = verts_pr_side * sides;

    Meshes::UID mesh_ID = Meshes::create("Cube", indices_count, vertex_count);
    Mesh& mesh = Meshes::get_mesh(mesh_ID);

    // Create the vertices.
    // [..TOP.. ..BOTTOM.. ..LEFT.. ..RIGHT.. ..FRONT.. ..BACK..]
    Vector3f* position_iterator = mesh.m_positions;
    for (unsigned int i = 0; i < verts_pr_edge; ++i) // Top
        for (unsigned int j = 0; j < verts_pr_edge; ++j)
            *position_iterator++ = Vector3f(halfsize - i * scale, halfsize, j * scale - halfsize);
    for (unsigned int i = 0; i < verts_pr_edge; ++i) // Bottom
        for (unsigned int j = 0; j < verts_pr_edge; ++j)
            *position_iterator++ = Vector3f(halfsize - i * scale, -halfsize, halfsize - j * scale);
    for (unsigned int i = 0; i < verts_pr_edge; ++i) // Left
        for (unsigned int j = 0; j < verts_pr_edge; ++j)
            *position_iterator++ = Vector3f(-halfsize, halfsize - i * scale, j * scale - halfsize);
    for (unsigned int i = 0; i < verts_pr_edge; ++i) // Right
        for (unsigned int j = 0; j < verts_pr_edge; ++j)
            *position_iterator++ = Vector3f(halfsize, i * scale - halfsize, j * scale - halfsize);
    for (unsigned int i = 0; i < verts_pr_edge; ++i) // Front
        for (unsigned int j = 0; j < verts_pr_edge; ++j)
            *position_iterator++ = Vector3f(i * scale - halfsize, halfsize - j * scale, -halfsize);
    for (unsigned int i = 0; i < verts_pr_edge; ++i) // Back
        for (unsigned int j = 0; j < verts_pr_edge; ++j)
            *position_iterator++ = Vector3f(halfsize - i * scale, halfsize - j * scale, halfsize);

    // Create the normals.
    Vector3f* normal_iterator = mesh.m_normals;
    while (normal_iterator < mesh.m_normals + verts_pr_side) // Top
        *normal_iterator++ = Vector3f(0, 1, 0);
    while (normal_iterator < mesh.m_normals + verts_pr_side * 2) // Bottom
        *normal_iterator++ = Vector3f(0, -1, 0);
    while (normal_iterator < mesh.m_normals + verts_pr_side * 3) // Left
        *normal_iterator++ = Vector3f(-1, 0, 0);
    while (normal_iterator < mesh.m_normals + verts_pr_side * 4) // Right
        *normal_iterator++ = Vector3f(1, 0, 0);
    while (normal_iterator < mesh.m_normals + verts_pr_side * 5) // Front
        *normal_iterator++ = Vector3f(0, 0, 1);
    while (normal_iterator < mesh.m_normals + verts_pr_side * 6) // Back
        *normal_iterator++ = Vector3f(0, 0, -1);

    // Default texcoords.
    float tc_normalizer = 1.0f / quads_pr_edge;
    for (unsigned int i = 0; i < verts_pr_edge; ++i)
        for (unsigned int j = 0; j < verts_pr_edge; ++j)
            mesh.m_texcoords[i * verts_pr_edge + j] = 
            mesh.m_texcoords[i * verts_pr_edge + j + verts_pr_side] =
            mesh.m_texcoords[i * verts_pr_edge + j + verts_pr_side * 2] =
            mesh.m_texcoords[i * verts_pr_edge + j + verts_pr_side * 3] =
            mesh.m_texcoords[i * verts_pr_edge + j + verts_pr_side * 4] =
            mesh.m_texcoords[i * verts_pr_edge + j + verts_pr_side * 5] = Vector2f(float(i), float(j)) * tc_normalizer;

    // Set indices.
    int index = 0;
    for (unsigned int side_offset = 0; side_offset < vertex_count; side_offset += verts_pr_side)
        for (unsigned int i = 0; i < quads_pr_edge; ++i)
            for (unsigned int j = 0; j < quads_pr_edge; ++j) {
                mesh.m_indices[index++] = Vector3ui(j + i * verts_pr_edge,
                                                    j + 1 + i * verts_pr_edge,
                                                    j + (i + 1) * verts_pr_edge) + side_offset;

                mesh.m_indices[index++] = Vector3ui(j + 1 + i * verts_pr_edge,
                                                    j + 1 + (i + 1) * verts_pr_edge,
                                                    j + (i + 1) * verts_pr_edge) + side_offset;
            }

    return mesh_ID;
}

} // NS MeshCreation
} // NS Assets
} // NS Cogwheel
