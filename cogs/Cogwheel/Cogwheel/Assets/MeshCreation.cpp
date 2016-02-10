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

Meshes::UID plane(unsigned int quads_pr_side) {
    if (quads_pr_side == 0)
        return Meshes::UID::invalid_UID();

    unsigned int size = quads_pr_side + 1;
    unsigned int quad_count = quads_pr_side * quads_pr_side;
    unsigned int indices_count = quad_count * 2;
    unsigned int vertex_count = size * size;
    
    Meshes::UID mesh_ID = Meshes::create("Plane", indices_count, vertex_count);
    Mesh& mesh = Meshes::get_mesh(mesh_ID);

    float tc_normalizer = 1.0f / quads_pr_side;
    for (unsigned int z = 0; z < size; ++z) {
        for (unsigned int x = 0; x < size; ++x) {
            mesh.m_positions[z * size + x] = Vector3f(x - quads_pr_side * 0.5f, 0.0f, z - quads_pr_side * 0.5f);
            mesh.m_normals[z * size + x] = Vector3f(0.0f, 1.0f, 0.0f);
            mesh.m_texcoords[z * size + x] = Vector2f(x * tc_normalizer, z * tc_normalizer);
        }
    }

    for (unsigned int z = 0; z < quads_pr_side; ++z) {
        for (unsigned int x = 0; x < quads_pr_side; ++x) {
            Vector3ui* indices = mesh.m_indices + (z * quads_pr_side + x) * 2;
            unsigned int base_index = x + z * size;
            indices[0] = Vector3ui(base_index, base_index + size, base_index + 1);
            indices[1] = Vector3ui(base_index + 1, base_index + size, base_index + size + 1);
        }
    }

    Meshes::compute_bounds(mesh_ID);

    return mesh_ID;
}

} // NS MeshCreation
} // NS Assets
} // NS Cogwheel
