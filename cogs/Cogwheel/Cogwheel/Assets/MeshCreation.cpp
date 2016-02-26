// Cogwheel mesh creation utilities.
// ---------------------------------------------------------------------------
// Copyright (C) 2015-2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#include <Cogwheel/Assets/MeshCreation.h>
#include <Cogwheel/Math/Constants.h>
#include <Cogwheel/Math/Vector.h>
#include <Cogwheel/Math/Utils.h>

using namespace Cogwheel::Math;

namespace Cogwheel {
namespace Assets {
namespace MeshCreation {

Meshes::UID plane(unsigned int quads_pr_edge) {
    if (quads_pr_edge == 0)
        return Meshes::UID::invalid_UID();

    unsigned int size = quads_pr_edge + 1;
    unsigned int vertex_count = size * size;
    unsigned int quad_count = quads_pr_edge * quads_pr_edge;
    unsigned int indices_count = quad_count * 2;
    
    Meshes::UID mesh_ID = Meshes::create("Plane", indices_count, vertex_count);
    Mesh& mesh = Meshes::get_mesh(mesh_ID);

    // Vertex attributes.
    float tc_normalizer = 1.0f / quads_pr_edge;
    for (unsigned int z = 0; z < size; ++z) {
        for (unsigned int x = 0; x < size; ++x) {
            mesh.positions[z * size + x] = Vector3f(x - quads_pr_edge * 0.5f, 0.0f, z - quads_pr_edge * 0.5f);
            mesh.normals[z * size + x] = Vector3f(0.0f, 1.0f, 0.0f);
            mesh.texcoords[z * size + x] = Vector2f(float(x), float(z)) * tc_normalizer;
        }
    }

    // Indices
    for (unsigned int z = 0; z < quads_pr_edge; ++z) {
        for (unsigned int x = 0; x < quads_pr_edge; ++x) {
            Vector3ui* indices = mesh.indices + (z * quads_pr_edge + x) * 2;
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
    Vector3f* position_iterator = mesh.positions;
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
    Vector3f* normal_iterator = mesh.normals;
    while (normal_iterator < mesh.normals + verts_pr_side) // Top
        *normal_iterator++ = Vector3f(0, 1, 0);
    while (normal_iterator < mesh.normals + verts_pr_side * 2) // Bottom
        *normal_iterator++ = Vector3f(0, -1, 0);
    while (normal_iterator < mesh.normals + verts_pr_side * 3) // Left
        *normal_iterator++ = Vector3f(-1, 0, 0);
    while (normal_iterator < mesh.normals + verts_pr_side * 4) // Right
        *normal_iterator++ = Vector3f(1, 0, 0);
    while (normal_iterator < mesh.normals + verts_pr_side * 5) // Front
        *normal_iterator++ = Vector3f(0, 0, 1);
    while (normal_iterator < mesh.normals + verts_pr_side * 6) // Back
        *normal_iterator++ = Vector3f(0, 0, -1);

    // Default texcoords.
    float tc_normalizer = 1.0f / quads_pr_edge;
    for (unsigned int i = 0; i < verts_pr_edge; ++i)
        for (unsigned int j = 0; j < verts_pr_edge; ++j)
            mesh.texcoords[i * verts_pr_edge + j] = 
            mesh.texcoords[i * verts_pr_edge + j + verts_pr_side] =
            mesh.texcoords[i * verts_pr_edge + j + verts_pr_side * 2] =
            mesh.texcoords[i * verts_pr_edge + j + verts_pr_side * 3] =
            mesh.texcoords[i * verts_pr_edge + j + verts_pr_side * 4] =
            mesh.texcoords[i * verts_pr_edge + j + verts_pr_side * 5] = Vector2f(float(i), float(j)) * tc_normalizer;

    // Set indices.
    int index = 0;
    for (unsigned int side_offset = 0; side_offset < vertex_count; side_offset += verts_pr_side)
        for (unsigned int i = 0; i < quads_pr_edge; ++i)
            for (unsigned int j = 0; j < quads_pr_edge; ++j) {
                mesh.indices[index++] = Vector3ui(j + i * verts_pr_edge,
                                                  j + 1 + i * verts_pr_edge,
                                                  j + (i + 1) * verts_pr_edge) + side_offset;

                mesh.indices[index++] = Vector3ui(j + 1 + i * verts_pr_edge,
                                                  j + 1 + (i + 1) * verts_pr_edge,
                                                  j + (i + 1) * verts_pr_edge) + side_offset;
            }

    Meshes::set_bounds(mesh_ID, AABB(Vector3f(-halfsize), Vector3f(halfsize)));

    return mesh_ID;
}

Meshes::UID cylinder(unsigned int vertical_quads, unsigned int circumference_quads) {
    if (vertical_quads == 0 || circumference_quads == 0)
        return Meshes::UID::invalid_UID();

    unsigned int lid_vertex_count = circumference_quads + 1;
    unsigned int side_vertex_count = (vertical_quads + 1) * circumference_quads;
    unsigned int vertex_count = 2 * lid_vertex_count + side_vertex_count;
    unsigned int lid_indices_count = circumference_quads;
    unsigned int side_indices_count = 2 * vertical_quads * circumference_quads;
    unsigned int indices_count = 2 * lid_indices_count + side_indices_count;
    float radius = 0.5f;

    Meshes::UID mesh_ID = Meshes::create("Cylinder", indices_count, vertex_count);
    Mesh& mesh = Meshes::get_mesh(mesh_ID);

    // Vertex layout is
    // [..TOP.. ..BOTTOM.. ..SIDE..]

    { // Vertices
        // Create top vertices.
        mesh.positions[0] = Vector3f(0.0f, radius, 0.0f);
        for (unsigned int v = 0; v < circumference_quads; ++v) {
            float radians = v / float(circumference_quads) * 2.0f * Math::PI<float>();
            mesh.positions[v + 1] = Vector3f(cos(radians) * radius, radius, sin(radians) * radius);
        }

        // Mirror top to create bottom vertices.
        for (unsigned int v = 0; v < lid_vertex_count; ++v) {
            mesh.positions[lid_vertex_count + v] = mesh.positions[v];
            mesh.positions[lid_vertex_count + v].y = -radius;
        }

        // Create side vertices
        for (unsigned int i = 0; i < vertical_quads + 1; ++i) {
            float l = i / float(vertical_quads);
            for (unsigned int j = 0; j < circumference_quads; ++j) {
                unsigned int vertex_index = 2 * lid_vertex_count + i * circumference_quads + j;
                mesh.positions[vertex_index] = mesh.positions[j+1];
                mesh.positions[vertex_index].y = lerp(radius, -radius, l);
            }
        }
    }

    { // Normals
        // Get rid of the loop counter.
        Vector3f* normal_iterator = mesh.normals;
        while (normal_iterator < mesh.normals + lid_vertex_count) // Top
            *normal_iterator++ = Vector3f(0, 1, 0);
        while (normal_iterator < mesh.normals + 2 * lid_vertex_count) // Bottom
            *normal_iterator++ = Vector3f(0, -1, 0);
        Vector3f* side_position_iterator = mesh.positions + 2 * lid_vertex_count;
        while (normal_iterator < mesh.normals + mesh.vertex_count) { // Side
            Vector3f position = *side_position_iterator++;
            *normal_iterator++ = normalize(Vector3f(position.x, 0.0f, position.z));
        }
    }

    { // tex coords
        // Top and bottom.
        for (unsigned int i = 0; i < 2 * lid_vertex_count; ++i) {
            Vector3f position = mesh.positions[i];
            mesh.texcoords[i] = Vector2f(position.x, position.z) + 0.5f;
        }

        // Side
        for (unsigned int i = 0; i < vertical_quads + 1; ++i) {
            float v = i / float(vertical_quads);
            for (unsigned int j = 0; j < circumference_quads; ++j) {
                unsigned int vertex_index = 2 * lid_vertex_count + i * circumference_quads + j;
                float u = abs(-2.0f * j / float(circumference_quads) + 1.0f); // Magic u mapping. Mirror repeat mapping of the texture coords.
                mesh.texcoords[vertex_index] = Vector2f(u, v);
            }
        }
    }

    { // Indices
        // Top
        for (unsigned int i = 0; i < lid_indices_count; ++i)
            mesh.indices[i] = Vector3ui(0, i + 1, i + 2);
        mesh.indices[lid_indices_count - 1].z = 1;
        
        // Bottom
        for (unsigned int i = 0; i < lid_indices_count; ++i)
            mesh.indices[i + lid_indices_count] = Vector3ui(0, i + 2, i + 1) + lid_vertex_count;
        mesh.indices[2 * lid_indices_count - 1].y = 1 + lid_vertex_count;

        // Side
        Vector3f* side_positions = mesh.positions + 2 * lid_vertex_count;
        unsigned int side_vertex_offset = 2 * lid_vertex_count;
        for (unsigned int i = 0; i < vertical_quads; ++i) {
            for (unsigned int j = 0; j < circumference_quads; ++j) {
                unsigned int side_index = 2 * lid_indices_count + 2 * (i * circumference_quads + j);
                
                unsigned int i0 = i * circumference_quads + j;
                unsigned int i1 = (i + 1) * circumference_quads + j;
                unsigned int j_plus_1 = (j + 1) < circumference_quads ? (j + 1) : 0; // Handle wrap around.
                unsigned int i2 = i * circumference_quads + j_plus_1;
                unsigned int i3 = (i + 1) * circumference_quads + j_plus_1;
                
                mesh.indices[side_index + 0] = Vector3ui(i0, i1, i3) + side_vertex_offset;
                mesh.indices[side_index + 1] = Vector3ui(i0, i3, i2) + side_vertex_offset;
            }
        }
    }

    Meshes::set_bounds(mesh_ID, AABB(Vector3f(-radius), Vector3f(radius)));

    return mesh_ID;
}

static Vector3f spherical_to_direction(float theta, float phi) {
    float sinTheta = sin(theta);
    float z = sinTheta * cos(phi);
    float x = -sinTheta * sin(phi);
    float y = cos(theta);
    return Vector3f(x, y, z);
}

Meshes::UID revolved_sphere(unsigned int longitude_quads, unsigned int latitude_quads) {
    if (longitude_quads < 3 || latitude_quads < 2)
        return Meshes::UID::invalid_UID();

    unsigned int latitude_size = latitude_quads + 1;
    unsigned int longitude_size = longitude_quads + 1;
    unsigned int vertex_count = latitude_size * longitude_size;
    unsigned int quad_count = latitude_quads * longitude_quads;
    unsigned int indices_count = quad_count * 2;
    float radius = 0.5f;

    Meshes::UID mesh_ID = Meshes::create("RevolvedSphere", indices_count, vertex_count);
    Mesh& mesh = Meshes::get_mesh(mesh_ID);

    { // Vertex attributes.
        Vector2f tc_normalizer = Vector2f(1.0f / longitude_quads, 1.0f / latitude_quads);
        for (unsigned int y = 0; y < latitude_size; ++y) {
            for (unsigned int x = 0; x < longitude_size; ++x) {
                unsigned int vertex_index = y * longitude_size + x;
                Vector2f tc = mesh.texcoords[vertex_index] = Vector2f(float(x), float(y)) * tc_normalizer;
                mesh.positions[vertex_index] = spherical_to_direction(tc.y * Math::PI<float>(),
                                                                        tc.x * 2.0f * Math::PI<float>()) * radius;
                mesh.normals[vertex_index] = normalize(mesh.positions[vertex_index]);
            }
        }

        // Hard set the poles to [0,1,0] and [0,-1,0].
        for (unsigned int x = 0; x < longitude_size; ++x) {
            mesh.positions[x] = Vector3f(0, radius, 0);
            mesh.positions[(latitude_size - 1) * longitude_size + x] = Vector3f(0, -radius, 0);
        }
    }

    // Indices
    for (unsigned int y = 0; y < latitude_quads; ++y) {
        for (unsigned int x = 0; x < longitude_quads; ++x) {
            Vector3ui* indices = mesh.indices + (y * longitude_quads + x) * 2;
            unsigned int base_vertex_index = x + y * longitude_size;
            indices[0] = Vector3ui(0, longitude_size, 1) + base_vertex_index;
            indices[1] = Vector3ui(1, longitude_size, longitude_size + 1) + base_vertex_index;
        }
    }

    Meshes::set_bounds(mesh_ID, AABB(Vector3f(-radius), Vector3f(radius)));

    return mesh_ID;
}

} // NS MeshCreation
} // NS Assets
} // NS Cogwheel
