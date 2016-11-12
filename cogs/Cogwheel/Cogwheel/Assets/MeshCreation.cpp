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
    unsigned int index_count = quad_count * 2;
    
    Mesh mesh = Meshes::create("Plane", index_count, vertex_count);

    // Vertex attributes.
    float tc_normalizer = 1.0f / quads_pr_edge;
    for (unsigned int z = 0; z < size; ++z) {
        for (unsigned int x = 0; x < size; ++x) {
            mesh.get_positions()[z * size + x] = Vector3f(x - quads_pr_edge * 0.5f, 0.0f, z - quads_pr_edge * 0.5f);
            mesh.get_normals()[z * size + x] = Vector3f(0.0f, 1.0f, 0.0f);
            mesh.get_texcoords()[z * size + x] = Vector2f(float(x), float(z)) * tc_normalizer;
        }
    }

    // Primitives.
    for (unsigned int z = 0; z < quads_pr_edge; ++z) {
        for (unsigned int x = 0; x < quads_pr_edge; ++x) {
            Vector3ui* primitives = mesh.get_primitives() + (z * quads_pr_edge + x) * 2;
            unsigned int base_index = x + z * size;
            primitives[0] = Vector3ui(base_index, base_index + size, base_index + 1);
            primitives[1] = Vector3ui(base_index + 1, base_index + size, base_index + size + 1);
        }
    }

    mesh.compute_bounds();

    return mesh.get_ID();
}

Meshes::UID cube(unsigned int quads_pr_edge) {
    if (quads_pr_edge == 0)
        return Meshes::UID::invalid_UID();

    unsigned int sides = 6;

    unsigned int verts_pr_edge = quads_pr_edge + 1;
    float scale = 1.0f / quads_pr_edge;
    float halfsize = 0.5f; // verts_pr_edge * 0.5f;
    unsigned int quad_count = quads_pr_edge * quads_pr_edge * sides;
    unsigned int index_count = quad_count * 2;
    unsigned int verts_pr_side = verts_pr_edge * verts_pr_edge;
    unsigned int vertex_count = verts_pr_side * sides;

    Mesh mesh = Meshes::create("Cube", index_count, vertex_count);

    // Create the vertices.
    // [..TOP.. ..BOTTOM.. ..LEFT.. ..RIGHT.. ..FRONT.. ..BACK..]
    Vector3f* position_iterator = mesh.get_positions();
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
            *position_iterator++ = Vector3f(halfsize - i * scale, halfsize - j * scale, halfsize);
    for (unsigned int i = 0; i < verts_pr_edge; ++i) // Back
        for (unsigned int j = 0; j < verts_pr_edge; ++j)
            *position_iterator++ = Vector3f(i * scale - halfsize, halfsize - j * scale, -halfsize);

    // Create the normals.
    Vector3f* normal_iterator = mesh.get_normals();
    while (normal_iterator < mesh.get_normals() + verts_pr_side) // Top
        *normal_iterator++ = Vector3f(0, 1, 0);
    while (normal_iterator < mesh.get_normals() + verts_pr_side * 2) // Bottom
        *normal_iterator++ = Vector3f(0, -1, 0);
    while (normal_iterator < mesh.get_normals() + verts_pr_side * 3) // Left
        *normal_iterator++ = Vector3f(-1, 0, 0);
    while (normal_iterator < mesh.get_normals() + verts_pr_side * 4) // Right
        *normal_iterator++ = Vector3f(1, 0, 0);
    while (normal_iterator < mesh.get_normals() + verts_pr_side * 5) // Front
        *normal_iterator++ = Vector3f(0, 0, 1);
    while (normal_iterator < mesh.get_normals() + verts_pr_side * 6) // Back
        *normal_iterator++ = Vector3f(0, 0, -1);

    // Default texcoords.
    Vector2f* texcoords = mesh.get_texcoords();
    float tc_normalizer = 1.0f / quads_pr_edge;
    for (unsigned int i = 0; i < verts_pr_edge; ++i)
        for (unsigned int j = 0; j < verts_pr_edge; ++j)
            texcoords[i * verts_pr_edge + j] = 
            texcoords[i * verts_pr_edge + j + verts_pr_side] =
            texcoords[i * verts_pr_edge + j + verts_pr_side * 2] =
            texcoords[i * verts_pr_edge + j + verts_pr_side * 3] =
            texcoords[i * verts_pr_edge + j + verts_pr_side * 4] =
            texcoords[i * verts_pr_edge + j + verts_pr_side * 5] = Vector2f(float(i), float(j)) * tc_normalizer;

    // Set indices.
    Vector3ui* primitives = mesh.get_primitives();
    for (unsigned int side_offset = 0; side_offset < vertex_count; side_offset += verts_pr_side)
        for (unsigned int i = 0; i < quads_pr_edge; ++i)
            for (unsigned int j = 0; j < quads_pr_edge; ++j) {
                *primitives++ = Vector3ui(j + i * verts_pr_edge,
                                       j + (i + 1) * verts_pr_edge,
                                       j + 1 + i * verts_pr_edge) + side_offset;

                *primitives++ = Vector3ui(j + 1 + i * verts_pr_edge,
                                       j + (i + 1) * verts_pr_edge,
                                       j + 1 + (i + 1) * verts_pr_edge) + side_offset;
            }

    mesh.set_bounds(AABB(Vector3f(-halfsize), Vector3f(halfsize)));

    return mesh.get_ID();
}

Meshes::UID cylinder(unsigned int vertical_quads, unsigned int circumference_quads) {
    if (vertical_quads == 0 || circumference_quads == 0)
        return Meshes::UID::invalid_UID();

    unsigned int lid_vertex_count = circumference_quads + 1;
    unsigned int side_vertex_count = (vertical_quads + 1) * circumference_quads;
    unsigned int vertex_count = 2 * lid_vertex_count + side_vertex_count;
    unsigned int lid_index_count = circumference_quads;
    unsigned int side_index_count = 2 * vertical_quads * circumference_quads;
    unsigned int index_count = 2 * lid_index_count + side_index_count;
    float radius = 0.5f;

    Mesh mesh = Meshes::create("Cylinder", index_count, vertex_count);

    // Vertex layout is
    // [..TOP.. ..BOTTOM.. ..SIDE..]

    { // Positions.
        Vector3f* positions = mesh.get_positions();
        // Create top positions.
        positions[0] = Vector3f(0.0f, radius, 0.0f);
        for (unsigned int v = 0; v < circumference_quads; ++v) {
            float radians = v / float(circumference_quads) * 2.0f * Math::PI<float>();
            positions[v + 1] = Vector3f(cos(radians) * radius, radius, sin(radians) * radius);
        }

        // Mirror top to create bottom positions.
        for (unsigned int v = 0; v < lid_vertex_count; ++v) {
            positions[lid_vertex_count + v] = positions[v];
            positions[lid_vertex_count + v].y = -radius;
        }

        // Create side positions.
        for (unsigned int i = 0; i < vertical_quads + 1; ++i) {
            float l = i / float(vertical_quads);
            for (unsigned int j = 0; j < circumference_quads; ++j) {
                unsigned int vertex_index = 2 * lid_vertex_count + i * circumference_quads + j;
                positions[vertex_index] = positions[j+1];
                positions[vertex_index].y = lerp(radius, -radius, l);
            }
        }
    }

    { // Normals.
        Vector3f* normal_iterator = mesh.get_normals();
        while (normal_iterator < mesh.get_normals() + lid_vertex_count) // Top
            *normal_iterator++ = Vector3f(0, 1, 0);
        while (normal_iterator < mesh.get_normals() + 2 * lid_vertex_count) // Bottom
            *normal_iterator++ = Vector3f(0, -1, 0);
        Vector3f* side_position_iterator = mesh.get_positions() + 2 * lid_vertex_count;
        while (normal_iterator < mesh.get_normals() + vertex_count) { // Side
            Vector3f position = *side_position_iterator++;
            *normal_iterator++ = normalize(Vector3f(position.x, 0.0f, position.z));
        }
    }

    { // Tex coords.
        Vector2f* texcoords = mesh.get_texcoords();

        // Top and bottom.
        for (unsigned int i = 0; i < 2 * lid_vertex_count; ++i) {
            Vector3f position = mesh.get_positions()[i];
            texcoords[i] = Vector2f(position.x, position.z) + 0.5f;
        }

        // Side.
        for (unsigned int i = 0; i < vertical_quads + 1; ++i) {
            float v = i / float(vertical_quads);
            for (unsigned int j = 0; j < circumference_quads; ++j) {
                unsigned int vertex_index = 2 * lid_vertex_count + i * circumference_quads + j;
                float u = abs(-2.0f * j / float(circumference_quads) + 1.0f); // Magic u mapping. Mirror repeat mapping of the texture coords.
                texcoords[vertex_index] = Vector2f(u, v);
            }
        }
    }

    { // Primitives.
        Vector3ui* primitives = mesh.get_primitives();

        // Top.
        for (unsigned int i = 0; i < lid_index_count; ++i)
            primitives[i] = Vector3ui(0, i + 2, i + 1);
        primitives[lid_index_count - 1].y = 1;

        // Bottom.
        for (unsigned int i = 0; i < lid_index_count; ++i)
            primitives[i + lid_index_count] = Vector3ui(0, i + 1, i + 2) + lid_vertex_count;
        primitives[2 * lid_index_count - 1].z = 1 + lid_vertex_count;

        // Side.
        unsigned int side_vertex_offset = 2 * lid_vertex_count;
        for (unsigned int i = 0; i < vertical_quads; ++i) {
            for (unsigned int j = 0; j < circumference_quads; ++j) {
                unsigned int side_index = 2 * lid_index_count + 2 * (i * circumference_quads + j);

                unsigned int i0 = i * circumference_quads + j;
                unsigned int i1 = (i + 1) * circumference_quads + j;
                unsigned int j_plus_1 = (j + 1) < circumference_quads ? (j + 1) : 0; // Handle wrap around.
                unsigned int i2 = i * circumference_quads + j_plus_1;
                unsigned int i3 = (i + 1) * circumference_quads + j_plus_1;

                primitives[side_index + 0] = Vector3ui(i0, i3, i1) + side_vertex_offset;
                primitives[side_index + 1] = Vector3ui(i0, i2, i3) + side_vertex_offset;
            }
        }
    }

    mesh.set_bounds(AABB(Vector3f(-radius), Vector3f(radius)));

    return mesh.get_ID();
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
    unsigned int index_count = (quad_count - longitude_quads) * 2;
    float radius = 0.5f;

    Mesh mesh = Meshes::create("RevolvedSphere", index_count, vertex_count);

    { // Vertex attributes.
        Vector3f* positions = mesh.get_positions();
        Vector3f* normals = mesh.get_normals();
        Vector2f* texcoords = mesh.get_texcoords();

        Vector2f tc_normalizer = Vector2f(1.0f / longitude_quads, 1.0f / latitude_quads);
        for (unsigned int y = 0; y < latitude_size; ++y) {
            for (unsigned int x = 0; x < longitude_size; ++x) {
                unsigned int vertex_index = y * longitude_size + x;
                Vector2f tc = texcoords[vertex_index] = Vector2f(float(x), float(y)) * tc_normalizer;
                positions[vertex_index] = spherical_to_direction(tc.y * Math::PI<float>(),
                                                                        tc.x * 2.0f * Math::PI<float>()) * radius;
                normals[vertex_index] = normalize(positions[vertex_index]);
            }
        }

        // Hard set the poles to [0,1,0] and [0,-1,0].
        for (unsigned int x = 0; x < longitude_size; ++x) {
            positions[x] = Vector3f(0, radius, 0);
            positions[(latitude_size - 1) * longitude_size + x] = Vector3f(0, -radius, 0);
        }
    }

    { // Primitives.
        Vector3ui* primitives = mesh.get_primitives();
        for (unsigned int y = 0; y < latitude_quads; ++y) {
            for (unsigned int x = 0; x < longitude_quads; ++x) {
                unsigned int base_vertex_index = x + y * longitude_size;
                if (y != 0)
                    *primitives++ = Vector3ui(0, 1, longitude_size) + base_vertex_index;
                if (y != latitude_quads - 1)
                    *primitives++ = Vector3ui(1, longitude_size + 1, longitude_size) + base_vertex_index;
            }
        }
    }

    mesh.set_bounds(AABB(Vector3f(-radius), Vector3f(radius)));

    return mesh.get_ID();
}

} // NS MeshCreation
} // NS Assets
} // NS Cogwheel
