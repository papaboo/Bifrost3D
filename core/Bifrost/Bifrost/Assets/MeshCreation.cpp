// Bifrost mesh creation utilities.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#include <Bifrost/Assets/MeshCreation.h>
#include <Bifrost/Core/Array.h>
#include <Bifrost/Math/Constants.h>
#include <Bifrost/Math/Matrix.h>
#include <Bifrost/Math/Vector.h>
#include <Bifrost/Math/Utils.h>

using namespace Bifrost::Math;

namespace Bifrost {
namespace Assets {
namespace MeshCreation {

Mesh plane(unsigned int quads_per_edge, MeshFlags buffer_bitmask) {
    if (quads_per_edge == 0)
        return Mesh();

    unsigned int size = quads_per_edge + 1;
    unsigned int vertex_count = size * size;
    unsigned int quad_count = quads_per_edge * quads_per_edge;
    unsigned int index_count = quad_count * 2;
    
    Mesh mesh = Mesh("Plane", index_count, vertex_count, buffer_bitmask);

    // Vertex attributes.
    float tc_normalizer = 1.0f / quads_per_edge;
    for (unsigned int z = 0; z < size; ++z) {
        for (unsigned int x = 0; x < size; ++x) {
            Vector2f texcoord = Vector2f(float(x), float(z)) * tc_normalizer;
            mesh.get_positions()[z * size + x] = Vector3f(texcoord.x - 0.5f, 0.0f, texcoord.y - 0.5f);
            if (mesh.get_normals() != nullptr)
                mesh.get_normals()[z * size + x] = Vector3f(0.0f, 1.0f, 0.0f);
            if (mesh.get_texcoords() != nullptr)
                mesh.get_texcoords()[z * size + x] = texcoord;
        }
    }

    // Primitives.
    Vector3ui* primitives = mesh.get_primitives();
    for (unsigned int z = 0; z < quads_per_edge; ++z) {
        for (unsigned int x = 0; x < quads_per_edge; ++x) {
            unsigned int base_index = x + z * size;
            *primitives++ = Vector3ui(base_index, base_index + size, base_index + 1);
            *primitives++ = Vector3ui(base_index + 1, base_index + size, base_index + size + 1);
        }
    }

    Vector3f extends = Vector3f(0.5f, 0.0f, 0.5f);
    mesh.set_bounds(AABB(-extends, extends));

    return mesh;
}

Mesh box(unsigned int quads_per_edge, Vector3f size, MeshFlags buffer_bitmask) {
    if (quads_per_edge == 0)
        return Mesh();

    unsigned int sides = 6;

    unsigned int verts_per_edge = quads_per_edge + 1;
    Vector3f quad_size = size / float(quads_per_edge);
    Vector3f half_size = 0.5f * size;
    unsigned int quad_count = quads_per_edge * quads_per_edge * sides;
    unsigned int index_count = quad_count * 2;
    unsigned int verts_per_side = verts_per_edge * verts_per_edge;
    unsigned int vertex_count = verts_per_side * sides;

    Mesh mesh = Mesh("Box", index_count, vertex_count, buffer_bitmask);

    // Create the vertices.
    // [..TOP.. ..BOTTOM.. ..LEFT.. ..RIGHT.. ..FRONT.. ..BACK..]
    Vector3f* position_iterator = mesh.get_positions();
    for (unsigned int i = 0; i < verts_per_edge; ++i) // Top
        for (unsigned int j = 0; j < verts_per_edge; ++j)
            *position_iterator++ = Vector3f(half_size.x - i * quad_size.x, half_size.y, j * quad_size.z - half_size.z);
    for (unsigned int i = 0; i < verts_per_edge; ++i) // Bottom
        for (unsigned int j = 0; j < verts_per_edge; ++j)
            *position_iterator++ = Vector3f(half_size.x - i * quad_size.x, -half_size.y, half_size.z - j * quad_size.z);
    for (unsigned int i = 0; i < verts_per_edge; ++i) // Left
        for (unsigned int j = 0; j < verts_per_edge; ++j)
            *position_iterator++ = Vector3f(-half_size.x, half_size.y - i * quad_size.y, j * quad_size.z - half_size.z);
    for (unsigned int i = 0; i < verts_per_edge; ++i) // Right
        for (unsigned int j = 0; j < verts_per_edge; ++j)
            *position_iterator++ = Vector3f(half_size.x, i * quad_size.y - half_size.y, j * quad_size.z - half_size.z);
    for (unsigned int i = 0; i < verts_per_edge; ++i) // Front
        for (unsigned int j = 0; j < verts_per_edge; ++j)
            *position_iterator++ = Vector3f(half_size.x - i * quad_size.x, half_size.y - j * quad_size.y, half_size.z);
    for (unsigned int i = 0; i < verts_per_edge; ++i) // Back
        for (unsigned int j = 0; j < verts_per_edge; ++j)
            *position_iterator++ = Vector3f(i * quad_size.x - half_size.x, half_size.y - j * quad_size.y, -half_size.z);

    if (mesh.get_normals() != nullptr) {
        Vector3f* normal_iterator = mesh.get_normals();
        while (normal_iterator < mesh.get_normals() + verts_per_side) // Top
            *normal_iterator++ = Vector3f(0, 1, 0);
        while (normal_iterator < mesh.get_normals() + verts_per_side * 2) // Bottom
            *normal_iterator++ = Vector3f(0, -1, 0);
        while (normal_iterator < mesh.get_normals() + verts_per_side * 3) // Left
            *normal_iterator++ = Vector3f(-1, 0, 0);
        while (normal_iterator < mesh.get_normals() + verts_per_side * 4) // Right
            *normal_iterator++ = Vector3f(1, 0, 0);
        while (normal_iterator < mesh.get_normals() + verts_per_side * 5) // Front
            *normal_iterator++ = Vector3f(0, 0, 1);
        while (normal_iterator < mesh.get_normals() + verts_per_side * 6) // Back
            *normal_iterator++ = Vector3f(0, 0, -1);
    }

    if (mesh.get_texcoords() != nullptr) {
        Vector2f* texcoords = mesh.get_texcoords();
        float tc_normalizer = 1.0f / quads_per_edge;
        for (unsigned int i = 0; i < verts_per_edge; ++i)
            for (unsigned int j = 0; j < verts_per_edge; ++j)
                texcoords[i * verts_per_edge + j] =
                texcoords[i * verts_per_edge + j + verts_per_side] =
                texcoords[i * verts_per_edge + j + verts_per_side * 2] =
                texcoords[i * verts_per_edge + j + verts_per_side * 3] =
                texcoords[i * verts_per_edge + j + verts_per_side * 4] =
                texcoords[i * verts_per_edge + j + verts_per_side * 5] = Vector2f(float(i), float(j)) * tc_normalizer;
    }

    // Set indices.
    Vector3ui* primitives = mesh.get_primitives();
    for (unsigned int side_offset = 0; side_offset < vertex_count; side_offset += verts_per_side)
        for (unsigned int i = 0; i < quads_per_edge; ++i)
            for (unsigned int j = 0; j < quads_per_edge; ++j) {
                *primitives++ = Vector3ui(j + i * verts_per_edge,
                                          j + (i + 1) * verts_per_edge,
                                          j + 1 + i * verts_per_edge) + side_offset;

                *primitives++ = Vector3ui(j + 1 + i * verts_per_edge,
                                          j + (i + 1) * verts_per_edge,
                                          j + 1 + (i + 1) * verts_per_edge) + side_offset;
            }

    mesh.set_bounds(AABB(-half_size, half_size));

    return mesh;
}

Mesh beveled_box(unsigned int quads_per_side, float bevel_size, Math::Vector3f size, MeshFlags buffer_bitmask) {
    if (quads_per_side == 0)
        return Mesh();

    // Create a regular box with additional 6 quads per side to be used for beveling.
    const int bevel_quads = 3;
    unsigned int full_quads_per_side = quads_per_side + 2 * bevel_quads;
    Mesh mesh = box(full_quads_per_side, size, buffer_bitmask);

    // Relocate vertices to move the vertices used for beveling near the edges of the box.
    // A band of three quads are moved such that they are bevel_size from the edges of the box and
    // the rest of the vertices are moved such that they are uniformly distributed on the flat side of the box.
    // The beveling effect is created by all vertices having bevel_size distance to an 'inner box', thus giving the rounded edges.
    // The distance to the 'inner box' is trivially upheld for vertices on the flat sides.
    // The box' vertices are in the range [-size / 2, size / 2].
    Vector3f half_size = 0.5f * size;
    Vector3f initial_quad_size = size / float(full_quads_per_side);
    Vector3f bevel_vertex_threshold = half_size - initial_quad_size * bevel_quads;
    Vector3f side_quad_scale = (half_size - bevel_size) / bevel_vertex_threshold;
    AABB inner_box = AABB(-half_size + bevel_size, half_size - bevel_size);

    Vector3f* positions = mesh.get_positions();
    for (unsigned int v = 0; v < mesh.get_vertex_count(); ++v) {
        Vector3f& position = positions[v];

        // Move position
        auto move_coordinate = [&](int d) -> bool {
            bool is_bevel_pos = abs(position[d]) > bevel_vertex_threshold[d];
            if (is_bevel_pos) {
                // Move the vertices closest to the edges inside the beveling threshold.
                float sign = non_zero_sign(position[d]);
                float bevel_t = inverse_lerp(bevel_vertex_threshold[d], half_size[d], abs(position[d]));
                position[d] = sign * lerp(half_size[d] - bevel_size, half_size[d], bevel_t);
            } else
                // Move vertex position on the side of the box to uniformly fill in the space between the beveled edges.
                position[d] *= side_quad_scale[d];
            return is_bevel_pos;
        };

        bool bevel_vertex = move_coordinate(0);
        bevel_vertex |= move_coordinate(1);
        bevel_vertex |= move_coordinate(2);

        if (bevel_vertex) {
            // Compute direction from closest point on inner box and use that direction as the normal and to determine the position.
            Vector3f closest_point = inner_box.closest_point_on_surface(position);
            Vector3f direction = normalize(position - closest_point);
            position = closest_point + bevel_size * direction;
        }
    }

    if (mesh.get_normals() != nullptr)
        MeshUtils::compute_normals(mesh);

    return mesh;
}

Mesh cylinder(unsigned int vertical_quads, unsigned int circumference_quads, MeshFlags buffer_bitmask) {
    if (vertical_quads == 0 || circumference_quads == 0)
        return Mesh();

    unsigned int lid_vertex_count = circumference_quads + 1;
    unsigned int side_vertex_count = (vertical_quads + 1) * circumference_quads;
    unsigned int vertex_count = 2 * lid_vertex_count + side_vertex_count;
    unsigned int lid_index_count = circumference_quads;
    unsigned int side_index_count = 2 * vertical_quads * circumference_quads;
    unsigned int index_count = 2 * lid_index_count + side_index_count;
    float radius = 0.5f;

    Mesh mesh = Mesh("Cylinder", index_count, vertex_count, buffer_bitmask);

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

    if (mesh.get_normals() != nullptr) {
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

    if (mesh.get_texcoords() != nullptr) {
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

    return mesh;
}

static Vector3f spherical_to_direction(float theta, float phi) {
    float sinTheta = sin(theta);
    float z = sinTheta * cos(phi);
    float x = -sinTheta * sin(phi);
    float y = cos(theta);
    return Vector3f(x, y, z);
}

Mesh revolved_sphere(unsigned int longitude_quads, unsigned int latitude_quads, MeshFlags buffer_bitmask) {
    if (longitude_quads < 3 || latitude_quads < 2)
        return Mesh();

    unsigned int latitude_size = latitude_quads + 1;
    unsigned int longitude_size = longitude_quads + 1;
    unsigned int vertex_count = latitude_size * longitude_size;
    unsigned int quad_count = latitude_quads * longitude_quads;
    unsigned int index_count = (quad_count - longitude_quads) * 2;
    float radius = 0.5f;

    Mesh mesh = Mesh("RevolvedSphere", index_count, vertex_count, buffer_bitmask);

    { // Vertex attributes.
        Vector3f* positions = mesh.get_positions();
        Vector3f* normals = mesh.get_normals();
        Vector2f* texcoords = mesh.get_texcoords();

        Vector2f tc_normalizer = Vector2f(1.0f / longitude_quads, 1.0f / latitude_quads);
        for (unsigned int y = 0; y < latitude_size; ++y) {
            for (unsigned int x = 0; x < longitude_size; ++x) {
                unsigned int vertex_index = y * longitude_size + x;
                Vector2f tc = Vector2f(float(x), float(y)) * tc_normalizer;
                if (texcoords)
                    texcoords[vertex_index] = tc;
                positions[vertex_index] = spherical_to_direction(tc.y * Math::PI<float>(),
                                                                 tc.x * 2.0f * Math::PI<float>()) * radius;
                if (normals)
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

    return mesh;
}

Mesh spherical_box(unsigned int quads_per_edge, MeshFlags buffer_bitmask) {
    Mesh mesh = box(quads_per_edge, Vector3f::one(), buffer_bitmask);

    Vector3f* positions = mesh.get_positions();
    Vector3f* normals = mesh.get_normals();

    for (unsigned int i = 0; i < mesh.get_vertex_count(); i++) {
        Vector3f normal = normalize(positions[i]);
        positions[i] = 0.5f * normal;
        normals[i] = normal;
    }

    return mesh;
}

Mesh torus(unsigned int revolution_quads, unsigned int circumference_quads, float minor_radius, MeshFlags buffer_bitmask) {
    if (revolution_quads == 0 || circumference_quads == 0)
        return Mesh();

    if (minor_radius <= 0.0f || minor_radius >= 0.5f)
    {
        minor_radius = clamp(minor_radius, next_float(0.0f), previous_float(0.5f));
        printf("MeshCreation::torus: minor_radius must be in range ]0, 0.5[, otherwise the surface normals and hard normals are inconsistent.");
    }

    unsigned int revolution_vertex_count = (revolution_quads + 1);
    unsigned int circumference_vertex_count = (circumference_quads + 1);
    unsigned int vertex_count = (revolution_quads + 1) * (circumference_quads + 1);
    unsigned int index_count = 2 * revolution_quads * circumference_quads;
    float major_radius = 0.5f;

    Mesh mesh = Mesh("Torus", index_count, vertex_count, buffer_bitmask);

    // Precompute local normal directions.
    Core::Array<Vector3f> local_normal_dirs(circumference_vertex_count);
    for (unsigned int x = 0; x < circumference_vertex_count-1; ++x) {
        float minor_radians = x / float(circumference_quads) * 2.0f * Math::PI<float>();
        local_normal_dirs[x] = Vector3f(cos(minor_radians), 0.0, sin(minor_radians));
    }
    local_normal_dirs[circumference_vertex_count - 1] = local_normal_dirs[0];

    // Vertex attributes.
    Vector2f tc_normalizer = Vector2f(1.0f / circumference_quads, 1.0f / revolution_quads);
    for (unsigned int z = 0; z < revolution_vertex_count; ++z) {
        // Create local coordinate system on the torus.
        float major_radians = z / float(revolution_quads) * 2.0f * Math::PI<float>();
        Vector3f center = Vector3f(cos(major_radians) * major_radius, 0.0, sin(major_radians) * major_radius);
        if (z == 0 || z == revolution_vertex_count - 1)
            center = Vector3f(major_radius, 0.0, 0.0f);

        Vector3f outward = normalize(center);
        Vector3f up = Vector3f::up();
        Vector3f tangent = cross(outward, up);
        Matrix3x3f local_coords = { outward, tangent, up };

        for (unsigned int x = 0; x < circumference_vertex_count; ++x) {
            unsigned int vertex_index = z * circumference_vertex_count + x;

            Vector3f normal_dir = local_normal_dirs[x] * local_coords;

            if (mesh.get_normals() != nullptr)
                mesh.get_normals()[vertex_index] = normal_dir;

            Vector3f offset = normal_dir * minor_radius;
            mesh.get_positions()[vertex_index] = center + offset;

            if (mesh.get_texcoords() != nullptr)
                mesh.get_texcoords()[vertex_index] = Vector2f(float(x), float(z)) * tc_normalizer;
        }
    }

    // Create primitives.
    Vector3ui* primitives = mesh.get_primitives();
    for (unsigned int z = 0; z < revolution_quads; ++z) {
        for (unsigned int x = 0; x < circumference_quads; ++x) {
            unsigned int base_index = x + z * circumference_vertex_count;
            unsigned int next_base_index = base_index + circumference_vertex_count;
            *primitives++ = Vector3ui(base_index, base_index + 1, next_base_index);
            *primitives++ = Vector3ui(base_index + 1, next_base_index + 1, next_base_index);
        }
    }

    // Set bounds.
    Vector3f max_corner = Vector3f(major_radius + minor_radius, minor_radius, major_radius + minor_radius);
    mesh.set_bounds(AABB(-max_corner, max_corner));

    return mesh;
}

} // NS MeshCreation
} // NS Assets
} // NS Bifrost
