// Bifrost mesh.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#include <Bifrost/Assets/Mesh.h>

#include <Bifrost/Math/Conversions.h>

#include <assert.h>

using namespace Bifrost::Math;

namespace Bifrost {
namespace Assets {

Meshes::UIDGenerator Meshes::m_UID_generator = UIDGenerator(0u);
std::string* Meshes::m_names = nullptr;
Meshes::Buffers* Meshes::m_buffers = nullptr;
AABB* Meshes::m_bounds = nullptr;

Core::ChangeSet<Meshes::Changes, Meshes::UID> Meshes::m_changes;

void Meshes::allocate(unsigned int capacity) {
    if (is_allocated())
        return;

    m_UID_generator = UIDGenerator(capacity);
    capacity = m_UID_generator.capacity();

    m_names = new std::string[capacity];
    m_buffers = new Buffers[capacity];
    m_bounds = new AABB[capacity];
    m_changes = Core::ChangeSet<Changes, UID>(capacity);

    // Allocate dummy element at 0.
    m_names[0] = "Dummy Node";
    Buffers buffers = {};
    m_buffers[0] = buffers;
    m_bounds[0] = AABB(Vector3f(nanf("")), Vector3f(nanf("")));
}

void Meshes::deallocate() {
    if (!is_allocated())
        return;

    for (UID id : m_UID_generator) {
        Buffers& buffers = m_buffers[id];
        delete[] buffers.primitives;
        delete[] buffers.positions;
        delete[] buffers.normals;
        delete[] buffers.texcoords;
    }
    delete[] m_names; m_names = nullptr;
    delete[] m_buffers; m_buffers = nullptr;
    delete[] m_bounds; m_bounds = nullptr;
    
    m_changes.resize(0);

    m_UID_generator = UIDGenerator(0u);
}

template <typename T>
static inline T* resize_and_copy_array(T* old_array, unsigned int new_capacity, unsigned int copyable_elements) {
    T* new_array = new T[new_capacity];
    std::copy(old_array, old_array + copyable_elements, new_array);
    delete[] old_array;
    return new_array;
}

void Meshes::reserve_mesh_data(unsigned int new_capacity, unsigned int old_capacity) {
    assert(m_names != nullptr);
    assert(m_buffers != nullptr);
    assert(m_bounds != nullptr);

    const unsigned int copyable_elements = new_capacity < old_capacity ? new_capacity : old_capacity;
    m_names = resize_and_copy_array(m_names, new_capacity, copyable_elements);
    m_buffers = resize_and_copy_array(m_buffers, new_capacity, copyable_elements);
    m_bounds = resize_and_copy_array(m_bounds, new_capacity, copyable_elements);
    m_changes.resize(new_capacity);
}

void Meshes::reserve(unsigned int new_capacity) {
    unsigned int old_capacity = capacity();
    m_UID_generator.reserve(new_capacity);
    reserve_mesh_data(m_UID_generator.capacity(), old_capacity);
}

Meshes::UID Meshes::create(const std::string& name, unsigned int primitive_count, unsigned int vertex_count, MeshFlags buffer_bitmask) {
    assert(m_buffers != nullptr);
    assert(m_names != nullptr);
    assert(m_bounds != nullptr);

    unsigned int old_capacity = m_UID_generator.capacity();
    UID id = m_UID_generator.generate();
    if (old_capacity != m_UID_generator.capacity())
        // The capacity has changed and the size of all arrays need to be adjusted.
        reserve_mesh_data(m_UID_generator.capacity(), old_capacity);

    m_names[id] = name;
    m_buffers[id].primitive_count = primitive_count;
    m_buffers[id].primitives = new Vector3ui[primitive_count];
    m_buffers[id].vertex_count = vertex_count;
    m_buffers[id].positions = (buffer_bitmask & MeshFlag::Position) ? new Vector3f[vertex_count] : nullptr;
    m_buffers[id].normals = (buffer_bitmask & MeshFlag::Normal) ? new Vector3f[vertex_count] : nullptr;
    m_buffers[id].texcoords = (buffer_bitmask & MeshFlag::Texcoord) ? new Vector2f[vertex_count] : nullptr;
    m_bounds[id] = AABB::invalid();
    m_changes.set_change(id, Change::Created);

    return id;
}

void Meshes::destroy(Meshes::UID mesh_ID) {
    if (m_UID_generator.erase(mesh_ID)) {

        Buffers& buffers = m_buffers[mesh_ID];
        delete[] buffers.primitives;
        delete[] buffers.positions;
        delete[] buffers.normals;
        delete[] buffers.texcoords;

        m_changes.set_change(mesh_ID, Change::Destroyed);
    }
}

AABB Meshes::compute_bounds(Meshes::UID mesh_ID) {
    Buffers& buffers = m_buffers[mesh_ID];

    AABB bounds = AABB(buffers.positions[0], buffers.positions[0]);
    for (Vector3f* position_itr = buffers.positions + 1; position_itr < (buffers.positions + buffers.vertex_count); ++position_itr)
        bounds.grow_to_contain(*position_itr);

    m_bounds[mesh_ID] = bounds;
    return bounds;
}

//-----------------------------------------------------------------------------
// Mesh utils.
//-----------------------------------------------------------------------------

namespace MeshUtils {

Meshes::UID deep_clone(Meshes::UID mesh_ID) {
    Mesh mesh = mesh_ID;
    Meshes::UID new_ID = Meshes::create(mesh.get_name() + "_clone", mesh.get_primitive_count(), mesh.get_vertex_count(), mesh.get_flags());

    Vector3f* positions_begin = mesh.get_positions();
    if (positions_begin != nullptr)
        std::copy_n(positions_begin, mesh.get_vertex_count(), Meshes::get_positions(new_ID));

    Vector3f* normals_begin = mesh.get_normals();
    if (normals_begin != nullptr)
        std::copy_n(normals_begin, mesh.get_vertex_count(), Meshes::get_normals(new_ID));

    Vector2f* texcoords_begin = mesh.get_texcoords();
    if (texcoords_begin != nullptr)
        std::copy_n(texcoords_begin, mesh.get_vertex_count(), Meshes::get_texcoords(new_ID));

    Vector3ui* primitives_begin = mesh.get_primitives();
    if (primitives_begin != nullptr)
        std::copy_n(primitives_begin, mesh.get_primitive_count(), Meshes::get_primitives(new_ID));

    Meshes::set_bounds(new_ID, mesh.get_bounds());

    return new_ID;
}

void transform_mesh(Meshes::UID mesh_ID, Matrix3x4f affine_transform) {
    Mesh mesh = mesh_ID;

    Matrix3x3f rotation;
    rotation.set_column(0, affine_transform.get_column(0));
    rotation.set_column(1, affine_transform.get_column(1));
    rotation.set_column(2, affine_transform.get_column(2));
    Vector3f translation = affine_transform.get_column(3);

    // Transform positions.
    Vector3f* positions_itr = mesh.get_positions();
    if (positions_itr != nullptr) {
        AABB bounding_box = AABB(Vector3f(std::numeric_limits<float>::infinity()), Vector3f(-std::numeric_limits<float>::infinity()));
        Vector3f* positions_end = positions_itr + mesh.get_vertex_count();
        for (; positions_itr != positions_end; ++positions_itr) {
            *positions_itr = rotation * *positions_itr + translation;
            bounding_box.grow_to_contain(*positions_itr);
        }
        mesh.set_bounds(bounding_box);
    }

    // Transform normals.
    Vector3f* normals_itr = mesh.get_normals();
    if (normals_itr != nullptr) {
        Matrix3x3f normal_rotation = transpose(invert(rotation));
        Vector3f* normals_end = normals_itr + mesh.get_vertex_count();
        for (; normals_itr != normals_end; ++normals_itr)
            *normals_itr = normal_rotation * *normals_itr;
    }
}

void transform_mesh(Meshes::UID mesh_ID, Transform transform) {
    transform_mesh(mesh_ID, to_matrix3x4(transform));
}

Meshes::UID combine(const std::string& name,
                    const TransformedMesh* const meshes_begin,
                    const TransformedMesh* const meshes_end,
                    MeshFlags flags) {

    auto meshes = Core::Iterable<const TransformedMesh* const>(meshes_begin, meshes_end);

    unsigned int primitive_count = 0u;
    unsigned int vertex_count = 0u;
    for (TransformedMesh transformed_mesh : meshes) {
        Mesh mesh = transformed_mesh.mesh_ID;
        primitive_count += mesh.get_primitive_count();
        vertex_count += mesh.get_vertex_count();
    }

    // Determine shared buffers.
    for (TransformedMesh transformed_mesh : meshes) {
        Mesh mesh = transformed_mesh.mesh_ID;
        flags &= mesh.get_flags();
    }

    Mesh merged_mesh = Mesh(Meshes::create(name, primitive_count, vertex_count, flags));

    { // Always combine primitives.
        Vector3ui* primitives = merged_mesh.get_primitives();
        unsigned int primitive_offset = 0u;
        for (TransformedMesh transformed_mesh : meshes) {
            Mesh mesh = transformed_mesh.mesh_ID;
            for (Vector3ui primitive : mesh.get_primitive_iterable())
                *(primitives++) = primitive + primitive_offset;
            primitive_offset += mesh.get_vertex_count();
        }
    }

    if (flags & MeshFlag::Position) {
        Vector3f* positions = merged_mesh.get_positions();
        for (TransformedMesh transformed_mesh : meshes) {
            Mesh mesh = transformed_mesh.mesh_ID;
            for (Vector3f position : mesh.get_position_iterable())
                *(positions++) = transformed_mesh.transform * position;
        }
    }

    if (flags & MeshFlag::Normal) {
        Vector3f* normals = merged_mesh.get_normals();
        for (TransformedMesh transformed_mesh : meshes) {
            Mesh mesh = transformed_mesh.mesh_ID;
            for (Vector3f normal : mesh.get_normal_iterable())
                *(normals++) = transformed_mesh.transform.rotation * normal;
        }
    }

    if (flags & MeshFlag::Texcoord) {
        Vector2f* texcoords = merged_mesh.get_texcoords();
        for (TransformedMesh transformed_mesh : meshes) {
            Mesh mesh = transformed_mesh.mesh_ID;
            memcpy(texcoords, mesh.get_texcoords(), sizeof(Vector2f) * mesh.get_vertex_count());
            texcoords += mesh.get_vertex_count();
        }
    }

    merged_mesh.compute_bounds();

    return merged_mesh.get_ID();
}

void compute_hard_normals(Vector3f* positions_begin, Vector3f* positions_end, Vector3f* normals_begin) {
    while (positions_begin < positions_end) {
        Vector3f p0 = *positions_begin++;
        Vector3f p1 = *positions_begin++;
        Vector3f p2 = *positions_begin++;

        Vector3f normal = normalize(cross(p1 - p0, p2 - p0));
        *normals_begin++ = normal;
        *normals_begin++ = normal;
        *normals_begin++ = normal;
    }
}

void compute_normals(Vector3ui* primitives_begin, Vector3ui* primitives_end,
                     Vector3f* normals_begin, Vector3f* normals_end, Vector3f* positions_begin) {
    std::fill(normals_begin, normals_end, Vector3f::zero());

    while (primitives_begin < primitives_end) {
        Vector3ui primitive = *primitives_begin++;

        Vector3f p0 = positions_begin[primitive.x];
        Vector3f p1 = positions_begin[primitive.y];
        Vector3f p2 = positions_begin[primitive.z];

        Vector3f normal = cross(p1 - p0, p2 - p0);
        normals_begin[primitive.x] += normal;
        normals_begin[primitive.y] += normal;
        normals_begin[primitive.z] += normal;
    }

    std::for_each(normals_begin, normals_end, [](Vector3f& n) { n = normalize(n); });
}

void compute_normals(Meshes::UID mesh_ID) {
    Mesh mesh = mesh_ID;
    compute_normals(mesh.get_primitives(), mesh.get_primitives() + mesh.get_primitive_count(),
                    mesh.get_normals(), mesh.get_normals() + mesh.get_vertex_count(),
                    mesh.get_positions());
}

} // NS MeshUtils

namespace MeshTests {

unsigned int normals_correspond_to_winding_order(Meshes::UID mesh_ID) {
    Mesh mesh = mesh_ID;
    unsigned int failed_primitives = 0;

    for (Vector3ui primitive : mesh.get_primitive_iterable()) {
        Vector3f v0 = mesh.get_positions()[primitive.x];
        Vector3f v1 = mesh.get_positions()[primitive.y];
        Vector3f v2 = mesh.get_positions()[primitive.z];
        Vector3f primitive_normal = cross(v1 - v0, v2 - v0); // Not normalized, as we only care about the sign of the dot product below.

        bool primitive_failed = false;
        Vector3f n0 = mesh.get_normals()[primitive.x];
        if (dot(primitive_normal, n0) <= 0.0f) {
            // error_callback(mesh_ID, primitive_index, primitive.x);
            primitive_failed = true;
        }

        Vector3f n1 = mesh.get_normals()[primitive.y];
        if (dot(primitive_normal, n1) <= 0.0f) {
            // error_callback(mesh_ID, primitive_index, primitive.x);
            primitive_failed = true;
        }

        Vector3f n2 = mesh.get_normals()[primitive.z];
        if (dot(primitive_normal, n2) <= 0.0f) {
            // error_callback(mesh_ID, primitive_index, primitive.x);
            primitive_failed = true;
        }

        if (primitive_failed)
            ++failed_primitives;
    }

    return failed_primitives;
}

unsigned int count_degenerate_primitives(Meshes::UID mesh_ID, float epsilon_squared) {
    Mesh mesh = mesh_ID;
    unsigned int degenerate_primitives = 0;

    for (Vector3ui primitive : mesh.get_primitive_iterable()) {
        bool degenerate_indices = primitive.x == primitive.y ||
                                  primitive.x == primitive.z ||
                                  primitive.y == primitive.z;
        Vector3f p0 = mesh.get_positions()[primitive.x];
        Vector3f p1 = mesh.get_positions()[primitive.y];
        Vector3f p2 = mesh.get_positions()[primitive.z];
        bool degenerate_positions = magnitude_squared(p0 - p1) < epsilon_squared ||
                                    magnitude_squared(p0 - p2) < epsilon_squared ||
                                    magnitude_squared(p1 - p2) < epsilon_squared;

        if (degenerate_indices || degenerate_positions)
            ++degenerate_primitives;
    }

    return degenerate_primitives;
}

} // NS MeshUtils

} // NS Assets
} // NS Bifrost
