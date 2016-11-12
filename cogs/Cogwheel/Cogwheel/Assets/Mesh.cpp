// Cogwheel mesh.
// ---------------------------------------------------------------------------
// Copyright (C) 2015-2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#include <Cogwheel/Assets/Mesh.h>

#include <assert.h>
#include <cmath>

using namespace Cogwheel::Math;

namespace Cogwheel {
namespace Assets {

Meshes::UIDGenerator Meshes::m_UID_generator = UIDGenerator(0u);
std::string* Meshes::m_names = nullptr;
Meshes::Buffers* Meshes::m_buffers = nullptr;
AABB* Meshes::m_bounds = nullptr;

unsigned char* Meshes::m_changes = nullptr;
std::vector<Meshes::UID> Meshes::m_meshes_changed = std::vector<Meshes::UID>(0);;

void Meshes::allocate(unsigned int capacity) {
    if (is_allocated())
        return;

    m_UID_generator = UIDGenerator(capacity);
    capacity = m_UID_generator.capacity();

    m_names = new std::string[capacity];
    m_buffers = new Buffers[capacity];
    m_bounds = new AABB[capacity];
    m_changes = new unsigned char[capacity];
    std::memset(m_changes, Changes::None, capacity);

    m_meshes_changed.reserve(capacity / 4);

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
    delete[] m_changes; m_changes = nullptr;

    m_meshes_changed.resize(0); m_meshes_changed.shrink_to_fit();

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
    m_changes = resize_and_copy_array(m_changes, new_capacity, copyable_elements);
    if (copyable_elements < new_capacity) // We need to zero the new change masks.
        std::memset(m_changes + copyable_elements, Changes::None, new_capacity - copyable_elements);
}

void Meshes::reserve(unsigned int new_capacity) {
    unsigned int old_capacity = capacity();
    m_UID_generator.reserve(new_capacity);
    reserve_mesh_data(m_UID_generator.capacity(), old_capacity);
}

Meshes::UID Meshes::create(const std::string& name, unsigned int primitive_count, unsigned int vertex_count, unsigned char buffer_bitmask) {
    assert(m_buffers != nullptr);
    assert(m_names != nullptr);
    assert(m_bounds != nullptr);

    unsigned int old_capacity = m_UID_generator.capacity();
    UID id = m_UID_generator.generate();
    if (old_capacity != m_UID_generator.capacity())
        // The capacity has changed and the size of all arrays need to be adjusted.
        reserve_mesh_data(m_UID_generator.capacity(), old_capacity);

    if (m_changes[id] == Changes::None)
        m_meshes_changed.push_back(id);

    m_names[id] = name;
    m_buffers[id].primitive_count = primitive_count;
    m_buffers[id].primitives = new Math::Vector3ui[primitive_count];
    m_buffers[id].vertex_count = vertex_count;
    m_buffers[id].positions = (buffer_bitmask & MeshFlags::Position) ? new Math::Vector3f[vertex_count] : nullptr;
    m_buffers[id].normals = (buffer_bitmask & MeshFlags::Normal) ? new Math::Vector3f[vertex_count] : nullptr;
    m_buffers[id].texcoords = (buffer_bitmask & MeshFlags::Texcoord) ? new Math::Vector2f[vertex_count] : nullptr;
    m_bounds[id] = AABB::invalid();
    m_changes[id] = Changes::Created;

    return id;
}

void Meshes::destroy(Meshes::UID mesh_ID) {
    if (m_UID_generator.erase(mesh_ID)) {

        Buffers& buffers = m_buffers[mesh_ID];
        delete[] buffers.primitives;
        delete[] buffers.positions;
        delete[] buffers.normals;
        delete[] buffers.texcoords;

        if (m_changes[mesh_ID] == Changes::None)
            m_meshes_changed.push_back(mesh_ID);
        m_changes[mesh_ID] = Changes::Destroyed;
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

void Meshes::reset_change_notifications() {
    std::memset(m_changes, Changes::None, capacity());
    m_meshes_changed.resize(0);
}

//-----------------------------------------------------------------------------
// Mesh utils.
//-----------------------------------------------------------------------------

namespace MeshUtils {

    Meshes::UID combine(const std::string& name, 
                        const TransformedMesh* const meshes_begin, 
                        const TransformedMesh* const meshes_end, 
                        unsigned int mesh_flags) {
        
        auto meshes = Core::Iterable<const TransformedMesh* const>(meshes_begin, meshes_end);

        unsigned int primitive_count = 0u;
        unsigned int vertex_count = 0u;
        for (TransformedMesh transformed_mesh : meshes) {
            Mesh mesh = transformed_mesh.mesh_ID;
            primitive_count += mesh.get_primitive_count();
            vertex_count += mesh.get_vertex_count();
        }

        // Determine meshflags if none are given.
        if (mesh_flags == MeshFlags::None)
            mesh_flags = MeshFlags::AllBuffers;
            for (TransformedMesh transformed_mesh : meshes) {
                Mesh mesh = transformed_mesh.mesh_ID;
                mesh_flags &= mesh.get_mesh_flags();
            }

        Mesh merged_mesh = Mesh(Meshes::create(name, primitive_count, vertex_count, mesh_flags));

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

        if (mesh_flags & MeshFlags::Position) {
            Vector3f* positions = merged_mesh.get_positions();
            for (TransformedMesh transformed_mesh : meshes) {
                Mesh mesh = transformed_mesh.mesh_ID;
                for (Vector3f position : mesh.get_position_iterator())
                    *(positions++) = transformed_mesh.transform * position;
            }
        }

        if (mesh_flags & MeshFlags::Normal) {
            Vector3f* normals = merged_mesh.get_normals();
            for (TransformedMesh transformed_mesh : meshes) {
                Mesh mesh = transformed_mesh.mesh_ID;
                for (Vector3f normal : mesh.get_normal_iterator())
                    *(normals++) = transformed_mesh.transform.rotation * normal;
            }
        }

        if (mesh_flags & MeshFlags::Texcoord) {
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
        bool degenerate_positions = squared_magnitude(p0 - p1) < epsilon_squared ||
                                    squared_magnitude(p0 - p2) < epsilon_squared ||
                                    squared_magnitude(p1 - p2) < epsilon_squared;

        if (degenerate_indices || degenerate_positions)
            ++degenerate_primitives;
    }

    return degenerate_primitives;
}

} // NS MeshUtils

} // NS Assets
} // NS Cogwheel
