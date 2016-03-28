// Cogwheel mesh.
// ---------------------------------------------------------------------------
// Copyright (C) 2015-2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#include <Cogwheel/Assets/Mesh.h>

#include <assert.h>

using namespace Cogwheel::Math;

namespace Cogwheel {
namespace Assets {

Meshes::UIDGenerator Meshes::m_UID_generator = UIDGenerator(0u);
std::string* Meshes::m_names = nullptr;
Mesh* Meshes::m_meshes = nullptr;
AABB* Meshes::m_bounds = nullptr;

std::vector<Meshes::UID> Meshes::m_meshes_created = std::vector<Meshes::UID>(0);
std::vector<Meshes::UID> Meshes::m_meshes_destroyed = std::vector<Meshes::UID>(0);

void Meshes::allocate(unsigned int capacity) {
    if (is_allocated())
        return;

    m_UID_generator = UIDGenerator(capacity);
    capacity = m_UID_generator.capacity();

    m_names = new std::string[capacity];
    m_meshes = new Mesh[capacity];
    m_bounds = new AABB[capacity];

    m_meshes_created.reserve(capacity / 4);
    m_meshes_destroyed.reserve(capacity / 4);

    // Allocate dummy element at 0.
    m_names[0] = "Dummy Node";
    m_meshes[0] = Mesh();
    m_bounds[0] = AABB(Vector3f(1e30f, 1e30f, 1e30f), Vector3f(-1e30f, -1e30f, -1e30f));
}

void Meshes::deallocate() {
    if (!is_allocated())
        return;

    delete[] m_names; m_names = nullptr;
    for (Mesh& mesh : Core::Iterable<Mesh*>(m_meshes, m_meshes + capacity())) {
        delete[] mesh.indices;
        delete[] mesh.positions;
        delete[] mesh.normals;
        delete[] mesh.texcoords;
    }
    delete[] m_meshes; m_meshes = nullptr;
    delete[] m_bounds; m_bounds = nullptr;

    m_meshes_created.resize(0); m_meshes_created.shrink_to_fit();
    m_meshes_destroyed.resize(0); m_meshes_destroyed.shrink_to_fit();

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
    assert(m_meshes != nullptr);
    assert(m_bounds != nullptr);

    const unsigned int copyable_elements = new_capacity < old_capacity ? new_capacity : old_capacity;
    m_names = resize_and_copy_array(m_names, new_capacity, copyable_elements);

    Mesh* new_meshes = new Mesh[new_capacity];
    for (unsigned int i = 0; i < copyable_elements; ++i)
        new_meshes[i] = std::move(m_meshes[i]);
    delete[] m_meshes;
    m_meshes = new_meshes;

    m_bounds = resize_and_copy_array(m_bounds, new_capacity, copyable_elements);
}

void Meshes::reserve(unsigned int new_capacity) {
    unsigned int old_capacity = capacity();
    m_UID_generator.reserve(new_capacity);
    reserve_mesh_data(m_UID_generator.capacity(), old_capacity);
}

Meshes::UID Meshes::create(const std::string& name, unsigned int indices_count, unsigned int vertex_count, unsigned char buffer_bitmask) {
    assert(m_meshes != nullptr);
    assert(m_names != nullptr);
    assert(m_bounds != nullptr);

    unsigned int old_capacity = m_UID_generator.capacity();
    UID id = m_UID_generator.generate();
    if (old_capacity != m_UID_generator.capacity())
        // The capacity has changed and the size of all arrays need to be adjusted.
        reserve_mesh_data(m_UID_generator.capacity(), old_capacity);

    m_names[id] = name;
    m_meshes[id] = Mesh(indices_count, vertex_count, buffer_bitmask);
    m_bounds[id] = AABB(Vector3f(-1e30f, -1e30f, -1e30f), Vector3f(1e30f, 1e30f, 1e30f));

    m_meshes_created.push_back(id);

    return id;
}

void Meshes::destroy(Meshes::UID mesh_ID) {
    if (m_UID_generator.erase(mesh_ID))
        m_meshes_destroyed.push_back(mesh_ID);
}

AABB Meshes::compute_bounds(Meshes::UID mesh_ID) {
    Mesh& mesh = get_mesh(mesh_ID);

    AABB bounds = AABB(mesh.positions[0], mesh.positions[0]);
    for (Vector3f* position_itr = mesh.positions + 1; position_itr < (mesh.positions + mesh.vertex_count); ++position_itr) {
        bounds.grow_to_contain(*position_itr);
    }

    m_bounds[mesh_ID] = bounds;
    return bounds;
}

void Meshes::reset_change_notifications() {
    m_meshes_created.resize(0);
    m_meshes_destroyed.resize(0);
}

} // NS Assets
} // NS Cogwheel
