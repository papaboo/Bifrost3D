// Bifrost model for meshes.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#include <Bifrost/Assets/MeshModel.h>

#include <assert.h>

namespace Bifrost {
namespace Assets {

MeshModels::UIDGenerator MeshModels::m_UID_generator = UIDGenerator(0u);
MeshModels::Model* MeshModels::m_models = nullptr;
Core::ChangeSet<MeshModels::Changes, MeshModels::UID> MeshModels::m_changes;

void MeshModels::allocate(unsigned int capacity) {
    if (is_allocated())
        return;

    m_UID_generator = UIDGenerator(capacity);
    capacity = m_UID_generator.capacity();

    m_models = new Model[capacity];
    m_changes = Core::ChangeSet<Changes, UID>(capacity);

    // Allocate dummy element at 0.
    m_models[0] = { Scene::SceneNodes::UID::invalid_UID(), Assets::Meshes::UID::invalid_UID() };
}

void MeshModels::deallocate() {
    if (!is_allocated())
        return;

    m_UID_generator = UIDGenerator(0u);
    delete[] m_models; m_models = nullptr;
    m_changes.resize(0);
}

template <typename T>
static inline T* resize_and_copy_array(T* old_array, unsigned int new_capacity, unsigned int copyable_elements) {
    T* new_array = new T[new_capacity];
    std::copy(old_array, old_array + copyable_elements, new_array);
    delete[] old_array;
    return new_array;
}

void MeshModels::reserve_model_data(unsigned int new_capacity, unsigned int old_capacity) {
    assert(m_models != nullptr);

    const unsigned int copyable_elements = new_capacity < old_capacity ? new_capacity : old_capacity;
    m_models = resize_and_copy_array(m_models, new_capacity, copyable_elements);
    m_changes.resize(new_capacity);
}

void MeshModels::reserve(unsigned int new_capacity) {
    unsigned int old_capacity = capacity();
    m_UID_generator.reserve(new_capacity);
    reserve_model_data(m_UID_generator.capacity(), old_capacity);
}

MeshModels::UID MeshModels::create(Scene::SceneNodes::UID scene_node_ID, Assets::Meshes::UID mesh_ID, Assets::Materials::UID material_ID) {
    assert(m_models != nullptr);
    assert(Scene::SceneNodes::has(scene_node_ID));
    assert(Assets::Meshes::has(mesh_ID));
    assert(Assets::Materials::has(material_ID));

    unsigned int old_capacity = m_UID_generator.capacity();
    UID id = m_UID_generator.generate();
    if (old_capacity != m_UID_generator.capacity())
        // The capacity has changed and the size of all arrays need to be adjusted.
        reserve_model_data(m_UID_generator.capacity(), old_capacity);

    m_models[id] = { scene_node_ID, mesh_ID, material_ID };
    m_changes.set_change(id, Change::Created);

    return id;
}

void MeshModels::destroy(MeshModels::UID model_ID) {
    if (m_UID_generator.erase(model_ID))
        m_changes.set_change(model_ID, Change::Destroyed);
}

void MeshModels::set_material_ID(MeshModels::UID model_ID, Assets::Materials::UID material_ID) {
    assert(has(model_ID));
    assert(Assets::Materials::has(material_ID));

    m_models[model_ID].material_ID = material_ID;
    m_changes.add_change(model_ID, Change::Material);
}

} // NS Assets
} // NS Bifrost
