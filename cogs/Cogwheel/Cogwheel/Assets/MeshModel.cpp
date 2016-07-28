// Cogwheel model for meshes.
// ---------------------------------------------------------------------------
// Copyright (C) 2015-2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#include <Cogwheel/Assets/MeshModel.h>

#include <assert.h>

namespace Cogwheel {
namespace Assets {

MeshModels::UIDGenerator MeshModels::m_UID_generator = UIDGenerator(0u);
MeshModels::Model* MeshModels::m_models = nullptr;
unsigned char* MeshModels::m_changes = nullptr;
std::vector<MeshModels::UID> MeshModels::m_models_changed = std::vector<MeshModels::UID>(0);

void MeshModels::allocate(unsigned int capacity) {
    if (is_allocated())
        return;

    m_UID_generator = UIDGenerator(capacity);
    capacity = m_UID_generator.capacity();

    m_models = new Model[capacity];
    m_changes = new unsigned char[capacity];
    std::memset(m_changes, Changes::None, capacity);
    
    m_models_changed.reserve(capacity / 4);

    // Allocate dummy element at 0.
    m_models[0] = { Scene::SceneNodes::UID::invalid_UID(), Assets::Meshes::UID::invalid_UID() };
}

void MeshModels::deallocate() {
    if (!is_allocated())
        return;

    m_UID_generator = UIDGenerator(0u);
    delete[] m_models; m_models = nullptr;
    delete[] m_changes; m_changes = nullptr;
    m_models_changed.resize(0); m_models_changed.shrink_to_fit();
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
    assert(m_changes != nullptr);

    const unsigned int copyable_elements = new_capacity < old_capacity ? new_capacity : old_capacity;
    m_models = resize_and_copy_array(m_models, new_capacity, copyable_elements);
    m_changes = resize_and_copy_array(m_changes, new_capacity, copyable_elements);
    if (copyable_elements < new_capacity)
        // We need to zero the new change masks.
        std::memset(m_changes + copyable_elements, Changes::None, new_capacity - copyable_elements);
}

void MeshModels::reserve(unsigned int new_capacity) {
    unsigned int old_capacity = capacity();
    m_UID_generator.reserve(new_capacity);
    reserve_model_data(m_UID_generator.capacity(), old_capacity);
}

bool MeshModels::has(MeshModels::UID model_ID) { 
    return m_UID_generator.has(model_ID) && !(m_changes[model_ID] & Changes::Destroyed);
}

MeshModels::UID MeshModels::create(Scene::SceneNodes::UID scene_node_ID, Assets::Meshes::UID mesh_ID, Assets::Materials::UID material_ID) {
    assert(m_models != nullptr);

    unsigned int old_capacity = m_UID_generator.capacity();
    UID id = m_UID_generator.generate();
    if (old_capacity != m_UID_generator.capacity())
        // The capacity has changed and the size of all arrays need to be adjusted.
        reserve_model_data(m_UID_generator.capacity(), old_capacity);

    if (m_changes[id] == Changes::None)
        m_models_changed.push_back(id);

    m_models[id] = { scene_node_ID, mesh_ID, material_ID };
    m_changes[id] = Changes::Created;

    return id;
}

void MeshModels::destroy(MeshModels::UID model_ID) {
    if (m_UID_generator.has(model_ID)) {
        unsigned char& changes = m_changes[model_ID];
        
        if (changes == Changes::None)
            m_models_changed.push_back(model_ID);

        changes |= Changes::Destroyed;
    }
}

void MeshModels::reset_change_notifications() {
    for (UID model_ID : m_models_changed)
        if (has_changes(model_ID, Changes::Destroyed))
            m_UID_generator.erase(model_ID);

    std::memset(m_changes, Changes::None, capacity());
    m_models_changed.resize(0);
}

} // NS Assets
} // NS Cogwheel