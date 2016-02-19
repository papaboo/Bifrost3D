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
MeshModel* MeshModels::m_models = nullptr;

std::vector<MeshModels::UID> MeshModels::m_models_created = std::vector<MeshModels::UID>(0);
std::vector<MeshModels::UID> MeshModels::m_models_destroyed = std::vector<MeshModels::UID>(0);

void MeshModels::allocate(unsigned int capacity) {
    if (is_allocated())
        return;

    m_UID_generator = UIDGenerator(capacity);
    capacity = m_UID_generator.capacity();

    m_models = new MeshModel[capacity];

    m_models_created.reserve(capacity / 4);
    m_models_destroyed.reserve(capacity / 4);

    // Allocate dummy element at 0.
    m_models[0] = { Scene::SceneNodes::UID::invalid_UID(), Assets::Meshes::UID::invalid_UID() };
}

void MeshModels::deallocate() {
    if (!is_allocated())
        return;

    m_UID_generator = UIDGenerator(0u);
    delete[] m_models; m_models = nullptr;

    m_models_created.resize(0); m_models_created.shrink_to_fit();
    m_models_destroyed.resize(0); m_models_destroyed.shrink_to_fit();
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
}

void MeshModels::reserve(unsigned int new_capacity) {
    unsigned int old_capacity = capacity();
    m_UID_generator.reserve(new_capacity);
    reserve_model_data(m_UID_generator.capacity(), old_capacity);
}

bool MeshModels::has(MeshModels::UID model_ID) { 
    return m_UID_generator.has(model_ID) && !(m_models[model_ID].properties & MeshModelProperties::Destroyed);
}

MeshModels::UID MeshModels::create(Scene::SceneNodes::UID scene_node_ID, Assets::Meshes::UID mesh_ID) {
    assert(m_models != nullptr);

    unsigned int old_capacity = m_UID_generator.capacity();
    UID id = m_UID_generator.generate();
    if (old_capacity != m_UID_generator.capacity())
        // The capacity has changed and the size of all arrays need to be adjusted.
        reserve_model_data(m_UID_generator.capacity(), old_capacity);

    m_models[id] = { scene_node_ID, mesh_ID };

    m_models_created.push_back(id);

    return id;
}

void MeshModels::destroy(MeshModels::UID model_ID) {
    MeshModel& model = m_models[model_ID];
    if (!(model.properties & MeshModelProperties::Destroyed) && m_UID_generator.has(model_ID)) {
        model.properties |= MeshModelProperties::Destroyed;
        m_models_destroyed.push_back(model_ID);
    }
}

void MeshModels::reset_change_notifications() {
    for (UID model_ID : m_models_destroyed)
        m_UID_generator.erase(model_ID);
    m_models_created.resize(0);
    m_models_destroyed.resize(0);
}

} // NS Assets
} // NS Cogwheel