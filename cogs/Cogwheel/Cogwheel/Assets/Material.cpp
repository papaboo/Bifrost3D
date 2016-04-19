// Cogwheel rendered material.
// ---------------------------------------------------------------------------
// Copyright (C) 2015-2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#include <Cogwheel/Assets/Material.h>

#include <assert.h>

using namespace Cogwheel::Math;

namespace Cogwheel {
namespace Assets {

Materials::UIDGenerator Materials::m_UID_generator = UIDGenerator(0u);
std::string* Materials::m_names = nullptr;
Materials::Data* Materials::m_materials = nullptr;
unsigned char* Materials::m_changes = nullptr;
std::vector<Materials::UID> Materials::m_materials_changed = std::vector<Materials::UID>(0);

void Materials::allocate(unsigned int capacity) {
    if (is_allocated())
        return;

    m_UID_generator = UIDGenerator(capacity);
    capacity = m_UID_generator.capacity();

    m_names = new std::string[capacity];
    m_materials = new Data[capacity];
    m_changes = new unsigned char[capacity];
    std::memset(m_changes, 0, capacity);

    m_materials_changed.reserve(capacity / 4);

    // Allocate dummy element at 0.
    m_names[0] = "Dummy Material";
    Data dummy_data = {};
    dummy_data.base_tint = Math::RGB::red();
    m_materials[0] = dummy_data;
}

void Materials::deallocate() {
    if (!is_allocated())
        return;

    m_UID_generator = UIDGenerator(0u);
    delete[] m_names; m_names = nullptr;
    delete[] m_materials; m_materials = nullptr;
    delete[] m_changes; m_changes = nullptr;

    m_materials_changed.resize(0); m_materials_changed.shrink_to_fit();
}

template <typename T>
static inline T* resize_and_copy_array(T* old_array, unsigned int new_capacity, unsigned int copyable_elements) {
    T* new_array = new T[new_capacity];
    std::copy(old_array, old_array + copyable_elements, new_array);
    delete[] old_array;
    return new_array;
}

void Materials::reserve_material_data(unsigned int new_capacity, unsigned int old_capacity) {
    assert(m_names != nullptr);
    assert(m_materials != nullptr);

    const unsigned int copyable_elements = new_capacity < old_capacity ? new_capacity : old_capacity;
    m_names = resize_and_copy_array(m_names, new_capacity, copyable_elements);
    m_materials = resize_and_copy_array(m_materials, new_capacity, copyable_elements);
    m_changes = resize_and_copy_array(m_changes, new_capacity, copyable_elements);
    if (copyable_elements < new_capacity)
        // We need to zero the new change masks.
        std::memset(m_changes + copyable_elements, Changes::None, new_capacity - copyable_elements);
}

void Materials::reserve(unsigned int new_capacity) {
    unsigned int old_capacity = capacity();
    m_UID_generator.reserve(new_capacity);
    reserve_material_data(m_UID_generator.capacity(), old_capacity);
}

Materials::UID Materials::create(const std::string& name, const Data& data) {
    assert(m_names != nullptr);
    assert(m_materials != nullptr);
    assert(m_changes != nullptr);

    unsigned int old_capacity = m_UID_generator.capacity();
    UID id = m_UID_generator.generate();
    if (old_capacity != m_UID_generator.capacity())
        // The capacity has changed and the size of all arrays need to be adjusted.
        reserve_material_data(m_UID_generator.capacity(), old_capacity);

    // Push the material to the list of changed materials, if this is the first change.
    // (If a material at the same position has been created an deleted in the same frame, then this check would fail.)
    if (m_changes[id] == Changes::None)
        m_materials_changed.push_back(id);

    m_names[id] = name;
    m_materials[id] = data;
    m_changes[id] = Changes::Created;

    return id;
}

void Materials::destroy(Materials::UID material_ID) {
    if (m_UID_generator.erase(material_ID)) {
        if (m_changes[material_ID] == Changes::None)
            m_materials_changed.push_back(material_ID);
        m_changes[material_ID] |= Changes::Destroyed;
        // TODO If the material was created as well this frame, then remove the change notification entirely from the list of changed materials.
    }
}

void Materials::set_base_tint(Materials::UID material_ID, Math::RGB tint) {
    m_materials[material_ID].base_tint = tint;
    flag_as_changed(material_ID);
}

void Materials::set_base_roughness(Materials::UID material_ID, float roughness) {
    m_materials[material_ID].base_roughness = roughness;
    flag_as_changed(material_ID);
}

void Materials::set_specularity(Materials::UID material_ID, float incident_specularity) {
    m_materials[material_ID].specularity = incident_specularity;
    flag_as_changed(material_ID);
}

void Materials::set_metallic(Materials::UID material_ID, float metallic) {
    m_materials[material_ID].metallic = metallic;
    flag_as_changed(material_ID);
}

void Materials::flag_as_changed(Materials::UID material_ID) {
    if (has_changes(material_ID, Changes::Updated))
        m_materials_changed.push_back(material_ID);
    m_changes[material_ID] |= Changes::Updated;
}

void Materials::reset_change_notifications() {
    // NOTE We could use some heuristic here to choose between looping over 
    // the notifications and only resetting the changed materials instead of resetting all.
    std::memset(m_changes, Changes::None, capacity());
    m_materials_changed.resize(0);
}

} // NS Assets
} // NS Cogwheel
