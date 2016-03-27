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
unsigned char* Materials::m_events = nullptr;

std::vector<Materials::UID> Materials::m_materials_created = std::vector<Materials::UID>(0);
std::vector<Materials::UID> Materials::m_materials_destroyed = std::vector<Materials::UID>(0);
std::vector<Materials::UID> Materials::m_materials_changed = std::vector<Materials::UID>(0);

void Materials::allocate(unsigned int capacity) {
    if (is_allocated())
        return;

    m_UID_generator = UIDGenerator(capacity);
    capacity = m_UID_generator.capacity();

    m_names = new std::string[capacity];
    m_materials = new Data[capacity];
    m_events = new unsigned char[capacity];

    m_materials_created.reserve(capacity / 4);
    m_materials_destroyed.reserve(capacity / 4);
    m_materials_changed.reserve(capacity / 4);

    // Allocate dummy element at 0.
    m_names[0] = "Dummy Material";
    Data dummy_data = {};
    dummy_data.base_color = Math::RGB::red();
    m_materials[0] = dummy_data;
}

void Materials::deallocate() {
    if (!is_allocated())
        return;

    m_UID_generator = UIDGenerator(0u);
    delete[] m_names; m_names = nullptr;
    delete[] m_materials; m_materials = nullptr;
    delete[] m_events; m_events = nullptr;

    m_materials_created.resize(0); m_materials_created.shrink_to_fit();
    m_materials_destroyed.resize(0); m_materials_destroyed.shrink_to_fit();
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
    m_events = resize_and_copy_array(m_events, new_capacity, copyable_elements);
}

void Materials::reserve(unsigned int new_capacity) {
    unsigned int old_capacity = capacity();
    m_UID_generator.reserve(new_capacity);
    reserve_material_data(m_UID_generator.capacity(), old_capacity);
}

Materials::UID Materials::create(const std::string& name, const Data& data) {
    assert(m_names != nullptr);
    assert(m_materials != nullptr);
    assert(m_events != nullptr);

    unsigned int old_capacity = m_UID_generator.capacity();
    UID id = m_UID_generator.generate();
    if (old_capacity != m_UID_generator.capacity())
        // The capacity has changed and the size of all arrays need to be adjusted.
        reserve_material_data(m_UID_generator.capacity(), old_capacity);

    m_names[id] = name;
    m_materials[id] = data;
    m_events[id] = Events::Created;

    m_materials_created.push_back(id);

    return id;
}

void Materials::destroy(Materials::UID material_ID) {
    if (m_UID_generator.erase(material_ID)) {
        m_events[material_ID] |= Events::Destroyed;
        m_materials_destroyed.push_back(material_ID);
        // TODO If material has been created this frame as well then remove that notification.
    }
}

void Materials::set_base_color(Materials::UID material_ID, Math::RGB color) {
    m_materials[material_ID].base_color = color;
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
    if ((m_events[material_ID] & Events::Changed) != Events::Changed) {
        m_events[material_ID] |= Events::Changed;
        m_materials_changed.push_back(material_ID);
    }
}

void Materials::reset_change_notifications() {
    // NOTE We could use some heuristic here to choose between looping over 
    // the notifications and only resetting the changed events.
    std::memset(m_events, 0, capacity());

    m_materials_created.resize(0);
    m_materials_destroyed.resize(0);
    m_materials_changed.resize(0);
}

} // NS Assets
} // NS Cogwheel
