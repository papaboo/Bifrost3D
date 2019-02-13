// Bifrost rendered material.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#include <Bifrost/Assets/Material.h>

#include <assert.h>

using namespace Bifrost::Math;

namespace Bifrost {
namespace Assets {

Materials::UIDGenerator Materials::m_UID_generator = UIDGenerator(0u);
std::string* Materials::m_names = nullptr;
Materials::Data* Materials::m_materials = nullptr;
Core::ChangeSet<Materials::Changes, Materials::UID> Materials::m_changes;

void Materials::allocate(unsigned int capacity) {
    if (is_allocated())
        return;

    m_UID_generator = UIDGenerator(capacity);
    capacity = m_UID_generator.capacity();

    m_names = new std::string[capacity];
    m_materials = new Data[capacity];
    m_changes = Core::ChangeSet<Changes, UID>(capacity);

    // Allocate dummy element at 0.
    m_names[0] = "Dummy Material";
    Data dummy_data = {};
    dummy_data.coverage = 1.0f;
    dummy_data.tint = Math::RGB::red();
    m_materials[0] = dummy_data;
}

void Materials::deallocate() {
    if (!is_allocated())
        return;

    m_UID_generator = UIDGenerator(0u);
    delete[] m_names; m_names = nullptr;
    delete[] m_materials; m_materials = nullptr;

    m_changes.resize(0);
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
    m_changes.resize(new_capacity);
}

void Materials::reserve(unsigned int new_capacity) {
    unsigned int old_capacity = capacity();
    m_UID_generator.reserve(new_capacity);
    reserve_material_data(m_UID_generator.capacity(), old_capacity);
}

Materials::UID Materials::create(const std::string& name, const Data& data) {
    assert(m_names != nullptr);
    assert(m_materials != nullptr);

    unsigned int old_capacity = m_UID_generator.capacity();
    UID id = m_UID_generator.generate();
    if (old_capacity != m_UID_generator.capacity())
        // The capacity has changed and the size of all arrays need to be adjusted.
        reserve_material_data(m_UID_generator.capacity(), old_capacity);

    m_names[id] = name;
    m_materials[id] = data;
    m_changes.set_change(id, Change::Created);

    return id;
}

void Materials::destroy(Materials::UID material_ID) {
    if (m_UID_generator.erase(material_ID))
        m_changes.set_change(material_ID, Change::Destroyed);
}

void Materials::set_flags(Materials::UID material_ID, Flags flags) {
    m_materials[material_ID].flags = flags;
    flag_as_updated(material_ID);
}

void Materials::set_tint(Materials::UID material_ID, Math::RGB tint) {
    m_materials[material_ID].tint = tint;
    flag_as_updated(material_ID);
}

void Materials::set_tint_roughness_texture_ID(Materials::UID material_ID, Textures::UID tint_roughness_texture_ID) {
    m_materials[material_ID].tint_roughness_texture_ID = tint_roughness_texture_ID;
    flag_as_updated(material_ID);
}

void Materials::set_roughness(Materials::UID material_ID, float roughness) {
    m_materials[material_ID].roughness = roughness;
    flag_as_updated(material_ID);
}

void Materials::set_specularity(Materials::UID material_ID, float incident_specularity) {
    m_materials[material_ID].specularity = incident_specularity;
    flag_as_updated(material_ID);
}

void Materials::set_metallic(Materials::UID material_ID, float metallic) {
    m_materials[material_ID].metallic = metallic;
    flag_as_updated(material_ID);
}

void Materials::set_metallic_texture_ID(Materials::UID material_ID, Textures::UID metallic_texture_ID) {
    m_materials[material_ID].metallic_texture_ID = metallic_texture_ID;
    flag_as_updated(material_ID);
}

void Materials::set_coverage(Materials::UID material_ID, float coverage) {
    m_materials[material_ID].coverage = coverage;
    flag_as_updated(material_ID);
}

void Materials::set_coverage_texture_ID(Materials::UID material_ID, Textures::UID coverage_texture_ID) {
    m_materials[material_ID].coverage_texture_ID = coverage_texture_ID;
    flag_as_updated(material_ID);
}

void Materials::set_transmission(Materials::UID material_ID, float transmission) {
    m_materials[material_ID].transmission = transmission;
    flag_as_updated(material_ID);
}

void Materials::flag_as_updated(Materials::UID material_ID) {
    m_changes.add_change(material_ID, Change::Updated);
}

} // NS Assets
} // NS Bifrost
