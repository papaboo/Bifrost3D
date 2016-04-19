// Cogwheel light source.
// ---------------------------------------------------------------------------
// Copyright (C) 2015-2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#include <Cogwheel/Scene/LightSource.h>

using namespace Cogwheel::Math;

namespace Cogwheel {
namespace Scene {

LightSources::UIDGenerator LightSources::m_UID_generator = UIDGenerator(0u);

SceneNodes::UID* LightSources::m_node_IDs = nullptr;
RGB* LightSources::m_power = nullptr;
float* LightSources::m_radius = nullptr;

unsigned char* LightSources::m_changes = nullptr;
std::vector<LightSources::UID> LightSources::m_lights_changed = std::vector<LightSources::UID>(0);

void LightSources::allocate(unsigned int capacity) {
    if (is_allocated())
        return;

    m_UID_generator = UIDGenerator(capacity);
    capacity = m_UID_generator.capacity();

    m_node_IDs = new SceneNodes::UID[capacity];
    m_power = new RGB[capacity];
    m_radius = new float[capacity];

    m_changes = new unsigned char[capacity];
    std::memset(m_changes, Changes::None, capacity);
    m_lights_changed.reserve(capacity / 4);

    // Allocate dummy element at 0.
    m_node_IDs[0] = SceneNodes::UID::invalid_UID();
    m_power[0] = RGB::black();
    m_radius[0] = 0.0f;
}

void LightSources::deallocate() {
    if (!is_allocated())
        return;

    m_UID_generator = UIDGenerator(0u);

    delete[] m_node_IDs; m_node_IDs = nullptr;
    delete[] m_power; m_power = nullptr;
    delete[] m_radius; m_radius = nullptr;
    delete[] m_changes; m_changes = nullptr;

    m_lights_changed.resize(0); m_lights_changed.shrink_to_fit();
}

template <typename T>
static inline T* resize_and_copy_array(T* old_array, unsigned int new_capacity, unsigned int copyable_elements) {
    T* new_array = new T[new_capacity];
    std::copy(old_array, old_array + copyable_elements, new_array);
    delete[] old_array;
    return new_array;
}

void LightSources::reserve_light_data(unsigned int new_capacity, unsigned int old_capacity) {
    assert(m_node_IDs != nullptr);
    assert(m_power != nullptr);
    assert(m_radius != nullptr);

    const unsigned int copyable_elements = new_capacity < old_capacity ? new_capacity : old_capacity;

    m_node_IDs = resize_and_copy_array(m_node_IDs, new_capacity, copyable_elements);
    m_power = resize_and_copy_array(m_power, new_capacity, copyable_elements);
    m_radius = resize_and_copy_array(m_radius, new_capacity, copyable_elements);
    m_changes = resize_and_copy_array(m_changes, new_capacity, copyable_elements);
    if (copyable_elements < new_capacity)
        // We need to zero the new change masks.
        std::memset(m_changes + copyable_elements, Changes::None, new_capacity - copyable_elements);
}

void LightSources::reserve(unsigned int new_capacity) {
    unsigned int old_capacity = capacity();
    m_UID_generator.reserve(new_capacity);
    reserve_light_data(m_UID_generator.capacity(), old_capacity);
}

LightSources::UID LightSources::create_sphere_light(SceneNodes::UID node_ID, Math::RGB power, float radius) {
    assert(m_node_IDs != nullptr);
    assert(m_power != nullptr);
    assert(m_radius != nullptr);

    unsigned int old_capacity = m_UID_generator.capacity();
    UID id = m_UID_generator.generate();
    if (old_capacity != m_UID_generator.capacity())
        // The capacity has changed and the size of all arrays need to be adjusted.
        reserve_light_data(m_UID_generator.capacity(), old_capacity);

    if (m_changes[id] == Changes::None)
        m_lights_changed.push_back(id);

    m_node_IDs[id] = node_ID;
    m_power[id] = power;
    m_radius[id] = radius;
    m_changes[id] = Changes::Created;

    return id;
}

void LightSources::destroy(LightSources::UID light_ID) {
    // We don't actually destroy anything when destroying a light. The properties will get overwritten later when a node is created in same the spot.
    if (m_UID_generator.erase(light_ID)) {
        if (m_changes[light_ID] == Changes::None)
            m_lights_changed.push_back(light_ID);

        m_changes[light_ID] |= Changes::Destroyed;
    }
}

void LightSources::reset_change_notifications() {
    std::memset(m_changes, Changes::None, capacity());
    m_lights_changed.resize(0);
}

} // NS Scene
} // NS Cogwheel
