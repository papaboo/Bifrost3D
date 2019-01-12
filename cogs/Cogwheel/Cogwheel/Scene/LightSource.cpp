// Cogwheel light source.
// ---------------------------------------------------------------------------
// Copyright (C) 2015-2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#include <Cogwheel/Scene/LightSource.h>

namespace Cogwheel {
namespace Scene {

LightSources::UIDGenerator LightSources::m_UID_generator = UIDGenerator(0u);

LightSources::Light* LightSources::m_lights = nullptr;

Core::ChangeSet<LightSources::Changes, LightSources::UID> LightSources::m_changes;

void LightSources::allocate(unsigned int capacity) {
    if (is_allocated())
        return;

    m_UID_generator = UIDGenerator(capacity);
    capacity = m_UID_generator.capacity();

    m_lights = new Light[capacity];

    m_changes = Core::ChangeSet<Changes, UID>(capacity);

    // Allocate dummy element at 0.
    m_lights[0].node_ID = SceneNodes::UID::invalid_UID();
    m_lights[0].type = LightSources::Type::Sphere;
    m_lights[0].color = Math::RGB(100000, 0, 100000);
    m_lights[0].sphere.radius = 0.0f;
}

void LightSources::deallocate() {
    if (!is_allocated())
        return;

    m_UID_generator = UIDGenerator(0u);

    delete[] m_lights; m_lights = nullptr;
    m_changes.resize(0);
}

template <typename T>
static inline T* resize_and_copy_array(T* old_array, unsigned int new_capacity, unsigned int copyable_elements) {
    T* new_array = new T[new_capacity];
    std::copy(old_array, old_array + copyable_elements, new_array);
    delete[] old_array;
    return new_array;
}

void LightSources::reserve_light_data(unsigned int new_capacity, unsigned int old_capacity) {
    assert(m_lights != nullptr);

    const unsigned int copyable_elements = new_capacity < old_capacity ? new_capacity : old_capacity;

    m_lights = resize_and_copy_array(m_lights, new_capacity, copyable_elements);
    m_changes.resize(new_capacity);
}

void LightSources::reserve(unsigned int new_capacity) {
    unsigned int old_capacity = capacity();
    m_UID_generator.reserve(new_capacity);
    reserve_light_data(m_UID_generator.capacity(), old_capacity);
}

LightSources::UID LightSources::create_sphere_light(SceneNodes::UID node_ID, Math::RGB power, float radius) {
    assert(m_lights != nullptr);

    if (!SceneNodes::has(node_ID))
        return LightSources::UID::invalid_UID();

    unsigned int old_capacity = m_UID_generator.capacity();
    UID id = m_UID_generator.generate();
    if (old_capacity != m_UID_generator.capacity())
        // The capacity has changed and the size of all arrays need to be adjusted.
        reserve_light_data(m_UID_generator.capacity(), old_capacity);

    m_lights[id].node_ID = node_ID;
    m_lights[id].type = LightSources::Type::Sphere;
    m_lights[id].color = power;
    m_lights[id].sphere.radius = radius;
    m_changes.set_change(id, Change::Created);

    return id;
}

LightSources::UID LightSources::create_directional_light(SceneNodes::UID node_ID, Math::RGB radiance) {
    assert(m_lights != nullptr);

    if (!SceneNodes::has(node_ID))
        return LightSources::UID::invalid_UID();

    unsigned int old_capacity = m_UID_generator.capacity();
    UID id = m_UID_generator.generate();
    if (old_capacity != m_UID_generator.capacity())
        // The capacity has changed and the size of all arrays need to be adjusted.
        reserve_light_data(m_UID_generator.capacity(), old_capacity);

    m_lights[id].node_ID = node_ID;
    m_lights[id].type = LightSources::Type::Directional;
    m_lights[id].color = radiance;
    m_changes.set_change(id, Change::Created);

    return id;
}

void LightSources::destroy(LightSources::UID light_ID) {
    // We don't actually destroy anything when destroying a light. 
    // The properties will get overwritten later when a node is created in same the spot.
    if (m_UID_generator.erase(light_ID)) 
        m_changes.set_change(light_ID, Change::Destroyed);
}

bool LightSources::is_delta_light(LightSources::UID light_ID) {
    switch (get_type(light_ID)) {
    case Type::Sphere:
        return is_delta_sphere_light(light_ID);
    case Type::Directional:
        return is_delta_directional_light(light_ID);
    }
    return false;
}

} // NS Scene
} // NS Cogwheel
