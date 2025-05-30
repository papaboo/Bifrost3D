// Bifrost light source.
// ------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#include <Bifrost/Scene/LightSource.h>

using namespace Bifrost::Math;

namespace Bifrost {
namespace Scene {

LightSourceIDGenerator LightSources::m_UID_generator = LightSourceIDGenerator(0u);

LightSources::Light* LightSources::m_lights = nullptr;

Core::ChangeSet<LightSources::Changes, LightSourceID> LightSources::m_changes;

void LightSources::allocate(unsigned int capacity) {
    if (is_allocated())
        return;

    m_UID_generator = LightSourceIDGenerator(capacity);
    capacity = m_UID_generator.capacity();

    m_lights = new Light[capacity];

    m_changes = Core::ChangeSet<Changes, LightSourceID>(capacity);

    // Allocate dummy element at 0.
    m_lights[0].node_ID = SceneNodeID::invalid_UID();
    m_lights[0].type = LightSources::Type::Sphere;
    m_lights[0].color = Math::RGB(100000, 0, 100000);
    m_lights[0].sphere.radius = 0.0f;
}

void LightSources::deallocate() {
    if (!is_allocated())
        return;

    m_UID_generator = LightSourceIDGenerator(0u);

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

inline LightSourceID LightSources::create_light(SceneNodeID node_ID, LightSources::Light light) {
    assert(m_lights != nullptr);

    if (!SceneNodes::has(node_ID))
        return LightSourceID::invalid_UID();

    unsigned int old_capacity = m_UID_generator.capacity();
    LightSourceID id = m_UID_generator.generate();
    if (old_capacity != m_UID_generator.capacity())
        // The capacity has changed and the size of all arrays need to be adjusted.
        reserve_light_data(m_UID_generator.capacity(), old_capacity);

    m_lights[id] = light;
    m_changes.set_change(id, Change::Created);

    return id;
}

LightSourceID LightSources::create_sphere_light(SceneNodeID node_ID, Math::RGB power, float radius) {
    LightSources::Light light = {};
    light.node_ID = node_ID;
    light.type = LightSources::Type::Sphere;
    light.color = power;
    light.sphere.radius = radius;
    return create_light(node_ID, light);
}

LightSourceID LightSources::create_spot_light(SceneNodeID node_ID, Math::RGB power, float radius, float cos_angle) {
    LightSources::Light light = {};
    light.node_ID = node_ID;
    light.type = LightSources::Type::Spot;
    light.color = power;
    light.spot.radius = half(radius);
    light.spot.cos_angle = unsigned short(cos_angle * USHRT_MAX + 0.5f);

    return create_light(node_ID, light);
}

LightSourceID LightSources::create_directional_light(SceneNodeID node_ID, Math::RGB radiance) {
    LightSources::Light light = {};
    light.node_ID = node_ID;
    light.type = LightSources::Type::Directional;
    light.color = radiance;

    return create_light(node_ID, light);
}

void LightSources::destroy(LightSourceID light_ID) {
    // We don't actually destroy anything when destroying a light. 
    // The properties will get overwritten later when a light is created in same the spot.
    if (has(light_ID)) 
        m_changes.add_change(light_ID, Change::Destroyed);
}

bool LightSources::is_delta_light(LightSourceID light_ID) {
    switch (get_type(light_ID)) {
    case Type::Sphere:
        return is_delta_sphere_light(light_ID);
    case Type::Spot:
        return is_delta_spot_light(light_ID);
    case Type::Directional:
        return is_delta_directional_light(light_ID);
    }
    return false;
}

void LightSources::flag_as_updated(LightSourceID light_ID) {
    m_changes.add_change(light_ID, Change::Updated);
}

void LightSources::reset_change_notifications() {
    for (LightSourceID light_ID : get_changed_lights())
        if (get_changes(light_ID).is_set(Change::Destroyed))
            m_UID_generator.erase(light_ID);
    m_changes.reset_change_notifications();
}

// ------------------------------------------------------------------------------------------------
// Sphere light modifiers.
// ------------------------------------------------------------------------------------------------

void LightSources::set_sphere_light_power(LightSourceID light_ID, Math::RGB power) {
    m_lights[light_ID].color = power;
    flag_as_updated(light_ID);
}
void LightSources::set_sphere_light_radius(LightSourceID light_ID, float radius) {
    m_lights[light_ID].sphere.radius = radius;
    flag_as_updated(light_ID);
}

// ------------------------------------------------------------------------------------------------
// Spot light modifiers.
// ------------------------------------------------------------------------------------------------

void LightSources::set_spot_light_power(LightSourceID light_ID, Math::RGB power) {
    m_lights[light_ID].color = power;
    flag_as_updated(light_ID);
}
void LightSources::set_spot_light_radius(LightSourceID light_ID, float radius) {
    m_lights[light_ID].spot.radius = half(radius);
    flag_as_updated(light_ID);
}
void LightSources::set_spot_light_cos_angle(LightSourceID light_ID, float cos_angle) {
    m_lights[light_ID].spot.cos_angle = unsigned short(cos_angle * USHRT_MAX + 0.5f);
    flag_as_updated(light_ID);
}

// ------------------------------------------------------------------------------------------------
// Directional light modifiers.
// ------------------------------------------------------------------------------------------------

void LightSources::set_directional_light_radiance(LightSourceID light_ID, Math::RGB radiance) {
    m_lights[light_ID].color = radiance;
    flag_as_updated(light_ID);
}

} // NS Scene
} // NS Bifrost
