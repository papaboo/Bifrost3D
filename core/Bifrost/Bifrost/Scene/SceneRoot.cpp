// Bifrost scene root.
// ------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#include <Bifrost/Scene/SceneRoot.h>

#include <assert.h>

namespace Bifrost {
namespace Scene {

SceneRootIDGenerator SceneRoots::m_UID_generator = SceneRootIDGenerator(0u);
SceneRoots::Scene* SceneRoots::m_scenes = nullptr;
Core::ChangeSet<SceneRoots::Changes, SceneRootID> SceneRoots::m_changes;

void SceneRoots::allocate(unsigned int capacity) {
    if (is_allocated())
        return;

    m_UID_generator = SceneRootIDGenerator(capacity);
    capacity = m_UID_generator.capacity();

    m_scenes = new Scene[capacity];
    m_changes = Core::ChangeSet<Changes, SceneRootID>(capacity);

    // Allocate dummy element at 0.
    m_scenes[0].root_node = SceneNodeID::invalid_UID();
    m_scenes[0].environment_tint = Math::RGB::black();
    m_scenes[0].environment_light = nullptr;
}

void SceneRoots::deallocate() {
    if (!is_allocated())
        return;

    m_UID_generator = SceneRootIDGenerator(0u);
    delete[] m_scenes; m_scenes = nullptr;

    m_changes.resize(0);
}

void SceneRoots::reserve(unsigned int new_capacity) {
    unsigned int old_capacity = capacity();
    m_UID_generator.reserve(new_capacity);
    reserve_scene_data(m_UID_generator.capacity(), old_capacity);
}

template <typename T>
static inline T* resize_and_copy_array(T* old_array, unsigned int new_capacity, unsigned int copyable_elements) {
    T* new_array = new T[new_capacity];
    std::copy(old_array, old_array + copyable_elements, new_array);
    delete[] old_array;
    return new_array;
}

void SceneRoots::reserve_scene_data(unsigned int new_capacity, unsigned int old_capacity) {
    assert(m_scenes != nullptr);

    const unsigned int copyable_elements = new_capacity < old_capacity ? new_capacity : old_capacity;

    m_scenes = resize_and_copy_array(m_scenes, new_capacity, copyable_elements);
    m_changes.resize(new_capacity);
}

SceneRootID SceneRoots::create(const std::string& name, Assets::TextureID environment_map, Math::RGB environment_tint) {
    assert(m_scenes != nullptr);

    unsigned int old_capacity = m_UID_generator.capacity();
    SceneRootID id = m_UID_generator.generate();
    if (old_capacity != m_UID_generator.capacity())
        // The capacity has changed and the size of all arrays need to be adjusted.
        reserve_scene_data(m_UID_generator.capacity(), old_capacity);

    m_scenes[id].root_node = SceneNodes::create(name);
    m_scenes[id].environment_tint = environment_tint;
    m_scenes[id].environment_light = Assets::Textures::has(environment_map) ? new Assets::InfiniteAreaLight(environment_map) : nullptr;
    m_changes.set_change(id, Change::Created);

    return id;
}

void SceneRoots::destroy(SceneRootID scene_ID) {
    // We don't actually destroy anything when destroying a scene.
    // The properties will get overwritten later when a scene is created in same the spot.
    if (has(scene_ID))
        m_changes.add_change(scene_ID, Change::Destroyed);
}

void SceneRoots::set_environment_tint(SceneRootID scene_ID, Math::RGB tint) {
    m_scenes[scene_ID].environment_tint = tint;
    m_changes.add_change(scene_ID, Change::EnvironmentTint);
}

void SceneRoots::set_environment_map(SceneRootID scene_ID, Assets::TextureID environment_map) {
    assert(environment_map == Assets::TextureID::invalid_UID() || Assets::Textures::has(environment_map));

    delete m_scenes[scene_ID].environment_light;
    m_scenes[scene_ID].environment_light = Assets::Textures::has(environment_map) ? new Assets::InfiniteAreaLight(environment_map) : nullptr;
    m_changes.add_change(scene_ID, Change::EnvironmentMap);
}

void SceneRoots::reset_change_notifications() {
    for (SceneRootID scene_ID : get_changed_scenes())
        if (get_changes(scene_ID).is_set(Change::Destroyed))
            m_UID_generator.erase(scene_ID);
    m_changes.reset_change_notifications();
}

} // NS Scene
} // NS Bifrost
