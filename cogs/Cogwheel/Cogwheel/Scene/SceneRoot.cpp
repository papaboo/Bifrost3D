// Cogwheel scene root.
// ---------------------------------------------------------------------------
// Copyright (C) 2015, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#include <Cogwheel/Scene/SceneRoot.h>

#include <assert.h>

namespace Cogwheel {
namespace Scene {

SceneRoots::UIDGenerator SceneRoots::m_UID_generator = UIDGenerator(0u);
SceneRoots::Scene* SceneRoots::m_scenes = nullptr;
unsigned char* SceneRoots::m_changes = nullptr;
std::vector<SceneRoots::UID> SceneRoots::m_scenes_changed = std::vector<SceneRoots::UID>(0);;

void SceneRoots::allocate(unsigned int capacity) {
    if (is_allocated())
        return;

    m_UID_generator = UIDGenerator(capacity);
    capacity = m_UID_generator.capacity();

    m_scenes = new Scene[capacity];
    m_changes = new unsigned char[capacity];
    std::memset(m_changes, Changes::None, capacity);

    // Allocate dummy element at 0.
    m_scenes[0].name = "Dummy Scene";
    m_scenes[0].root_node = SceneNodes::UID::invalid_UID();
    m_scenes[0].background_color = Math::RGB::black();
    m_scenes[0].environment_map = Assets::Textures::UID::invalid_UID();
}

void SceneRoots::deallocate() {
    if (!is_allocated())
        return;

    m_UID_generator = UIDGenerator(0u);
    delete[] m_scenes; m_scenes = nullptr;

    delete[] m_changes; m_changes = nullptr;
    m_scenes_changed.resize(0); m_scenes_changed.shrink_to_fit();
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
    m_changes = resize_and_copy_array(m_changes, new_capacity, copyable_elements);

    if (copyable_elements < new_capacity) // We need to zero the new change masks.
        std::memset(m_changes + copyable_elements, Changes::None, new_capacity - copyable_elements);
}

SceneRoots::UID SceneRoots::create(const std::string& name, SceneNodes::UID root, Math::RGB background_color) {
    assert(m_scenes != nullptr);

    unsigned int old_capacity = m_UID_generator.capacity();
    UID id = m_UID_generator.generate();
    if (old_capacity != m_UID_generator.capacity())
        // The capacity has changed and the size of all arrays need to be adjusted.
        reserve_scene_data(m_UID_generator.capacity(), old_capacity);

    if (m_changes[id] == Changes::None)
        m_scenes_changed.push_back(id);

    m_scenes[id].name = name;
    m_scenes[id].root_node = root;
    m_scenes[id].background_color = background_color;
    m_scenes[id].environment_map = Assets::Textures::UID::invalid_UID();
    m_changes[id] = Changes::Created;

    return id;
}

SceneRoots::UID SceneRoots::create(const std::string& name, SceneNodes::UID root, Assets::Textures::UID environment_map) {
    assert(m_scenes != nullptr);

    unsigned int old_capacity = m_UID_generator.capacity();
    UID id = m_UID_generator.generate();
    if (old_capacity != m_UID_generator.capacity())
        // The capacity has changed and the size of all arrays need to be adjusted.
        reserve_scene_data(m_UID_generator.capacity(), old_capacity);

    if (m_changes[id] == Changes::None)
        m_scenes_changed.push_back(id);

    m_scenes[id].name = name;
    m_scenes[id].root_node = root;
    m_scenes[id].background_color = Math::RGB::white();
    m_scenes[id].environment_map = environment_map;
    m_changes[id] = Changes::Created;

    return id;
}

void SceneRoots::destroy(SceneRoots::UID scene_ID) {
    // We don't actually destroy anything when destroying a scene.
    // The properties will get overwritten later when a scene is created in same the spot.
    if (m_UID_generator.erase(scene_ID))
        flag_as_changed(scene_ID, Changes::Destroyed);
}

void SceneRoots::set_background_color(SceneRoots::UID scene_ID, Math::RGB color) {
    m_scenes[scene_ID].background_color = color;
    flag_as_changed(scene_ID, Changes::BackgroundColor);
}

void SceneRoots::set_environment_map(SceneRoots::UID scene_ID, Assets::Textures::UID environment_map) {
    m_scenes[scene_ID].environment_map = environment_map;
    flag_as_changed(scene_ID, Changes::EnvironmentMap);
}

void SceneRoots::flag_as_changed(SceneRoots::UID node_ID, unsigned char change) {
    if (m_changes[node_ID] == Changes::None)
        m_scenes_changed.push_back(node_ID);
    m_changes[node_ID] |= change;
}

void SceneRoots::reset_change_notifications() {
    std::memset(m_changes, Changes::None, capacity());
    m_scenes_changed.resize(0);
}

} // NS Scene
} // NS Cogwheel
