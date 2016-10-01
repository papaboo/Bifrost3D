// Cogwheel scene root.
// ---------------------------------------------------------------------------
// Copyright (C) 2015, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _COGWHEEL_SCENE_SCENE_ROOT_H_
#define _COGWHEEL_SCENE_SCENE_ROOT_H_

#include <Cogwheel/Assets/Texture.h>
#include <Cogwheel/Core/Iterable.h>
#include <Cogwheel/Core/UniqueIDGenerator.h>
#include <Cogwheel/Math/Color.h>
#include <Cogwheel/Scene/SceneNode.h>

namespace Cogwheel {
namespace Scene {

// ---------------------------------------------------------------------------
// The scene root contains the root scene node and scene specific properties,
// such as the environment map and tint.
// Future work
// * Environment projection modes: infinite sphere, camera X m above the earth, cube, sphere, ...
// * Change flags and iterator. Wait until we have multiscene support.
// * Do we need to store SceneRoots::UIDs of the owning scene in the scene nodes?
//   Wait until we have multi scene support to add it.
// * IBL generation using a GGX kernel and 
//   http://http.developer.nvidia.com/GPUGems3/gpugems3_ch20.html.
//   * Can also be used by the path tracer to reduce noise from 
//     environment sampling in the first frames.
// ---------------------------------------------------------------------------
class SceneRoots final {
public:
    typedef Core::TypedUIDGenerator<SceneRoots> UIDGenerator;
    typedef UIDGenerator::UID UID;
    typedef UIDGenerator::ConstIterator ConstUIDIterator;

    static bool is_allocated() { return m_scenes != nullptr; }
    static void allocate(unsigned int capacity);
    static void deallocate();

    static inline unsigned int capacity() { return m_UID_generator.capacity(); }
    static void reserve(unsigned int new_capacity);
    static bool has(SceneRoots::UID scene_ID) { return m_UID_generator.has(scene_ID); }

    static SceneRoots::UID create(const std::string& name, SceneNodes::UID root, Math::RGB environment_tint);
    static SceneRoots::UID create(const std::string& name, SceneNodes::UID root, Assets::Textures::UID environment_map);
    static void destroy(SceneRoots::UID scene_ID);

    static ConstUIDIterator begin() { return m_UID_generator.begin(); }
    static ConstUIDIterator end() { return m_UID_generator.end(); }
    static Core::Iterable<ConstUIDIterator> get_iterable() { return Core::Iterable<ConstUIDIterator>(begin(), end()); }

    static inline std::string get_name(SceneRoots::UID scene_ID) { return m_scenes[scene_ID].name; }
    static inline SceneNodes::UID get_root_node(SceneRoots::UID scene_ID) { return m_scenes[scene_ID].root_node; }
    static inline Math::RGB get_environment_tint(SceneRoots::UID scene_ID) { return m_scenes[scene_ID].environment_tint; }
    static void set_environment_tint(SceneRoots::UID scene_ID, Math::RGB tint);
    static inline Assets::Textures::UID get_environment_map(SceneRoots::UID scene_ID) { return m_scenes[scene_ID].environment_map; }
    static void set_environment_map(SceneRoots::UID scene_ID, Assets::Textures::UID environment_map);

    //-------------------------------------------------------------------------
    // Changes since last game loop tick.
    //-------------------------------------------------------------------------
    static struct Changes {
        static const unsigned char None = 0u;
        static const unsigned char Created = 1u << 0u;
        static const unsigned char Destroyed = 1u << 1u;
        static const unsigned char EnvironmentTint = 1u << 2u;
        static const unsigned char EnvironmentMap = 1u << 3u;
        static const unsigned char All = Created | Destroyed | EnvironmentTint | EnvironmentMap;
    };

    static inline unsigned char get_changes(SceneRoots::UID scene_ID) { return m_changes[scene_ID]; }
    static inline bool has_changes(SceneRoots::UID scene_ID, unsigned char change_bitmask = Changes::All) {
        return (m_changes[scene_ID] & change_bitmask) != Changes::None;
    }

    typedef std::vector<UID>::iterator ChangedIterator;
    static Core::Iterable<ChangedIterator> get_changed_scenes() {
        return Core::Iterable<ChangedIterator>(m_scenes_changed.begin(), m_scenes_changed.end());
    }

    static void reset_change_notifications();
private:

    static void reserve_scene_data(unsigned int new_capacity, unsigned int old_capacity);

    static void flag_as_changed(SceneRoots::UID node_ID, unsigned char change);

    static UIDGenerator m_UID_generator;

    struct Scene {
        std::string name;
        SceneNodes::UID root_node;
        Math::RGB environment_tint;
        Assets::Textures::UID environment_map;
    };

    static Scene* m_scenes;

    static unsigned char* m_changes; // Bitmask of changes.
    static std::vector<UID> m_scenes_changed;
};

// ---------------------------------------------------------------------------
// Cogwheel scene wrapper.
// ---------------------------------------------------------------------------
class SceneRoot final {
public:
    // -----------------------------------------------------------------------
    // Constructors and destructors.
    // -----------------------------------------------------------------------
    SceneRoot() : m_ID(SceneRoots::UID::invalid_UID()) {}
    SceneRoot(SceneRoots::UID id) : m_ID(id) {}

    inline const SceneRoots::UID get_ID() const { return m_ID; }
    inline bool exists() const { return SceneRoots::has(m_ID); }

    inline bool operator==(SceneRoot rhs) const { return m_ID == rhs.m_ID; }
    inline bool operator!=(SceneRoot rhs) const { return m_ID != rhs.m_ID; }

    // -----------------------------------------------------------------------
    // Getters and setters.
    // -----------------------------------------------------------------------
    inline std::string get_name() const { return SceneRoots::get_name(m_ID); }
    inline SceneNodes::UID get_root_node() const { return SceneRoots::get_root_node(m_ID); }
    inline Math::RGB get_environment_tint() const { return SceneRoots::get_environment_tint(m_ID); }
    inline void set_environment_tint(Math::RGB tint) { SceneRoots::set_environment_tint(m_ID, tint); }
    inline Assets::Textures::UID get_environment_map() const { return SceneRoots::get_environment_map(m_ID); }
    inline void set_environment_map(Assets::Textures::UID environment_map) { SceneRoots::set_environment_map(m_ID, environment_map); }

    //-------------------------------------------------------------------------
    // Changes since last game loop tick.
    //-------------------------------------------------------------------------
    inline unsigned char get_changes() const { return SceneRoots::get_changes(m_ID); }
    inline bool has_changes(unsigned char changes) const { return SceneRoots::has_changes(m_ID, changes); }

private:
    SceneRoots::UID m_ID;
};

} // NS Scene
} // NS Cogwheel

#endif // _COGWHEEL_SCENE_SCENE_ROOT_H_