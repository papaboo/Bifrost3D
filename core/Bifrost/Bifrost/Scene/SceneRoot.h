// Bifrost scene root.
// ------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _BIFROST_SCENE_SCENE_ROOT_H_
#define _BIFROST_SCENE_SCENE_ROOT_H_

#include <Bifrost/Assets/InfiniteAreaLight.h>
#include <Bifrost/Assets/Texture.h>
#include <Bifrost/Core/Bitmask.h>
#include <Bifrost/Core/ChangeSet.h>
#include <Bifrost/Core/Iterable.h>
#include <Bifrost/Core/UniqueIDGenerator.h>
#include <Bifrost/Math/Color.h>
#include <Bifrost/Scene/SceneNode.h>

namespace Bifrost {
namespace Scene {

//----------------------------------------------------------------------------
// Scene root ID
//----------------------------------------------------------------------------
class SceneRoots;
typedef Core::TypedUIDGenerator<SceneRoots> SceneRootIDGenerator;
typedef SceneRootIDGenerator::UID SceneRootID;

// ------------------------------------------------------------------------------------------------
// The scene root contains the root scene node and scene specific properties,
// such as the environment map and tint.
// Future work
// * Environment projection modes: infinite sphere, camera X m above the earth, cube, sphere, ...
// * Do we need to store SceneRootIDs of the owning scene in the scene nodes?
//   Wait until we have multi scene support to add it.
// * IBL generation using a GGX kernel and 
//   http://http.developer.nvidia.com/GPUGems3/gpugems3_ch20.html.
//   * Can also be used by the path tracer to reduce noise from 
//     environment sampling in the first frames, but how would that work with MIS and weighting?
// ------------------------------------------------------------------------------------------------
class SceneRoots final {
public:
    using Iterator = SceneRootIDGenerator::ConstIterator;

    static bool is_allocated() { return m_scenes != nullptr; }
    static void allocate(unsigned int capacity);
    static void deallocate();

    static inline unsigned int capacity() { return m_UID_generator.capacity(); }
    static void reserve(unsigned int new_capacity);
    static bool has(SceneRootID scene_ID) { return m_UID_generator.has(scene_ID) && get_changes(scene_ID).not_set(Change::Destroyed); }

    static SceneRootID create(const std::string& name, Assets::TextureID environment_map, Math::RGB environment_tint = Math::RGB::white());
    static SceneRootID create(const std::string& name, Math::RGB environment_tint) {
        return create(name, Assets::TextureID::invalid_UID(), environment_tint);
    }
    static void destroy(SceneRootID scene_ID);

    static Iterator begin() { return m_UID_generator.begin(); }
    static Iterator end() { return m_UID_generator.end(); }
    static Core::Iterable<Iterator> get_iterable() { return Core::Iterable<Iterator>(begin(), end()); }

    static inline std::string get_name(SceneRootID scene_ID) { return SceneNodes::get_name(m_scenes[scene_ID].root_node); }
    static inline SceneNodeID get_root_node(SceneRootID scene_ID) { return m_scenes[scene_ID].root_node; }
    static inline Math::RGB get_environment_tint(SceneRootID scene_ID) { return m_scenes[scene_ID].environment_tint; }
    static void set_environment_tint(SceneRootID scene_ID, Math::RGB tint);
    static inline Assets::InfiniteAreaLight* get_environment_light(SceneRootID scene_ID) { return m_scenes[scene_ID].environment_light; }
    static inline Assets::TextureID get_environment_map(SceneRootID scene_ID) { 
        auto environment_light = m_scenes[scene_ID].environment_light;
        return environment_light == nullptr ? Assets::TextureID::invalid_UID() : environment_light->get_texture().get_ID();
    }
    static void set_environment_map(SceneRootID scene_ID, Assets::TextureID environment_map);

    //---------------------------------------------------------------------------------------------
    // Changes since last game loop tick.
    //---------------------------------------------------------------------------------------------
    enum class Change : unsigned char {
        None            = 0u,
        Created         = 1u << 0u,
        Destroyed       = 1u << 1u,
        EnvironmentTint = 1u << 2u,
        EnvironmentMap  = 1u << 3u,
        All = Created | Destroyed | EnvironmentTint | EnvironmentMap
    };
    typedef Core::Bitmask<Change> Changes;

    static inline Changes get_changes(SceneRootID scene_ID) { return m_changes.get_changes(scene_ID); }

    typedef std::vector<SceneRootID>::iterator ChangedIterator;
    static Core::Iterable<ChangedIterator> get_changed_scenes() { return m_changes.get_changed_resources(); }

    static void reset_change_notifications();
private:

    static void reserve_scene_data(unsigned int new_capacity, unsigned int old_capacity);

    static SceneRootIDGenerator m_UID_generator;

    struct Scene {
        SceneNodeID root_node;
        Math::RGB environment_tint;
        Assets::InfiniteAreaLight* environment_light;
    };

    static Scene* m_scenes;

    static Core::ChangeSet<Changes, SceneRootID> m_changes;
};

// ------------------------------------------------------------------------------------------------
// Bifrost scene wrapper.
// ------------------------------------------------------------------------------------------------
class SceneRoot final {
public:
    // --------------------------------------------------------------------------------------------
    // Constructors and destructors.
    // --------------------------------------------------------------------------------------------
    SceneRoot() : m_ID(SceneRootID::invalid_UID()) {}
    SceneRoot(SceneRootID id) : m_ID(id) {}
    SceneRoot(const std::string& name, Math::RGB environment_tint)
        : m_ID(SceneRoots::create(name, Assets::TextureID::invalid_UID(), environment_tint)) {}

    SceneRoot(const std::string& name, Assets::Texture environment_map, Math::RGB environment_tint = Math::RGB::white())
        : m_ID(SceneRoots::create(name, environment_map.get_ID(), environment_tint)) {}

    static SceneRoot invalid() { return SceneRootID::invalid_UID(); }

    inline void destroy() { SceneRoots::destroy(m_ID); }
    inline bool exists() const { return SceneRoots::has(m_ID); }
    inline const SceneRootID get_ID() const { return m_ID; }

    inline bool operator==(SceneRoot rhs) const { return m_ID == rhs.m_ID; }
    inline bool operator!=(SceneRoot rhs) const { return m_ID != rhs.m_ID; }

    //---------------------------------------------------------------------------------------------
    // Getters and setters.
    //---------------------------------------------------------------------------------------------
    inline std::string get_name() const { return SceneRoots::get_name(m_ID); }
    inline SceneNode get_root_node() const { return SceneRoots::get_root_node(m_ID); }
    inline Math::RGB get_environment_tint() const { return SceneRoots::get_environment_tint(m_ID); }
    inline void set_environment_tint(Math::RGB tint) { SceneRoots::set_environment_tint(m_ID, tint); }
    inline Assets::Texture get_environment_map() const { return SceneRoots::get_environment_map(m_ID); }
    inline void set_environment_map(Assets::Texture environment_map) { SceneRoots::set_environment_map(m_ID, environment_map.get_ID()); }
    inline Assets::InfiniteAreaLight* get_environment_light() { return SceneRoots::get_environment_light(m_ID); }

    //---------------------------------------------------------------------------------------------
    // Changes since last game loop tick.
    //---------------------------------------------------------------------------------------------
    inline SceneRoots::Changes get_changes() const { return SceneRoots::get_changes(m_ID); }

private:
    SceneRootID m_ID;
};

} // NS Scene
} // NS Bifrost

#endif // _BIFROST_SCENE_SCENE_ROOT_H_
