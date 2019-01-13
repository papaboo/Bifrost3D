// Bifrost scene root.
// ------------------------------------------------------------------------------------------------
// Copyright (C) 2015, Bifrost. See AUTHORS.txt for authors
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

// ------------------------------------------------------------------------------------------------
// The scene root contains the root scene node and scene specific properties,
// such as the environment map and tint.
// Future work
// * Environment projection modes: infinite sphere, camera X m above the earth, cube, sphere, ...
// * Do we need to store SceneRoots::UIDs of the owning scene in the scene nodes?
//   Wait until we have multi scene support to add it.
// * IBL generation using a GGX kernel and 
//   http://http.developer.nvidia.com/GPUGems3/gpugems3_ch20.html.
//   * Can also be used by the path tracer to reduce noise from 
//     environment sampling in the first frames, but how would that work with MIS and weighting?
// ------------------------------------------------------------------------------------------------
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

    static SceneRoots::UID create(const std::string& name, Assets::Textures::UID environment_map, Math::RGB environment_tint = Math::RGB::white());
    static SceneRoots::UID create(const std::string& name, Math::RGB environment_tint) {
        return create(name, Assets::Textures::UID::invalid_UID(), environment_tint);
    }
    static void destroy(SceneRoots::UID scene_ID);

    static ConstUIDIterator begin() { return m_UID_generator.begin(); }
    static ConstUIDIterator end() { return m_UID_generator.end(); }
    static Core::Iterable<ConstUIDIterator> get_iterable() { return Core::Iterable<ConstUIDIterator>(begin(), end()); }

    static inline std::string get_name(SceneRoots::UID scene_ID) { return SceneNodes::get_name(m_scenes[scene_ID].root_node); }
    static inline SceneNodes::UID get_root_node(SceneRoots::UID scene_ID) { return m_scenes[scene_ID].root_node; }
    static inline Math::RGB get_environment_tint(SceneRoots::UID scene_ID) { return m_scenes[scene_ID].environment_tint; }
    static void set_environment_tint(SceneRoots::UID scene_ID, Math::RGB tint);
    static inline Assets::InfiniteAreaLight* get_environment_light(SceneRoots::UID scene_ID) { return m_scenes[scene_ID].environment_light; }
    static inline Assets::Textures::UID get_environment_map(SceneRoots::UID scene_ID) { 
        auto environment_light = m_scenes[scene_ID].environment_light;
        return environment_light == nullptr ? Assets::Textures::UID::invalid_UID() : environment_light->get_texture_ID();
    }
    static void set_environment_map(SceneRoots::UID scene_ID, Assets::Textures::UID environment_map);

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

    static inline Changes get_changes(SceneRoots::UID scene_ID) { return m_changes.get_changes(scene_ID); }

    typedef std::vector<UID>::iterator ChangedIterator;
    static Core::Iterable<ChangedIterator> get_changed_scenes() { return m_changes.get_changed_resources(); }

    static void reset_change_notifications() { m_changes.reset_change_notifications(); }
private:

    static void reserve_scene_data(unsigned int new_capacity, unsigned int old_capacity);

    static UIDGenerator m_UID_generator;

    struct Scene {
        SceneNodes::UID root_node;
        Math::RGB environment_tint;
        Assets::InfiniteAreaLight* environment_light;
    };

    static Scene* m_scenes;

    static Core::ChangeSet<Changes, UID> m_changes;
};

// ------------------------------------------------------------------------------------------------
// Bifrost scene wrapper.
// ------------------------------------------------------------------------------------------------
class SceneRoot final {
public:
    // --------------------------------------------------------------------------------------------
    // Constructors and destructors.
    // --------------------------------------------------------------------------------------------
    SceneRoot() : m_ID(SceneRoots::UID::invalid_UID()) {}
    SceneRoot(SceneRoots::UID id) : m_ID(id) {}

    inline const SceneRoots::UID get_ID() const { return m_ID; }
    inline bool exists() const { return SceneRoots::has(m_ID); }

    inline bool operator==(SceneRoot rhs) const { return m_ID == rhs.m_ID; }
    inline bool operator!=(SceneRoot rhs) const { return m_ID != rhs.m_ID; }

    //---------------------------------------------------------------------------------------------
    // Getters and setters.
    //---------------------------------------------------------------------------------------------
    inline std::string get_name() const { return SceneRoots::get_name(m_ID); }
    inline SceneNodes::UID get_root_node() const { return SceneRoots::get_root_node(m_ID); }
    inline Math::RGB get_environment_tint() const { return SceneRoots::get_environment_tint(m_ID); }
    inline void set_environment_tint(Math::RGB tint) { SceneRoots::set_environment_tint(m_ID, tint); }
    inline Assets::Textures::UID get_environment_map() const { return SceneRoots::get_environment_map(m_ID); }
    inline void set_environment_map(Assets::Textures::UID environment_map) { SceneRoots::set_environment_map(m_ID, environment_map); }
    inline Assets::InfiniteAreaLight* get_environment_light() { return SceneRoots::get_environment_light(m_ID); }

    //---------------------------------------------------------------------------------------------
    // Changes since last game loop tick.
    //---------------------------------------------------------------------------------------------
    inline SceneRoots::Changes get_changes() const { return SceneRoots::get_changes(m_ID); }

private:
    SceneRoots::UID m_ID;
};

} // NS Scene
} // NS Bifrost

#endif // _BIFROST_SCENE_SCENE_ROOT_H_
