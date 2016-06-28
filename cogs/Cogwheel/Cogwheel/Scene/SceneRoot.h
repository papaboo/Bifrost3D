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
// such as the background color or environment map.
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

    static SceneRoots::UID create(const std::string& name, SceneNodes::UID root, Math::RGB background_color);
    static SceneRoots::UID create(const std::string& name, SceneNodes::UID root, Assets::Textures::UID environment_map);
    // static void destroy(SceneRoots::UID scene_ID);

    static ConstUIDIterator begin() { return m_UID_generator.begin(); }
    static ConstUIDIterator end() { return m_UID_generator.end(); }
    static Core::Iterable<ConstUIDIterator> get_iterable() { return Core::Iterable<ConstUIDIterator>(begin(), end()); }

    static inline std::string get_name(SceneRoots::UID scene_ID) { return m_scenes[scene_ID].name; }
    static inline SceneNodes::UID get_root_node(SceneRoots::UID scene_ID) { return m_scenes[scene_ID].root_node; }
    static inline Math::RGB get_background_color(SceneRoots::UID scene_ID) { return m_scenes[scene_ID].background_color; }
    static inline Assets::Textures::UID get_environment_map(SceneRoots::UID scene_ID) { return m_scenes[scene_ID].environment_map; }

private:

    static void reserve_scene_data(unsigned int new_capacity, unsigned int old_capacity);

    static UIDGenerator m_UID_generator;

    struct Scene {
        std::string name;
        SceneNodes::UID root_node;
        Math::RGB background_color;
        Assets::Textures::UID environment_map;
    };

    static Scene* m_scenes;
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
    inline Math::RGB get_background_color() const { return SceneRoots::get_background_color(m_ID); }
    inline Assets::Textures::UID get_environment_map() const { return SceneRoots::get_environment_map(m_ID); }

    //-------------------------------------------------------------------------
    // Changes since last game loop tick.
    //-------------------------------------------------------------------------
    // inline unsigned char get_changes() const { return SceneRoots::get_changes(m_ID); }
    // inline bool has_changes(unsigned char changes) const { return SceneRoots::has_changes(m_ID, changes); }

private:
    SceneRoots::UID m_ID;
};

} // NS Scene
} // NS Cogwheel

#endif // _COGWHEEL_SCENE_SCENE_ROOT_H_