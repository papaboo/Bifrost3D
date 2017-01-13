// Cogwheel light source.
// ---------------------------------------------------------------------------
// Copyright (C) 2015-2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _COGWHEEL_SCENE_LIGHT_SOURCE_H_
#define _COGWHEEL_SCENE_LIGHT_SOURCE_H_

#include <Cogwheel/Core/Bitmask.h>
#include <Cogwheel/Core/ChangeSet.h>
#include <Cogwheel/Core/Iterable.h>
#include <Cogwheel/Core/UniqueIDGenerator.h>
#include <Cogwheel/Math/Color.h>
#include <Cogwheel/Scene/SceneNode.h>

namespace Cogwheel {
namespace Scene {

// ---------------------------------------------------------------------------
// Container class for cogwheel light sources.
// Future work
// * Setters.
// ---------------------------------------------------------------------------
class LightSources final {
public:
    typedef Core::TypedUIDGenerator<LightSources> UIDGenerator;
    typedef UIDGenerator::UID UID;
    typedef UIDGenerator::ConstIterator ConstUIDIterator;

    enum class Type {
        Sphere,
        Directional
    };

    static bool is_allocated() { return m_lights != nullptr; }
    static void allocate(unsigned int capacity);
    static void deallocate();

    static inline unsigned int capacity() { return m_UID_generator.capacity(); }
    static void reserve(unsigned int new_capacity);
    static bool has(LightSources::UID light_ID) { return m_UID_generator.has(light_ID); }

    static LightSources::UID create_sphere_light(SceneNodes::UID node_ID, Math::RGB power, float radius);
    static LightSources::UID create_directional_light(SceneNodes::UID node_ID, Math::RGB radiance);
    static void destroy(LightSources::UID light_ID);

    static ConstUIDIterator begin() { return m_UID_generator.begin(); }
    static ConstUIDIterator end() { return m_UID_generator.end(); }
    static Core::Iterable<ConstUIDIterator> get_iterable() { return Core::Iterable<ConstUIDIterator>(begin(), end()); }

    static inline SceneNodes::UID get_node_ID(LightSources::UID light_ID) { return m_lights[light_ID].node_ID; }
    static inline Type get_type(LightSources::UID light_ID) { return m_lights[light_ID].type; }
    static bool is_delta_light(LightSources::UID light_ID);

    // Sphere light.
    static inline bool is_delta_sphere_light(LightSources::UID light_ID) { assert(get_type(light_ID) == Type::Sphere); return m_lights[light_ID].sphere.radius == 0.0f; }
    static inline Math::RGB get_sphere_light_power(LightSources::UID light_ID) { return m_lights[light_ID].color; }
    static inline float get_sphere_light_radius(LightSources::UID light_ID) { assert(get_type(light_ID) == Type::Sphere); return m_lights[light_ID].sphere.radius; }

    // Directional light.
    static inline bool is_delta_directional_light(LightSources::UID light_ID) { assert(get_type(light_ID) == Type::Directional); return true; }
    static inline Math::RGB get_directional_light_radiance(LightSources::UID light_ID) { return m_lights[light_ID].color; }

    //-------------------------------------------------------------------------
    // Changes since last game loop tick.
    //-------------------------------------------------------------------------
    enum class Change {
        None = 0u,
        Created = 1u << 0u,
        Destroyed = 1u << 1u,
        All = Created | Destroyed
    };
    typedef Core::Bitmask<Change> Changes;

    static inline Changes get_changes(LightSources::UID light_ID) { return m_changes.get_changes(light_ID); }

    typedef std::vector<UID>::iterator ChangedIterator;
    static Core::Iterable<ChangedIterator> get_changed_lights() { return m_changes.get_changed_resources(); }

    static void reset_change_notifications() { m_changes.reset_change_notifications(); }

private:
    static void reserve_light_data(unsigned int new_capacity, unsigned int old_capacity);

    static UIDGenerator m_UID_generator;

    struct Light {
        SceneNodes::UID node_ID;
        Math::RGB color; // The power in case of sphere lights and radiance in case of directional lights.
        Type type;
        union {
            struct {
                float radius;
            } sphere;

            struct {
            } directional;
        };
    };

    static Light* m_lights;

    static Core::ChangeSet<Changes, UID> m_changes;
};

// ---------------------------------------------------------------------------
// Cogwheel sphere light wrapper.
// ---------------------------------------------------------------------------
class SphereLight final {
public:
    // -----------------------------------------------------------------------
    // Constructors and destructors.
    // -----------------------------------------------------------------------
    SphereLight() : m_ID(LightSources::UID::invalid_UID()) {}
    SphereLight(LightSources::UID id) : m_ID(id) { assert(LightSources::get_type(id) == LightSources::Type::Sphere); }

    inline const LightSources::UID get_ID() const { return m_ID; }
    inline bool exists() const { return LightSources::has(m_ID); }

    inline bool operator==(SphereLight rhs) const { return m_ID == rhs.m_ID; }
    inline bool operator!=(SphereLight rhs) const { return m_ID != rhs.m_ID; }

    // -----------------------------------------------------------------------
    // Getters and setters.
    // -----------------------------------------------------------------------
    inline SceneNode get_node() const { return LightSources::get_node_ID(m_ID); }
    inline bool is_delta_light() const { return LightSources::is_delta_sphere_light(m_ID); }
    inline Math::RGB get_power() const { return LightSources::get_sphere_light_power(m_ID); }
    inline float get_radius() const { return LightSources::get_sphere_light_radius(m_ID); }

    //-------------------------------------------------------------------------
    // Changes since last game loop tick.
    //-------------------------------------------------------------------------
    inline LightSources::Changes get_changes() const { return LightSources::get_changes(m_ID); }

private:
    LightSources::UID m_ID;
};

// ---------------------------------------------------------------------------
// Cogwheel directional light wrapper.
// ---------------------------------------------------------------------------
class DirectionalLight final {
public:
    // -----------------------------------------------------------------------
    // Constructors and destructors.
    // -----------------------------------------------------------------------
    DirectionalLight() : m_ID(LightSources::UID::invalid_UID()) {}
    DirectionalLight(LightSources::UID id) : m_ID(id) { assert(LightSources::get_type(id) == LightSources::Type::Directional); }

    inline const LightSources::UID get_ID() const { return m_ID; }
    inline bool exists() const { return LightSources::has(m_ID); }

    inline bool operator==(DirectionalLight rhs) const { return m_ID == rhs.m_ID; }
    inline bool operator!=(DirectionalLight rhs) const { return m_ID != rhs.m_ID; }

    // -----------------------------------------------------------------------
    // Getters and setters.
    // -----------------------------------------------------------------------
    inline SceneNode get_node() const { return LightSources::get_node_ID(m_ID); }
    inline bool is_delta_light() const { return LightSources::is_delta_directional_light(m_ID); }
    inline Math::RGB get_radiance() const { return LightSources::get_directional_light_radiance(m_ID); }

    //-------------------------------------------------------------------------
    // Changes since last game loop tick.
    //-------------------------------------------------------------------------
    inline LightSources::Changes get_changes() const { return LightSources::get_changes(m_ID); }

private:
    LightSources::UID m_ID;
};

} // NS Scene
} // NS Cogwheel

#endif // _COGWHEEL_SCENE_LIGHT_SOURCE_H_