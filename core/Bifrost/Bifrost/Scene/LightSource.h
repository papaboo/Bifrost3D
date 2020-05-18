// Bifrost light source.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _BIFROST_SCENE_LIGHT_SOURCE_H_
#define _BIFROST_SCENE_LIGHT_SOURCE_H_

#include <Bifrost/Core/Bitmask.h>
#include <Bifrost/Core/ChangeSet.h>
#include <Bifrost/Core/Iterable.h>
#include <Bifrost/Core/UniqueIDGenerator.h>
#include <Bifrost/Math/Color.h>
#include <Bifrost/Math/Half.h>
#include <Bifrost/Scene/SceneNode.h>

namespace Bifrost {
namespace Scene {

// ---------------------------------------------------------------------------
// Container class for Bifrost light sources.
// Future work
// * Setters.
// ---------------------------------------------------------------------------
class LightSources final {
public:
    typedef Core::TypedUIDGenerator<LightSources> UIDGenerator;
    typedef UIDGenerator::UID UID;
    typedef UIDGenerator::ConstIterator ConstUIDIterator;

    enum class Type : unsigned char {
        Sphere,
        Spot,
        Directional
    };

    static bool is_allocated() { return m_lights != nullptr; }
    static void allocate(unsigned int capacity);
    static void deallocate();

    static inline unsigned int capacity() { return m_UID_generator.capacity(); }
    static void reserve(unsigned int new_capacity);
    static bool has(LightSources::UID light_ID) { return m_UID_generator.has(light_ID); }

    static LightSources::UID create_sphere_light(SceneNodes::UID node_ID, Math::RGB power, float radius);
    static LightSources::UID create_spot_light(SceneNodes::UID node_ID, Math::RGB power, float radius, float cos_angle);
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
    static void set_sphere_light_power(LightSources::UID light_ID, Math::RGB power);
    static inline float get_sphere_light_radius(LightSources::UID light_ID) { assert(get_type(light_ID) == Type::Sphere); return m_lights[light_ID].sphere.radius; }
    static void set_sphere_light_radius(LightSources::UID light_ID, float radius);

    // Spot light
    static inline bool is_delta_spot_light(LightSources::UID light_ID) { assert(get_type(light_ID) == Type::Spot); return m_lights[light_ID].spot.radius == 0.0f || m_lights[light_ID].spot.cos_angle == USHRT_MAX; }
    static inline Math::RGB get_spot_light_power(LightSources::UID light_ID) { return m_lights[light_ID].color; }
    static void set_spot_light_power(LightSources::UID light_ID, Math::RGB power);
    static inline float get_spot_light_radius(LightSources::UID light_ID) { assert(get_type(light_ID) == Type::Spot); return m_lights[light_ID].spot.radius; }
    static void set_spot_light_radius(LightSources::UID light_ID, float radius);
    static inline float get_spot_light_cos_angle(LightSources::UID light_ID) { assert(get_type(light_ID) == Type::Spot); return m_lights[light_ID].spot.cos_angle / float(USHRT_MAX); }
    static void set_spot_light_cos_angle(LightSources::UID light_ID, float cos_angle);

    // Directional light.
    static inline bool is_delta_directional_light(LightSources::UID light_ID) { assert(get_type(light_ID) == Type::Directional); return true; }
    static inline Math::RGB get_directional_light_radiance(LightSources::UID light_ID) { return m_lights[light_ID].color; }
    static void set_directional_light_radiance(LightSources::UID light_ID, Math::RGB radiance);

    //-------------------------------------------------------------------------
    // Changes since last game loop tick.
    //-------------------------------------------------------------------------
    enum class Change : unsigned char {
        None = 0u,
        Created = 1u << 0u,
        Destroyed = 1u << 1u,
        Updated = 1u << 2u,
        All = Created | Destroyed | Updated
    };
    typedef Core::Bitmask<Change> Changes;

    static inline Changes get_changes(LightSources::UID light_ID) { return m_changes.get_changes(light_ID); }

    typedef std::vector<UID>::iterator ChangedIterator;
    static Core::Iterable<ChangedIterator> get_changed_lights() { return m_changes.get_changed_resources(); }

    static void reset_change_notifications() { m_changes.reset_change_notifications(); }

private:
    static UIDGenerator m_UID_generator;

    struct Light {
        SceneNodes::UID node_ID;
        Math::RGB color; // The power in case of sphere- or cone light and radiance in case of directional lights.
        Type type;
        union {
            struct {
                float radius;
            } sphere;

            struct {
                half_float::half radius;
                unsigned short cos_angle;
            } spot;

            struct {
            } directional;
        };
    };

    static Light* m_lights;

    static void flag_as_updated(LightSources::UID light_ID);

    static Core::ChangeSet<Changes, UID> m_changes;

    static LightSources::UID create_light(SceneNodes::UID node_ID, LightSources::Light light);
    static void reserve_light_data(unsigned int new_capacity, unsigned int old_capacity);
};

// ---------------------------------------------------------------------------
// Bifrost sphere light wrapper.
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
    inline void set_power(Math::RGB power) { LightSources::set_sphere_light_power(m_ID, power); }
    inline float get_radius() const { return LightSources::get_sphere_light_radius(m_ID); }
    inline void set_radius(float radius) { LightSources::set_sphere_light_radius(m_ID, radius); }

    //-------------------------------------------------------------------------
    // Changes since last game loop tick.
    //-------------------------------------------------------------------------
    inline LightSources::Changes get_changes() const { return LightSources::get_changes(m_ID); }

private:
    LightSources::UID m_ID;
};

// ---------------------------------------------------------------------------
// Bifrost spot light wrapper.
// ---------------------------------------------------------------------------
class SpotLight final {
public:
    // -----------------------------------------------------------------------
    // Constructors and destructors.
    // -----------------------------------------------------------------------
    SpotLight() : m_ID(LightSources::UID::invalid_UID()) {}
    SpotLight(LightSources::UID id) : m_ID(id) { assert(LightSources::get_type(id) == LightSources::Type::Spot); }

    inline const LightSources::UID get_ID() const { return m_ID; }
    inline bool exists() const { return LightSources::has(m_ID); }

    inline bool operator==(SpotLight rhs) const { return m_ID == rhs.m_ID; }
    inline bool operator!=(SpotLight rhs) const { return m_ID != rhs.m_ID; }

    // -----------------------------------------------------------------------
    // Getters and setters.
    // -----------------------------------------------------------------------
    inline SceneNode get_node() const { return LightSources::get_node_ID(m_ID); }
    inline bool is_delta_light() const { return LightSources::is_delta_spot_light(m_ID); }
    inline Math::RGB get_power() const { return LightSources::get_spot_light_power(m_ID); }
    inline void set_power(Math::RGB power) { LightSources::set_spot_light_power(m_ID, power); }
    inline float get_radius() const { return LightSources::get_spot_light_radius(m_ID); }
    inline void set_radius(float radius) { LightSources::set_spot_light_radius(m_ID, radius); }
    inline float get_cos_angle() const { return LightSources::get_spot_light_cos_angle(m_ID); }
    inline float get_angle() const { return acos(get_cos_angle()); }
    inline void set_cos_angle(float cos_angle) { LightSources::set_spot_light_cos_angle(m_ID, cos_angle); }

    //-------------------------------------------------------------------------
    // Changes since last game loop tick.
    //-------------------------------------------------------------------------
    inline LightSources::Changes get_changes() const { return LightSources::get_changes(m_ID); }

private:
    LightSources::UID m_ID;
};

// ---------------------------------------------------------------------------
// Bifrost directional light wrapper.
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
    inline void set_radiance(Math::RGB radiance) { LightSources::set_directional_light_radiance(m_ID, radiance); }

    //-------------------------------------------------------------------------
    // Changes since last game loop tick.
    //-------------------------------------------------------------------------
    inline LightSources::Changes get_changes() const { return LightSources::get_changes(m_ID); }

private:
    LightSources::UID m_ID;
};

} // NS Scene
} // NS Bifrost

#endif // _BIFROST_SCENE_LIGHT_SOURCE_H_
