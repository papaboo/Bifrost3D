// Cogwheel light source.
// ---------------------------------------------------------------------------
// Copyright (C) 2015-2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _COGWHEEL_SCENE_LIGHT_SOURCE_H_
#define _COGWHEEL_SCENE_LIGHT_SOURCE_H_

#include <Cogwheel/Core/Iterable.h>
#include <Cogwheel/Core/UniqueIDGenerator.h>
#include <Cogwheel/Math/Color.h>
#include <Cogwheel/Scene/SceneNode.h>

#include <vector>

namespace Cogwheel {
namespace Scene {

// ---------------------------------------------------------------------------
// Container class for cogwheel light sources.
// Future work
// * Light source type 'easy to use' wrappers. (See SceneNode)
// * Directional light.
// * Environment map.
// * Importance sampled environment map.
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

    // Sphere light.
    static inline bool is_delta_sphere_light(LightSources::UID light_ID) { assert(get_type(light_ID) == Type::Sphere); return m_lights[light_ID].sphere.radius == 0.0f; }
    static inline Math::RGB get_sphere_light_power(LightSources::UID light_ID) { return m_lights[light_ID].color; }
    static inline float get_sphere_light_radius(LightSources::UID light_ID) { assert(get_type(light_ID) == Type::Sphere); return m_lights[light_ID].sphere.radius; }

    // Directional light.
    static inline Math::RGB get_directional_light_radiance(LightSources::UID light_ID) { return m_lights[light_ID].color; }

    //-------------------------------------------------------------------------
    // Changes since last game loop tick.
    //-------------------------------------------------------------------------
    struct Changes {
        static const unsigned char None = 0u;
        static const unsigned char Created = 1u << 0u;
        static const unsigned char Destroyed = 1u << 1u;
        static const unsigned char All = Created | Destroyed;
    };

    static inline unsigned char get_changes(LightSources::UID light_ID) { return m_changes[light_ID]; }
    static inline bool has_changes(LightSources::UID light_ID, unsigned char change_bitmask = Changes::All) {
        return (m_changes[light_ID] & change_bitmask) != Changes::None;
    }

    typedef std::vector<UID>::iterator ChangedIterator;
    static Core::Iterable<ChangedIterator> get_changed_lights() {
        return Core::Iterable<ChangedIterator>(m_lights_changed.begin(), m_lights_changed.end());
    }

    static void reset_change_notifications();

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

    static unsigned char* m_changes; // Bitmask of changes.
    static std::vector<UID> m_lights_changed;
};

} // NS Scene
} // NS Cogwheel

#endif // _COGWHEEL_SCENE_LIGHT_SOURCE_H_