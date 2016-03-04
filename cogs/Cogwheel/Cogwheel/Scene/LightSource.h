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
// * Environment map.
// * Light source type 'easy to use' wrappers. (See SceneNode)
// * Directional light.
// * Importance sampled environment map.
// * Setters.
// * Perhaps just have a single notification buffer?
// ---------------------------------------------------------------------------
class LightSources final {
public:
    typedef Core::TypedUIDGenerator<LightSources> UIDGenerator;
    typedef UIDGenerator::UID UID;
    typedef UIDGenerator::ConstIterator ConstUIDIterator;

    static bool is_allocated() { return m_node_IDs != nullptr; }
    static void allocate(unsigned int capacity);
    static void deallocate();

    static inline unsigned int capacity() { return m_UID_generator.capacity(); }
    static void reserve(unsigned int new_capacity);
    static bool has(LightSources::UID light_ID) { return m_UID_generator.has(light_ID); }

    static LightSources::UID create_point_light(SceneNodes::UID node_ID, Math::RGB power, float radius);
    static void destroy(LightSources::UID light_ID);

    static inline SceneNodes::UID get_node_ID(LightSources::UID light_ID) { return m_node_IDs[light_ID]; }
    static inline bool is_delta_light(LightSources::UID light_ID) { return m_radius[light_ID] == 0.0f; }
    static inline Math::RGB get_power(LightSources::UID light_ID) { return m_power[light_ID]; }
    static inline float get_radius(LightSources::UID light_ID) { return m_radius[light_ID]; }

    static ConstUIDIterator begin() { return m_UID_generator.begin(); }
    static ConstUIDIterator end() { return m_UID_generator.end(); }
    static Core::Iterable<ConstUIDIterator> get_iterable() { return Core::Iterable<ConstUIDIterator>(begin(), end()); }

    //-------------------------------------------------------------------------
    // Changes since last game loop tick.
    //-------------------------------------------------------------------------
    typedef std::vector<UID>::iterator light_created_iterator;
    static Core::Iterable<light_created_iterator> get_created_lights() {
        return Core::Iterable<light_created_iterator>(m_lights_created.begin(), m_lights_created.end());
    }

    typedef std::vector<UID>::iterator light_destroyed_iterator;
    static Core::Iterable<light_destroyed_iterator> get_destroyed_lights() {
        return Core::Iterable<light_destroyed_iterator>(m_lights_destroyed.begin(), m_lights_destroyed.end());
    }

    static void reset_change_notifications();

private:
    static void reserve_light_data(unsigned int new_capacity, unsigned int old_capacity);

    static UIDGenerator m_UID_generator;

    static SceneNodes::UID* m_node_IDs;
    static Math::RGB* m_power;
    static float* m_radius;

    // Change notifications.
    static std::vector<UID> m_lights_created;
    static std::vector<UID> m_lights_destroyed;
};

} // NS Scene
} // NS Cogwheel

#endif // _COGWHEEL_SCENE_LIGHT_SOURCE_H_