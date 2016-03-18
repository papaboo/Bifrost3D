// Cogwheel rendered material.
// ---------------------------------------------------------------------------
// Copyright (C) 2015-2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _COGWHEEL_ASSETS_MATERIAL_H_
#define _COGWHEEL_ASSETS_MATERIAL_H_

#include <Cogwheel/Core/Iterable.h>
#include <Cogwheel/Core/UniqueIDGenerator.h>
#include <Cogwheel/Math/Color.h>

#include <vector>

namespace Cogwheel {
namespace Assets {

// TODO Description
class Materials final {
public:
    typedef Core::TypedUIDGenerator<Materials> UIDGenerator;
    typedef UIDGenerator::UID UID;
    typedef UIDGenerator::ConstIterator ConstUIDIterator;

    // TODO align to float4.
    struct Data {
        Math::RGB base_color;
        float base_roughness;
    };

    static bool is_allocated() { return m_materials != nullptr; }
    static void allocate(unsigned int capacity);
    static void deallocate();

    static inline unsigned int capacity() { return m_UID_generator.capacity(); }
    static void reserve(unsigned int new_capacity);
    static bool has(Materials::UID material_ID) { return m_UID_generator.has(material_ID); }

    static Materials::UID create(const std::string& name, const Data& data);
    static void destroy(Materials::UID material_ID);

    static ConstUIDIterator begin() { return m_UID_generator.begin(); }
    static ConstUIDIterator end() { return m_UID_generator.end(); }
    static Core::Iterable<ConstUIDIterator> get_iterable() { return Core::Iterable<ConstUIDIterator>(begin(), end()); }

    static inline std::string get_name(Materials::UID material_ID) { return m_names[material_ID]; }
    static inline void set_name(Materials::UID material_ID, const std::string& name) { m_names[material_ID] = name; }

    static inline Math::RGB get_base_color(Materials::UID material_ID) { return m_materials[material_ID].base_color; }
    static void set_base_color(Materials::UID material_ID, Math::RGB color);
    static inline float get_base_roughness(Materials::UID material_ID) { return m_materials[material_ID].base_roughness; }
    static void set_base_roughness(Materials::UID material_ID, float roughness);

    //-------------------------------------------------------------------------
    // Changes since last game loop tick.
    //-------------------------------------------------------------------------
    struct Events {
        static const unsigned char None = 0u;
        static const unsigned char Created = 1u << 0u;
        static const unsigned char Destroyed = 1u << 1u;
        static const unsigned char Changed = 1u << 2u;
    };

    static inline unsigned char get_material_events(Materials::UID material_ID) { return m_events[material_ID]; }
    static inline bool has_events(Materials::UID material_ID, unsigned char event_bitmask) {
        return (m_events[material_ID] & event_bitmask) == event_bitmask;
    }

    typedef std::vector<UID>::iterator material_created_iterator;
    static Core::Iterable<material_created_iterator> get_created_materials() {
        return Core::Iterable<material_created_iterator>(m_materials_created.begin(), m_materials_created.end());
    }

    typedef std::vector<UID>::iterator material_destroyed_iterator;
    static Core::Iterable<material_destroyed_iterator> get_destroyed_materials() {
        return Core::Iterable<material_destroyed_iterator>(m_materials_destroyed.begin(), m_materials_destroyed.end());
    }

    typedef std::vector<UID>::iterator material_changed_iterator;
    static Core::Iterable<material_changed_iterator> get_changed_materials() {
        return Core::Iterable<material_changed_iterator>(m_materials_changed.begin(), m_materials_changed.end());
    }

    static void reset_change_notifications();

private:
    static void reserve_material_data(unsigned int new_capacity, unsigned int old_capacity);

    static void flag_as_changed(Materials::UID material_ID);

    static UIDGenerator m_UID_generator;

    static std::string* m_names;
    static Data* m_materials;
    static unsigned char* m_events; // Bitmask of change events. Could be reduce to 4 bits pr material.

    // Change notifications.
    static std::vector<UID> m_materials_created;
    static std::vector<UID> m_materials_destroyed;
    static std::vector<UID> m_materials_changed;
};

class Material final {
private:
    const Materials::UID m_ID;

public:
    // -----------------------------------------------------------------------
    // Class management.
    // -----------------------------------------------------------------------
    Material() : m_ID(Materials::UID::invalid_UID()) {}
    Material(Materials::UID id) : m_ID(id) {}

    inline const Materials::UID get_ID() const { return m_ID; }
    inline bool exists() const { return Materials::has(m_ID); }

    inline bool operator==(Material rhs) const { return m_ID == rhs.m_ID; }
    inline bool operator!=(Material rhs) const { return m_ID != rhs.m_ID; }

    // -----------------------------------------------------------------------
    // Getters and setters.
    // -----------------------------------------------------------------------
    inline std::string get_name() const { return Materials::get_name(m_ID); }
    inline void set_name(const std::string& name) { Materials::set_name(m_ID, name); }

    inline Math::RGB get_base_color() { return Materials::get_base_color(m_ID); }
    void set_base_color(Math::RGB color) { Materials::set_base_color(m_ID, color); }
    inline float get_base_roughness() { return Materials::get_base_roughness(m_ID); }
    void set_base_roughness(float roughness) { Materials::set_base_roughness(m_ID, roughness); }

    inline unsigned char get_events() { return Materials::get_material_events(m_ID); }
    inline bool has_events(unsigned char event_bitmask) { return Materials::has_events(m_ID, event_bitmask); }
};

} // NS Assets
} // NS Cogwheel

#endif // _COGWHEEL_ASSETS_MATERIAL_H_