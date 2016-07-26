// Cogwheel rendered material.
// ---------------------------------------------------------------------------
// Copyright (C) 2015-2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _COGWHEEL_ASSETS_MATERIAL_H_
#define _COGWHEEL_ASSETS_MATERIAL_H_

#include <Cogwheel/Assets/Texture.h>
#include <Cogwheel/Core/Iterable.h>
#include <Cogwheel/Core/UniqueIDGenerator.h>
#include <Cogwheel/Math/Color.h>

#include <vector>

namespace Cogwheel {
namespace Assets {

// ---------------------------------------------------------------------------
// Cogwheel material properties container.
// ---------------------------------------------------------------------------
class Materials final {
public:
    typedef Core::TypedUIDGenerator<Materials> UIDGenerator;
    typedef UIDGenerator::UID UID;
    typedef UIDGenerator::ConstIterator ConstUIDIterator;

    struct Data {
        Math::RGB base_tint;
        Textures::UID base_tint_texture_ID;
        float base_roughness;
        float specularity;
        float metallic;
    };

    static bool is_allocated() { return m_materials != nullptr; }
    static void allocate(unsigned int capacity);
    static void deallocate();

    static inline unsigned int capacity() { return m_UID_generator.capacity(); }
    static void reserve(unsigned int new_capacity);
    static inline bool has(Materials::UID material_ID) { return m_UID_generator.has(material_ID); }

    static Materials::UID create(const std::string& name, const Data& data);
    static void destroy(Materials::UID material_ID);

    static inline ConstUIDIterator begin() { return m_UID_generator.begin(); }
    static inline ConstUIDIterator end() { return m_UID_generator.end(); }
    static inline Core::Iterable<ConstUIDIterator> get_iterable() { return Core::Iterable<ConstUIDIterator>(begin(), end()); }

    static inline std::string get_name(Materials::UID material_ID) { return m_names[material_ID]; }
    static inline void set_name(Materials::UID material_ID, const std::string& name) { m_names[material_ID] = name; }

    static inline Math::RGB get_base_tint(Materials::UID material_ID) { return m_materials[material_ID].base_tint; }
    static void set_base_tint(Materials::UID material_ID, Math::RGB tint);
    static inline Textures::UID get_base_tint_texture_ID(Materials::UID material_ID) { return m_materials[material_ID].base_tint_texture_ID; }
    static void set_base_tint_texture_ID(Materials::UID material_ID, Textures::UID tint_texture_ID);
    static inline float get_base_roughness(Materials::UID material_ID) { return m_materials[material_ID].base_roughness; }
    static void set_base_roughness(Materials::UID material_ID, float roughness);
    static inline float get_specularity(Materials::UID material_ID) { return m_materials[material_ID].specularity; }
    static void set_specularity(Materials::UID material_ID, float incident_specularity);
    static inline float get_metallic(Materials::UID material_ID) { return m_materials[material_ID].metallic; }
    static void set_metallic(Materials::UID material_ID, float metallic);

    //-------------------------------------------------------------------------
    // Changes since last game loop tick.
    //-------------------------------------------------------------------------
    struct Changes {
        static const unsigned char None = 0u;
        static const unsigned char Created = 1u << 0u;
        static const unsigned char Destroyed = 1u << 1u;
        static const unsigned char Updated = 1u << 2u;
        static const unsigned char All = Created | Destroyed | Updated;
    };

    static inline unsigned char get_changes(Materials::UID material_ID) { return m_changes[material_ID]; }
    static inline bool has_changes(Materials::UID material_ID, unsigned char change_bitmask = Changes::All) {
        return (m_changes[material_ID] & change_bitmask) != Changes::None;
    }

    typedef std::vector<UID>::iterator ChangedIterator;
    static inline Core::Iterable<ChangedIterator> get_changed_materials() {
        return Core::Iterable<ChangedIterator>(m_materials_changed.begin(), m_materials_changed.end());
    }

    static void reset_change_notifications();

private:
    static void reserve_material_data(unsigned int new_capacity, unsigned int old_capacity);

    static void flag_as_updated(Materials::UID material_ID);

    static UIDGenerator m_UID_generator;

    static std::string* m_names;
    static Data* m_materials;
    static unsigned char* m_changes; // Bitmask of changes. Could be reduced to 4 bits pr material.
    static std::vector<UID> m_materials_changed;
};

// ---------------------------------------------------------------------------
// Material UID wrapper.
// ---------------------------------------------------------------------------
class Material final {
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

    inline Math::RGB get_base_tint() { return Materials::get_base_tint(m_ID); }
    inline void set_base_tint(Math::RGB tint) { Materials::set_base_tint(m_ID, tint); }
    inline Textures::UID get_base_tint_texture_ID() { return Materials::get_base_tint_texture_ID(m_ID); }
    inline void set_base_tint(Textures::UID tint_texture_ID) { Materials::set_base_tint_texture_ID(m_ID, tint_texture_ID); }
    inline float get_base_roughness() { return Materials::get_base_roughness(m_ID); }
    inline void set_base_roughness(float roughness) { Materials::set_base_roughness(m_ID, roughness); }
    inline float get_specularity() { return Materials::get_specularity(m_ID); }
    inline void set_specularity(float specularity) { Materials::set_specularity(m_ID, specularity); }
    inline float get_metallic() { return Materials::get_metallic(m_ID); }
    inline void set_metallic(float metallic) { Materials::set_metallic(m_ID, metallic); }

    inline unsigned char get_changes() { return Materials::get_changes(m_ID); }
    inline bool has_changes(unsigned char change_bitmask) { return Materials::has_changes(m_ID, change_bitmask); }

private:
    const Materials::UID m_ID;
};

} // NS Assets
} // NS Cogwheel

#endif // _COGWHEEL_ASSETS_MATERIAL_H_