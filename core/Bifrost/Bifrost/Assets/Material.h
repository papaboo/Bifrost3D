// Bifrost rendered material.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _BIFROST_ASSETS_MATERIAL_H_
#define _BIFROST_ASSETS_MATERIAL_H_

#include <Bifrost/Assets/Texture.h>
#include <Bifrost/Core/Bitmask.h>
#include <Bifrost/Core/ChangeSet.h>
#include <Bifrost/Core/Iterable.h>
#include <Bifrost/Core/UniqueIDGenerator.h>
#include <Bifrost/Math/Color.h>

namespace Bifrost {
namespace Assets {

// ---------------------------------------------------------------------------
// Bifrost material flags.
// ---------------------------------------------------------------------------
enum class MaterialFlag : unsigned char {
    None = 0u,
    ThinWalled = 1u,
    Cutout = 2u,
};

// ---------------------------------------------------------------------------
// Bifrost shading models.
// ---------------------------------------------------------------------------
enum class ShadingModel : unsigned char {
    Default = 0u,
    Diffuse = 1u,
    Count = 2
};

// ---------------------------------------------------------------------------
// Metal tints.
// ---------------------------------------------------------------------------
const Math::RGB iron_tint = Math::RGB(0.560f, 0.570f, 0.580f);
const Math::RGB silver_tint = Math::RGB(0.972f, 0.960f, 0.915f);
const Math::RGB aluminum_tint = Math::RGB(0.913f, 0.921f, 0.925f);
const Math::RGB gold_tint = Math::RGB(1.000f, 0.766f, 0.336f);
const Math::RGB copper_tint = Math::RGB(0.955f, 0.637f, 0.538f);
const Math::RGB chromium_tint = Math::RGB(0.550f, 0.556f, 0.554f);
const Math::RGB nickel_tint = Math::RGB(0.660f, 0.609f, 0.526f);
const Math::RGB titanium_tint = Math::RGB(0.542f, 0.497f, 0.449f);
const Math::RGB cobalt_tint = Math::RGB(0.662f, 0.655f, 0.634f);
const Math::RGB platinum_tint = Math::RGB(0.672f, 0.637f, 0.585f);

//----------------------------------------------------------------------------
// Material ID
//----------------------------------------------------------------------------
class Materials;
typedef Core::TypedUIDGenerator<Materials> MaterialIDGenerator;
typedef MaterialIDGenerator::UID MaterialID;

// ---------------------------------------------------------------------------
// Bifrost material properties container.
// Be aware that coverage and the cutout threshold map to the same storage, as they are mutually exclusive,
// and the interpretation depends on whether the material is a cutout or not.
// ---------------------------------------------------------------------------
class Materials final {
public:
    using Iterator = MaterialIDGenerator::ConstIterator;

    typedef Core::Bitmask<MaterialFlag> Flags;

    struct Data {
        ShadingModel shading_model;
        Math::RGB tint;
        TextureID tint_roughness_texture_ID;
        float roughness;
        float specularity;
        float metallic;
        float coat;
        float coat_roughness;
        TextureID metallic_texture_ID;
        float coverage;
        TextureID coverage_texture_ID;
        float transmission;
        Flags flags;

        static Data create_dielectric(Math::RGB tint, float roughness, float specularity) {
            Data res = {};
            res.tint = tint;
            res.roughness = roughness;
            res.specularity = specularity;
            res.coverage = 1.0f;
            return res;
        }

        static Data create_metal(Math::RGB tint, float roughness) {
            Data res = {};
            res.tint = tint;
            res.roughness = roughness;
            res.specularity = 1.0f;
            res.coverage = 1.0f;
            res.metallic = 1.0f;
            return res;
        }

        static Data create_coated_dielectric(Math::RGB tint, float roughness, float specularity, float coat_roughness) {
            Data res = {};
            res.tint = tint;
            res.roughness = roughness;
            res.specularity = specularity;
            res.coat = 1.0f;
            res.coat_roughness = coat_roughness;
            res.coverage = 1.0f;
            return res;
        }
    };

    static bool is_allocated() { return m_materials != nullptr; }
    static void allocate(unsigned int capacity);
    static void deallocate();

    static inline unsigned int capacity() { return m_UID_generator.capacity(); }
    static void reserve(unsigned int new_capacity);
    static inline bool has(MaterialID material_ID) { return m_UID_generator.has(material_ID); }

    static MaterialID create(const std::string& name, const Data& data);
    static void destroy(MaterialID material_ID);

    static inline Iterator begin() { return m_UID_generator.begin(); }
    static inline Iterator end() { return m_UID_generator.end(); }
    static inline Core::Iterable<Iterator> get_iterable() { return Core::Iterable<Iterator>(begin(), end()); }

    static inline std::string get_name(MaterialID material_ID) { return m_names[material_ID]; }
    static inline void set_name(MaterialID material_ID, const std::string& name) { m_names[material_ID] = name; }

    static inline ShadingModel get_shading_model(MaterialID material_ID) { return m_materials[material_ID].shading_model; }
    static void set_shading_model(MaterialID material_ID, ShadingModel shading_model);

    static inline Flags get_flags(MaterialID material_ID) { return m_materials[material_ID].flags; }
    static void set_flags(MaterialID material_ID, Flags flags);
    static inline bool is_thin_walled(MaterialID material_ID) { return get_flags(material_ID).is_set(MaterialFlag::ThinWalled); }
    static inline bool is_cutout(MaterialID material_ID) { return get_flags(material_ID).is_set(MaterialFlag::Cutout); }

    static inline Math::RGB get_tint(MaterialID material_ID) { return m_materials[material_ID].tint; }
    static void set_tint(MaterialID material_ID, Math::RGB tint);
    static inline float get_roughness(MaterialID material_ID) { return m_materials[material_ID].roughness; }
    static void set_roughness(MaterialID material_ID, float roughness);
    static inline TextureID get_tint_roughness_texture_ID(MaterialID material_ID) { return m_materials[material_ID].tint_roughness_texture_ID; }
    static void set_tint_roughness_texture_ID(MaterialID material_ID, TextureID tint_roughness_texture_ID);
    static bool has_tint_texture(MaterialID material_ID);
    static bool has_roughness_texture(MaterialID material_ID);

    static inline float get_specularity(MaterialID material_ID) { return m_materials[material_ID].specularity; }
    static void set_specularity(MaterialID material_ID, float incident_specularity);

    static inline float get_metallic(MaterialID material_ID) { return m_materials[material_ID].metallic; }
    static void set_metallic(MaterialID material_ID, float metallic);
    static inline TextureID get_metallic_texture_ID(MaterialID material_ID) { return m_materials[material_ID].metallic_texture_ID; }
    static void set_metallic_texture_ID(MaterialID material_ID, TextureID metallic_texture_ID);

    static inline float get_coat(MaterialID material_ID) { return m_materials[material_ID].coat; }
    static void set_coat(MaterialID material_ID, float coat);
    static inline float get_coat_roughness(MaterialID material_ID) { return m_materials[material_ID].coat_roughness; }
    static void set_coat_roughness(MaterialID material_ID, float coat_roughness);

    // Coverage and cutout getters and setters.
    static inline float get_coverage(MaterialID material_ID) { return !is_cutout(material_ID) ? m_materials[material_ID].coverage : 1.0F; }
    static void set_coverage(MaterialID material_ID, float cutout_threshold) { if (!is_cutout(material_ID)) set_coverage_and_cutout_threshold(material_ID, cutout_threshold); }
    static inline float get_cutout_threshold(MaterialID material_ID) { return is_cutout(material_ID) ? m_materials[material_ID].coverage : 1.0f; }
    static void set_cutout_threshold(MaterialID material_ID, float cutout_threshold) { if (is_cutout(material_ID)) set_coverage_and_cutout_threshold(material_ID, cutout_threshold); }
    static inline TextureID get_coverage_texture_ID(MaterialID material_ID) { return m_materials[material_ID].coverage_texture_ID; }
    static void set_coverage_texture_ID(MaterialID material_ID, TextureID coverage_texture_ID);

    static inline float get_transmission(MaterialID material_ID) { return m_materials[material_ID].transmission; }
    static void set_transmission(MaterialID material_ID, float transmission);

    //-------------------------------------------------------------------------
    // Changes since last game loop tick.
    //-------------------------------------------------------------------------
    enum class Change : unsigned char {
        None = 0,
        Created = 1,
        Destroyed = 2,
        Updated = 4,
        ShadingModel = 8,
        All = Created | Destroyed | Updated
    };
    typedef Core::Bitmask<Change> Changes;

    static inline Changes get_changes(MaterialID material_ID) { return m_changes.get_changes(material_ID); }

    typedef std::vector<MaterialID>::iterator ChangedIterator;
    static Core::Iterable<ChangedIterator> get_changed_materials() { return m_changes.get_changed_resources(); }

    static void reset_change_notifications() { m_changes.reset_change_notifications(); }

private:
    static void reserve_material_data(unsigned int new_capacity, unsigned int old_capacity);

    static void set_coverage_and_cutout_threshold(MaterialID material_ID, float coverage_and_cutout);

    static void flag_as_updated(MaterialID material_ID);

    static MaterialIDGenerator m_UID_generator;

    static std::string* m_names;
    static Data* m_materials;
    static Core::ChangeSet<Changes, MaterialID> m_changes;
};

// ---------------------------------------------------------------------------
// Material ID wrapper.
// ---------------------------------------------------------------------------
class Material final {
public:
    // -----------------------------------------------------------------------
    // Class management.
    // -----------------------------------------------------------------------
    Material() : m_ID(MaterialID::invalid_UID()) {}
    Material(MaterialID id) : m_ID(id) {}

    inline const MaterialID get_ID() const { return m_ID; }
    inline bool exists() const { return Materials::has(m_ID); }

    inline bool operator==(Material rhs) const { return m_ID == rhs.m_ID; }
    inline bool operator!=(Material rhs) const { return m_ID != rhs.m_ID; }

    // -----------------------------------------------------------------------
    // Getters and setters.
    // -----------------------------------------------------------------------
    inline std::string get_name() const { return Materials::get_name(m_ID); }
    inline void set_name(const std::string& name) { Materials::set_name(m_ID, name); }

    inline ShadingModel get_shading_model() { return Materials::get_shading_model(m_ID); }
    inline void set_shading_model(ShadingModel shading_model) { return Materials::set_shading_model(m_ID, shading_model); }

    inline void set_flags(Materials::Flags flags) { Materials::set_flags(m_ID, flags); }
    inline Materials::Flags get_flags() const { return Materials::get_flags(m_ID); }
    inline bool is_thin_walled() const { return get_flags().is_set(MaterialFlag::ThinWalled); }
    inline bool is_cutout() const { return get_flags().is_set(MaterialFlag::Cutout); }

    inline Math::RGB get_tint() const { return Materials::get_tint(m_ID); }
    inline void set_tint(Math::RGB tint) { Materials::set_tint(m_ID, tint); }
    inline float get_roughness() const { return Materials::get_roughness(m_ID); }
    inline void set_roughness(float roughness) { Materials::set_roughness(m_ID, roughness); }
    inline const Texture get_tint_roughness_texture() const { return Materials::get_tint_roughness_texture_ID(m_ID); }
    inline TextureID get_tint_roughness_texture_ID() const { return Materials::get_tint_roughness_texture_ID(m_ID); }
    inline void set_tint_roughness_texture(Texture tint_roughness_texture) { Materials::set_tint_roughness_texture_ID(m_ID, tint_roughness_texture.get_ID()); }
    inline bool has_tint_texture() const { return Materials::has_tint_texture(m_ID); }
    inline bool has_roughness_texture() const { return Materials::has_roughness_texture(m_ID); }

    inline float get_specularity() const { return Materials::get_specularity(m_ID); }
    inline void set_specularity(float specularity) { Materials::set_specularity(m_ID, specularity); }

    inline float get_metallic() const { return Materials::get_metallic(m_ID); }
    inline void set_metallic(float metallic) { Materials::set_metallic(m_ID, metallic); }
    inline const Texture get_metallic_texture() const { return Materials::get_metallic_texture_ID(m_ID); }
    inline TextureID get_metallic_texture_ID() const { return Materials::get_metallic_texture_ID(m_ID); }
    inline void set_metallic_texture(Texture metallic_texture) { Materials::set_metallic_texture_ID(m_ID, metallic_texture.get_ID()); }

    inline float get_coat() const { return Materials::get_coat(m_ID); }
    inline void set_coat(float coat) { Materials::set_coat(m_ID, coat); }
    inline float get_coat_roughness() const { return Materials::get_coat_roughness(m_ID); }
    inline void set_coat_roughness(float coat_roughness) { Materials::set_coat_roughness(m_ID, coat_roughness); }

    inline float get_coverage() const { return Materials::get_coverage(m_ID); }
    inline void set_coverage(float coverage) { Materials::set_coverage(m_ID, coverage); }
    inline float get_cutout_threshold() const { return Materials::get_cutout_threshold(m_ID); }
    inline void set_cutout_threshold(float cutout_threshold) { Materials::set_cutout_threshold(m_ID, cutout_threshold); }
    inline const Texture get_coverage_texture() const { return Materials::get_coverage_texture_ID(m_ID); }
    inline TextureID get_coverage_texture_ID() const { return Materials::get_coverage_texture_ID(m_ID); }
    inline void set_coverage_texture_ID(TextureID texture_ID) { Materials::set_coverage_texture_ID(m_ID, texture_ID); }

    inline float get_transmission() const { return Materials::get_transmission(m_ID); }
    inline void set_transmission(float transmission) { Materials::set_coverage(m_ID, transmission); }

    inline Materials::Changes get_changes() { return Materials::get_changes(m_ID); }

private:
    const MaterialID m_ID;
};

} // NS Assets
} // NS Bifrost

#endif // _BIFROST_ASSETS_MATERIAL_H_
