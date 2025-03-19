// Bifrost rendered material.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#include <Bifrost/Assets/Material.h>

#include <assert.h>

using namespace Bifrost::Math;

namespace Bifrost {
namespace Assets {

MaterialIDGenerator Materials::m_UID_generator = MaterialIDGenerator(0u);
std::string* Materials::m_names = nullptr;
Materials::Data* Materials::m_materials = nullptr;
Core::ChangeSet<Materials::Changes, MaterialID> Materials::m_changes;

#ifdef NDEBUG 
__forceinline void assert_coverage_texture(Texture coverage_tex) {}
__forceinline void assert_metallic_texture(Texture metallic_tex) {}
__forceinline void assert_tint_roughness_texture(Texture metallic_tex) {}
#else
__forceinline void assert_coverage_texture(Texture coverage_tex) {
    assert(!coverage_tex.exists() || coverage_tex.get_image().get_pixel_format() == PixelFormat::Alpha8);
}
__forceinline void assert_metallic_texture(Texture metallic_tex) {
    assert(!metallic_tex.exists() || metallic_tex.get_image().get_pixel_format() == PixelFormat::Alpha8);
}
__forceinline void assert_tint_roughness_texture(Texture tint_roughness_tex) {
    assert(!tint_roughness_tex.exists() || tint_roughness_tex.get_image().get_pixel_format() != PixelFormat::Unknown);
}
#endif

void Materials::allocate(unsigned int capacity) {
    if (is_allocated())
        return;

    m_UID_generator = MaterialIDGenerator(capacity);
    capacity = m_UID_generator.capacity();

    m_names = new std::string[capacity];
    m_materials = new Data[capacity];
    m_changes = Core::ChangeSet<Changes, MaterialID>(capacity);

    // Allocate dummy element at 0.
    m_names[0] = "Dummy Material";
    Data dummy_data = {};
    dummy_data.coverage = 1.0f;
    dummy_data.tint = Math::RGB::red();
    m_materials[0] = dummy_data;
}

void Materials::deallocate() {
    if (!is_allocated())
        return;

    m_UID_generator = MaterialIDGenerator(0u);
    delete[] m_names; m_names = nullptr;
    delete[] m_materials; m_materials = nullptr;

    m_changes.resize(0);
}

template <typename T>
static inline T* resize_and_copy_array(T* old_array, unsigned int new_capacity, unsigned int copyable_elements) {
    T* new_array = new T[new_capacity];
    std::copy(old_array, old_array + copyable_elements, new_array);
    delete[] old_array;
    return new_array;
}

void Materials::reserve_material_data(unsigned int new_capacity, unsigned int old_capacity) {
    assert(m_names != nullptr);
    assert(m_materials != nullptr);

    const unsigned int copyable_elements = new_capacity < old_capacity ? new_capacity : old_capacity;
    m_names = resize_and_copy_array(m_names, new_capacity, copyable_elements);
    m_materials = resize_and_copy_array(m_materials, new_capacity, copyable_elements);
    m_changes.resize(new_capacity);
}

void Materials::reserve(unsigned int new_capacity) {
    unsigned int old_capacity = capacity();
    m_UID_generator.reserve(new_capacity);
    reserve_material_data(m_UID_generator.capacity(), old_capacity);
}

MaterialID Materials::create(const std::string& name, const Data& data) {
    assert(m_names != nullptr);
    assert(m_materials != nullptr);
    assert_coverage_texture(data.coverage_texture_ID);
    assert_metallic_texture(data.metallic_texture_ID);
    assert_tint_roughness_texture(data.tint_roughness_texture_ID);

    unsigned int old_capacity = m_UID_generator.capacity();
    MaterialID id = m_UID_generator.generate();
    if (old_capacity != m_UID_generator.capacity())
        // The capacity has changed and the size of all arrays need to be adjusted.
        reserve_material_data(m_UID_generator.capacity(), old_capacity);

    m_names[id] = name;
    m_materials[id] = data;
    m_changes.set_change(id, Change::Created);

    return id;
}

void Materials::destroy(MaterialID material_ID) {
    if (has(material_ID))
        m_changes.add_change(material_ID, Change::Destroyed);
}

void Materials::set_shading_model(MaterialID material_ID, ShadingModel shading_model) {
    m_materials[material_ID].shading_model = shading_model;
    m_changes.add_change(material_ID, Change::ShadingModel);
}

void Materials::set_flags(MaterialID material_ID, Flags flags) {
    m_materials[material_ID].flags = flags;
    flag_as_updated(material_ID);
}

void Materials::set_tint(MaterialID material_ID, Math::RGB tint) {
    m_materials[material_ID].tint = tint;
    flag_as_updated(material_ID);
}

void Materials::set_roughness(MaterialID material_ID, float roughness) {
    m_materials[material_ID].roughness = roughness;
    flag_as_updated(material_ID);
}

void Materials::set_tint_roughness_texture_ID(MaterialID material_ID, TextureID tint_roughness_texture_ID) {
    assert_tint_roughness_texture(tint_roughness_texture_ID);
    m_materials[material_ID].tint_roughness_texture_ID = tint_roughness_texture_ID;
    flag_as_updated(material_ID);
}

bool Materials::has_tint_texture(MaterialID material_ID) {
    Texture tex = m_materials[material_ID].tint_roughness_texture_ID;
    if (!tex.exists() || !tex.get_image().exists())
        return false;

    auto pixel_format = tex.get_image().get_pixel_format();
    return channel_count(pixel_format) >= 3;
}

bool Materials::has_roughness_texture(MaterialID material_ID) {
    Texture tex = m_materials[material_ID].tint_roughness_texture_ID;
    if (!tex.exists() || !tex.get_image().exists())
        return false;

    auto pixel_format = tex.get_image().get_pixel_format();
    return channel_count(pixel_format) == 4 || pixel_format == PixelFormat::Roughness8;
}

void Materials::set_specularity(MaterialID material_ID, float incident_specularity) {
    m_materials[material_ID].specularity = incident_specularity;
    flag_as_updated(material_ID);
}

void Materials::set_metallic(MaterialID material_ID, float metallic) {
    m_materials[material_ID].metallic = metallic;
    flag_as_updated(material_ID);
}

void Materials::set_metallic_texture_ID(MaterialID material_ID, TextureID metallic_texture_ID) {
    assert_metallic_texture(metallic_texture_ID);
    m_materials[material_ID].metallic_texture_ID = metallic_texture_ID;
    flag_as_updated(material_ID);
}

void Materials::set_coat(MaterialID material_ID, float coat) {
    m_materials[material_ID].coat = coat;
    flag_as_updated(material_ID);
}

void Materials::set_coat_roughness(MaterialID material_ID, float coat_roughness) {
    m_materials[material_ID].coat_roughness = coat_roughness;
    flag_as_updated(material_ID);
}

void Materials::set_coverage_and_cutout_threshold(MaterialID material_ID, float coverage_and_cutout) {
    m_materials[material_ID].coverage = coverage_and_cutout;
    flag_as_updated(material_ID);
}

void Materials::set_coverage_texture_ID(MaterialID material_ID, TextureID coverage_texture_ID) {
    assert_coverage_texture(coverage_texture_ID);
    m_materials[material_ID].coverage_texture_ID = coverage_texture_ID;
    flag_as_updated(material_ID);
}

void Materials::set_transmission(MaterialID material_ID, float transmission) {
    m_materials[material_ID].transmission = transmission;
    flag_as_updated(material_ID);
}

void Materials::flag_as_updated(MaterialID material_ID) {
    m_changes.add_change(material_ID, Change::Updated);
}

void Materials::reset_change_notifications() {
    for (MaterialID material_ID : get_changed_materials())
        if (get_changes(material_ID).is_set(Change::Destroyed))
            m_UID_generator.erase(material_ID);
    m_changes.reset_change_notifications();
}

} // NS Assets
} // NS Bifrost
