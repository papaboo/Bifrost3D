// Cogwheel texture.
// ---------------------------------------------------------------------------
// Copyright (C) 2015-2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#include <Cogwheel/Assets/Texture.h>
#include <Cogwheel/Math/Constants.h>

#include <assert.h>

using namespace Cogwheel::Math;

namespace Cogwheel {
namespace Assets {

Textures::UIDGenerator Textures::m_UID_generator = UIDGenerator(0u);
Textures::Sampler* Textures::m_samplers = nullptr;
unsigned char* Textures::m_changes = nullptr;
std::vector<Textures::UID> Textures::m_textures_changed = std::vector<Textures::UID>(0);

void Textures::allocate(unsigned int capacity) {
    if (is_allocated())
        return;

    m_UID_generator = UIDGenerator(capacity);
    capacity = m_UID_generator.capacity();

    m_samplers = new Sampler[capacity];
    m_changes = new unsigned char[capacity];
    std::memset(m_changes, Changes::None, capacity);

    m_textures_changed.reserve(capacity / 4);

    // Allocate dummy element at 0.
    m_samplers[0].image_ID = Images::UID::invalid_UID();
    m_samplers[0].type = Type::OneD;
    m_samplers[0].magnification_filter = MagnificationFilter::None;
    m_samplers[0].minification_filter = MinificationFilter::None;
    m_samplers[0].wrapmode_U = WrapMode::Repeat;
    m_samplers[0].wrapmode_V = WrapMode::Repeat;
    m_samplers[0].wrapmode_W = WrapMode::Repeat;
}

void Textures::deallocate() {
    if (!is_allocated())
        return;

    m_UID_generator = UIDGenerator(0u);
    delete[] m_samplers; m_samplers = nullptr;
    delete[] m_changes; m_changes = nullptr;
    m_textures_changed.resize(0); m_textures_changed.shrink_to_fit();
}

template <typename T>
static inline T* resize_and_copy_array(T* old_array, unsigned int new_capacity, unsigned int copyable_elements) {
    T* new_array = new T[new_capacity];
    std::copy(old_array, old_array + copyable_elements, new_array);
    delete[] old_array;
    return new_array;
}

void Textures::reserve_image_data(unsigned int new_capacity, unsigned int old_capacity) {
    assert(m_samplers != nullptr);
    assert(m_changes != nullptr);

    const unsigned int copyable_elements = new_capacity < old_capacity ? new_capacity : old_capacity;
    m_samplers = resize_and_copy_array(m_samplers, new_capacity, copyable_elements);
    m_changes = resize_and_copy_array(m_changes, new_capacity, copyable_elements);
    if (copyable_elements < new_capacity) // We need to zero the new change masks.
        std::memset(m_changes + copyable_elements, Changes::None, new_capacity - copyable_elements);
}

void Textures::reserve(unsigned int new_capacity) {
    unsigned int old_capacity = capacity();
    m_UID_generator.reserve(new_capacity);
    reserve_image_data(m_UID_generator.capacity(), old_capacity);
}

bool Textures::has(Textures::UID image_ID) {
    return m_UID_generator.has(image_ID) && !(m_changes[image_ID] & Changes::Destroyed);
}

Textures::UID Textures::create2D(Images::UID image_ID, MagnificationFilter magnification_filter, MinificationFilter minification_filter, WrapMode wrapmode_U, WrapMode wrapmode_V) {
    assert(m_samplers != nullptr);
    assert(m_changes != nullptr);

    unsigned int old_capacity = m_UID_generator.capacity();
    UID id = m_UID_generator.generate();
    if (old_capacity != m_UID_generator.capacity())
        // The capacity has changed and the size of all arrays need to be adjusted.
        reserve_image_data(m_UID_generator.capacity(), old_capacity);

    if (m_changes[id] == Changes::None)
        m_textures_changed.push_back(id);

    m_samplers[id].image_ID = image_ID;
    m_samplers[id].type = Type::TwoD;
    m_samplers[id].magnification_filter = magnification_filter;
    m_samplers[id].minification_filter = minification_filter;
    m_samplers[id].wrapmode_U = wrapmode_U;
    m_samplers[id].wrapmode_V = wrapmode_V;
    m_samplers[id].wrapmode_W = WrapMode::Repeat;
    m_changes[id] = Changes::Created;

    return id;
}

void Textures::destroy(Textures::UID texture_ID) {
    if (m_UID_generator.erase(texture_ID)) {
        if (m_changes[texture_ID] == Changes::None)
            m_textures_changed.push_back(texture_ID);

        m_changes[texture_ID] = Changes::Destroyed;
    }
}

void Textures::reset_change_notifications() {
    std::memset(m_changes, Changes::None, capacity());
    m_textures_changed.resize(0);
}

//-----------------------------------------------------------------------------
// Sampling functions.
//-----------------------------------------------------------------------------

Math::RGBA sample2D(Textures::UID texture_ID, Vector2f texcoord) {
    TextureND texture = texture_ID;

    { // Modify tex coord based on wrap mode.
        if (texture.get_wrapmode_U() == WrapMode::Clamp)
            texcoord.x = clamp(texcoord.x, 0.0f, nearly_one);
        else { // WrapMode::Repeat
            texcoord.x -= (int)texcoord.x;
            if (texcoord.x < -0.0f)
                texcoord.x += 1.0f;
        }

        if (texture.get_wrapmode_V() == WrapMode::Clamp)
            texcoord.y = clamp(texcoord.y, 0.0f, nearly_one);
        else { // WrapMode::Repeat
            texcoord.y -= (int)texcoord.y;
            if (texcoord.y < -0.0f)
                texcoord.y += 1.0f;
        }
    }

    { // Get pixels.
        Image image = texture.get_image();

        if (texture.get_minification_filter() == MinificationFilter::None) {
            Vector2ui pixel_coord = Vector2ui(unsigned int(texcoord.x * image.get_width()),
                                              unsigned int(texcoord.y * image.get_height()));
            return image.get_pixel(pixel_coord);
        } else { // MinificationFilter::Linear
            unsigned int width = image.get_width(), height = image.get_height();
            texcoord = Vector2f((texcoord.x + 1.0f) * float(width), (texcoord.y + 1.0f) * float(height)) - 0.5f;
            Vector2i lower_left_coord = Vector2i(int(texcoord.x), int(texcoord.y));
            
            auto lookup_pixel = [](int pixelcoord_x, int pixelcoord_y, TextureND texture, Image image) {
                if (texture.get_wrapmode_U() == WrapMode::Clamp)
                    pixelcoord_x = clamp(pixelcoord_x, 0, int(image.get_width()));
                else // WrapMode::Repeat
                    pixelcoord_x = pixelcoord_x % image.get_width();

                if (texture.get_wrapmode_V() == WrapMode::Clamp)
                    pixelcoord_y = clamp(pixelcoord_y, 0, int(image.get_height()));
                else // WrapMode::Repeat
                    pixelcoord_y = pixelcoord_y % image.get_height();

                return image.get_pixel(Vector2ui(pixelcoord_x, pixelcoord_y));
            };

            float u_lerp = abs(texcoord.x - float(lower_left_coord.x));
            RGBA lower_texel = lerp(lookup_pixel(lower_left_coord.x, lower_left_coord.y, texture, image),
                                    lookup_pixel(lower_left_coord.x+1, lower_left_coord.y, texture, image), 
                                    u_lerp);

            RGBA upper_texel = lerp(lookup_pixel(lower_left_coord.x, lower_left_coord.y+1, texture, image),
                                    lookup_pixel(lower_left_coord.x+1, lower_left_coord.y+1, texture, image),
                                    u_lerp);
            
            float v_lerp = abs(texcoord.y - float(lower_left_coord.y));
            return lerp(lower_texel, upper_texel, v_lerp);
        }
    }
}

} // NS Assets
} // NS Cogwheel