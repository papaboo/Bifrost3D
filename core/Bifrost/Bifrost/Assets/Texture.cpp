// Bifrost texture.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#include <Bifrost/Assets/Texture.h>
#include <Bifrost/Math/Constants.h>

#include <assert.h>

using namespace Bifrost::Math;

namespace Bifrost {
namespace Assets {

Textures::UIDGenerator Textures::m_UID_generator = UIDGenerator(0u);
Textures::Sampler* Textures::m_samplers = nullptr;
Core::ChangeSet<Textures::Changes, Textures::UID> Textures::m_changes;

void Textures::allocate(unsigned int capacity) {
    if (is_allocated())
        return;

    m_UID_generator = UIDGenerator(capacity);
    capacity = m_UID_generator.capacity();

    m_samplers = new Sampler[capacity];
    m_changes = Core::ChangeSet<Changes, UID>(capacity);

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
    m_changes.resize(0);
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

    const unsigned int copyable_elements = new_capacity < old_capacity ? new_capacity : old_capacity;
    m_samplers = resize_and_copy_array(m_samplers, new_capacity, copyable_elements);
    m_changes.resize(new_capacity);
}

void Textures::reserve(unsigned int new_capacity) {
    unsigned int old_capacity = capacity();
    m_UID_generator.reserve(new_capacity);
    reserve_image_data(m_UID_generator.capacity(), old_capacity);
}

Textures::UID Textures::create2D(Images::UID image_ID, MagnificationFilter magnification_filter, MinificationFilter minification_filter, WrapMode wrapmode_U, WrapMode wrapmode_V) {
    assert(m_samplers != nullptr);

    if (!Images::has(image_ID))
        return Textures::UID::invalid_UID();

    unsigned int old_capacity = m_UID_generator.capacity();
    UID id = m_UID_generator.generate();
    if (old_capacity != m_UID_generator.capacity())
        // The capacity has changed and the size of all arrays need to be adjusted.
        reserve_image_data(m_UID_generator.capacity(), old_capacity);

    m_samplers[id].image_ID = image_ID;
    m_samplers[id].type = Type::TwoD;
    m_samplers[id].magnification_filter = magnification_filter;
    m_samplers[id].minification_filter = minification_filter;
    m_samplers[id].wrapmode_U = wrapmode_U;
    m_samplers[id].wrapmode_V = wrapmode_V;
    m_samplers[id].wrapmode_W = WrapMode::Repeat;
    m_changes.set_change(id, Change::Created);

    return id;
}

void Textures::destroy(Textures::UID texture_ID) {
    if (m_UID_generator.erase(texture_ID))
        m_changes.set_change(texture_ID, Change::Destroyed);
}

//-----------------------------------------------------------------------------
// Sampling functions.
//-----------------------------------------------------------------------------

Math::RGBA sample2D(Textures::UID texture_ID, Vector2f texcoord, int mipmap_level) {
    Texture texture = texture_ID;

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
            Vector2ui pixel_coord = Vector2ui(unsigned int(texcoord.x * image.get_width(mipmap_level)),
                                              unsigned int(texcoord.y * image.get_height(mipmap_level)));
            return image.get_pixel(pixel_coord);
        } else { // MinificationFilter::Linear
            unsigned int width = image.get_width(mipmap_level), height = image.get_height(mipmap_level);
            texcoord = Vector2f(texcoord.x * float(width), texcoord.y * float(height)) - 0.5f;
            Vector2i lower_left_coord = Vector2i(int(texcoord.x), int(texcoord.y));
            float u_lerp = texcoord.x - float(lower_left_coord.x);
            if (u_lerp < 0.0f) u_lerp += 1.0f;
            float v_lerp = texcoord.y - float(lower_left_coord.y);
            if (v_lerp < 0.0f) v_lerp += 1.0f;

            auto lookup_pixel = [](int pixelcoord_x, int pixelcoord_y, int mipmap_level, Texture texture, Image image) {
                int width = image.get_width(mipmap_level);
                if (texture.get_wrapmode_U() == WrapMode::Clamp)
                    pixelcoord_x = clamp(pixelcoord_x, 0, width - 1);
                else // WrapMode::Repeat
                    pixelcoord_x = (pixelcoord_x + width) % width;

                int height = image.get_height(mipmap_level);
                if (texture.get_wrapmode_V() == WrapMode::Clamp)
                    pixelcoord_y = clamp(pixelcoord_y, 0, height - 1);
                else // WrapMode::Repeat
                    pixelcoord_y = (pixelcoord_y + height) % height;

                return image.get_pixel(Vector2ui(pixelcoord_x, pixelcoord_y), mipmap_level);
            };

            RGBA lower_texel = lerp(lookup_pixel(lower_left_coord.x, lower_left_coord.y, mipmap_level, texture, image),
                                    lookup_pixel(lower_left_coord.x + 1, lower_left_coord.y, mipmap_level, texture, image),
                                    u_lerp);

            RGBA upper_texel = lerp(lookup_pixel(lower_left_coord.x, lower_left_coord.y + 1, mipmap_level, texture, image),
                                    lookup_pixel(lower_left_coord.x + 1, lower_left_coord.y + 1, mipmap_level, texture, image),
                                    u_lerp);

            return lerp(lower_texel, upper_texel, v_lerp);
        }
    }
}

} // NS Assets
} // NS Bifrost
