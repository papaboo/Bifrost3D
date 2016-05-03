// Cogwheel image.
// ---------------------------------------------------------------------------
// Copyright (C) 2015-2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#include <Cogwheel/Assets/Image.h>

#include <assert.h>

using namespace Cogwheel::Math;

namespace Cogwheel {
namespace Assets {

Images::UIDGenerator Images::m_UID_generator = UIDGenerator(0u);
Images::MetaInfo* Images::m_metainfo = nullptr;
Images::PixelData* Images::m_pixels = nullptr;
unsigned char* Images::m_changes = nullptr;
std::vector<Images::UID> Images::m_images_changed = std::vector<Images::UID>(0);

void Images::allocate(unsigned int capacity) {
    if (is_allocated())
        return;

    m_UID_generator = UIDGenerator(capacity);
    capacity = m_UID_generator.capacity();

    m_metainfo = new MetaInfo[capacity];
    m_pixels = new PixelData[capacity];
    m_changes = new unsigned char[capacity];
    std::memset(m_changes, Changes::None, capacity);

    m_images_changed.reserve(capacity / 4);

    // Allocate dummy element at 0.
    MetaInfo info = { "Dummy image", 0u, 0u, 0u, 0u, PixelFormat::Unknown };
    m_metainfo[0] = info;
    m_pixels[0] = nullptr;
}

void Images::deallocate() {
    if (!is_allocated())
        return;

    m_UID_generator = UIDGenerator(0u);
    delete[] m_metainfo; m_metainfo = nullptr;
    delete[] m_pixels; m_pixels = nullptr;
    delete[] m_changes; m_changes = nullptr;
    m_images_changed.resize(0); m_images_changed.shrink_to_fit();
}

template <typename T>
static inline T* resize_and_copy_array(T* old_array, unsigned int new_capacity, unsigned int copyable_elements) {
    T* new_array = new T[new_capacity];
    std::copy(old_array, old_array + copyable_elements, new_array);
    delete[] old_array;
    return new_array;
}

void Images::reserve_image_data(unsigned int new_capacity, unsigned int old_capacity) {
    assert(m_metainfo != nullptr);
    assert(m_pixels != nullptr);
    assert(m_changes != nullptr);

    const unsigned int copyable_elements = new_capacity < old_capacity ? new_capacity : old_capacity;
    m_metainfo = resize_and_copy_array(m_metainfo, new_capacity, copyable_elements);
    m_pixels = resize_and_copy_array(m_pixels, new_capacity, copyable_elements);
    m_changes = resize_and_copy_array(m_changes, new_capacity, copyable_elements);
    if (copyable_elements < new_capacity)
        // We need to zero the new change masks.
        std::memset(m_changes + copyable_elements, Changes::None, new_capacity - copyable_elements);
}

void Images::reserve(unsigned int new_capacity) {
    unsigned int old_capacity = capacity();
    m_UID_generator.reserve(new_capacity);
    reserve_image_data(m_UID_generator.capacity(), old_capacity);
}

bool Images::has(Images::UID image_ID) {
    return m_UID_generator.has(image_ID) && !(m_changes[image_ID] & Changes::Destroyed);
}

static inline Images::PixelData allocate_pixels(PixelFormat format, unsigned int pixel_count) {
    switch (format) {
    case PixelFormat::RGB24:
        return new unsigned char[3 * pixel_count];
    case PixelFormat::RGBA32:
        return new unsigned char[4 * pixel_count];
    case PixelFormat::RGB_Float:
        return new float[3 * pixel_count];
    case PixelFormat::RGBA_Float:
        return new float[4 * pixel_count];
    case PixelFormat::Unknown:
        return nullptr;
    }
    return nullptr;
}

Images::UID Images::create(const std::string& name, PixelFormat format, Math::Vector3ui size, unsigned int mipmap_count) {
    assert(m_metainfo != nullptr);
    assert(m_pixels != nullptr);
    assert(m_changes != nullptr);

    unsigned int old_capacity = m_UID_generator.capacity();
    UID id = m_UID_generator.generate();
    if (old_capacity != m_UID_generator.capacity())
        // The capacity has changed and the size of all arrays need to be adjusted.
        reserve_image_data(m_UID_generator.capacity(), old_capacity);

    if (m_changes[id] == Changes::None)
        m_images_changed.push_back(id);

    // TODO Clamp mipmap count here in case it is bigger than what the image sizes allow.

    m_metainfo[id].name = name;
    m_metainfo[id].pixel_format = format;
    m_metainfo[id].width = size.x;
    m_metainfo[id].height = size.y;
    m_metainfo[id].depth = size.z;
    m_metainfo[id].mipmap_count = mipmap_count;
    unsigned int pixel_count = size.x * size.y * size.z;
    m_pixels[id] = allocate_pixels(format, pixel_count);
    m_changes[id] = Changes::Created;

    return id;
}

void Images::destroy(Images::UID image_ID) {
    if (m_UID_generator.erase(image_ID)) {
        delete[] m_pixels[image_ID];

        if (m_changes[image_ID] == Changes::None)
            m_images_changed.push_back(image_ID);

        m_changes[image_ID] |= Changes::Destroyed;
    }
}

Math::RGBA Images::get_pixel(Images::UID image_ID, unsigned int index, int mipmap_level) {
    switch (m_metainfo[image_ID].pixel_format) {
    case PixelFormat::RGB24: {
        unsigned char* pixel = ((unsigned char*)m_pixels[image_ID]) + index * 3;
        return RGBA(pixel[0] / 255.0f, pixel[1] / 255.0f, pixel[2] / 255.0f, 1.0f);
    }
    case PixelFormat::RGBA32: {
        unsigned char* pixel = ((unsigned char*)m_pixels[image_ID]) + index * 4;
        return RGBA(pixel[0] / 255.0f, pixel[1] / 255.0f, pixel[2] / 255.0f, pixel[3] / 255.0f);
    }
    case PixelFormat::RGB_Float: {
        float* pixel = ((float*)m_pixels[image_ID]) + index * 3;
        return RGBA(pixel[0], pixel[1], pixel[2], 1.0f);
    }
    case PixelFormat::RGBA_Float: {
        float* pixel = ((float*)m_pixels[image_ID]) + index * 4;
        return RGBA(pixel[0], pixel[1], pixel[2], pixel[3]);
    }
    case PixelFormat::Unknown:
        return RGBA::red();
    }
    return RGBA::red();
}

RGBA Images::get_pixel(Images::UID image_ID, Math::Vector2ui index, int mipmap_level) {
    const MetaInfo& info = m_metainfo[image_ID];
    const unsigned int pixel_index = index.x + info.width * index.y;
    return get_pixel(image_ID, pixel_index, mipmap_level);
}

RGBA Images::get_pixel(Images::UID image_ID, Math::Vector3ui index, int mipmap_level) {
    const MetaInfo& info = m_metainfo[image_ID];
    const unsigned int pixel_index = index.x + info.width * (index.y + info.height * index.z);
    return get_pixel(image_ID, pixel_index, mipmap_level);
}

void Images::set_pixel(Images::UID image_ID, Math::RGBA color, unsigned int index, int mipmap_level) {
    switch (m_metainfo[image_ID].pixel_format) {
    case PixelFormat::RGB24: {
        unsigned char* pixel = ((unsigned char*)m_pixels[image_ID]) + index * 3;
        pixel[0] = unsigned char(Math::clamp(color.r * 255.0f, 0.0f, 255.0f));
        pixel[1] = unsigned char(Math::clamp(color.g * 255.0f, 0.0f, 255.0f));
        pixel[2] = unsigned char(Math::clamp(color.b * 255.0f, 0.0f, 255.0f));
        break;
    }
    case PixelFormat::RGBA32: {
        unsigned char* pixel = ((unsigned char*)m_pixels[image_ID]) + index * 4;
        pixel[0] = unsigned char(Math::clamp(color.r * 255.0f, 0.0f, 255.0f));
        pixel[1] = unsigned char(Math::clamp(color.g * 255.0f, 0.0f, 255.0f));
        pixel[2] = unsigned char(Math::clamp(color.b * 255.0f, 0.0f, 255.0f));
        pixel[3] = unsigned char(Math::clamp(color.a * 255.0f, 0.0f, 255.0f));
        break;
    }
    case PixelFormat::RGB_Float: {
        float* pixel = ((float*)m_pixels[image_ID]) + index * 3;
        pixel[0] = color.r;
        pixel[1] = color.g;
        pixel[2] = color.b;
        break;
    }
    case PixelFormat::RGBA_Float: {
        float* pixel = ((float*)m_pixels[image_ID]) + index * 4;
        pixel[0] = color.r;
        pixel[1] = color.g;
        pixel[2] = color.b;
        pixel[3] = color.a;
        break;
    }
    case PixelFormat::Unknown:
        ;
    }

    flag_as_changed(image_ID, Changes::PixelsUpdated);
}

void Images::set_pixel(Images::UID image_ID, Math::RGBA color, Math::Vector2ui index, int mipmap_level) {
    const MetaInfo& info = m_metainfo[image_ID];
#if _DEBUG
    if (index.x >= info.width || index.y >= info.height) {
        printf("Pixel index [%u, %u] is outside the bounds [%u, %u]\n", index.x, index.y, info.width, info.height);
        index.x = Math::min(index.x, info.width - 1u);
        index.y = Math::min(index.y, info.height - 1u);
    }
#endif
    const unsigned int pixel_index = index.x + info.width * index.y;
    set_pixel(image_ID, color, pixel_index, mipmap_level);
}

void Images::set_pixel(Images::UID image_ID, Math::RGBA color, Math::Vector3ui index, int mipmap_level) {
    const MetaInfo& info = m_metainfo[image_ID];
#if _DEBUG
    if (index.x >= info.width || index.y >= info.height) {
        printf("Pixel index [%u, %u, %u] is outside the bounds [%u, %u, %u]\n", index.x, index.y, index.z, info.width, info.height, info.depth);
        index.x = Math::min(index.x, info.width - 1u);
        index.y = Math::min(index.y, info.height - 1u);
        index.z = Math::min(index.z, info.depth - 1u);
    }
#endif
    const unsigned int pixel_index = index.x + info.width * (index.y + info.height * index.z);
    set_pixel(image_ID, color, pixel_index, mipmap_level);
}

void Images::flag_as_changed(Images::UID image_ID, unsigned int change) {
    if (m_changes[image_ID] == Changes::None)
        m_images_changed.push_back(image_ID);

    m_changes[image_ID] |= change;
}

void Images::reset_change_notifications() {
    std::memset(m_changes, Changes::None, capacity());
    m_images_changed.resize(0);
}


//*****************************************************************************
// Image Utilities
//*****************************************************************************

namespace ImageUtils {

Images::UID change_format(Images::UID image_ID, PixelFormat new_format) {
    unsigned int width = Images::get_width(image_ID);
    unsigned int height = Images::get_height(image_ID);
    unsigned int depth = Images::get_depth(image_ID);
    unsigned int mipmap_count = Images::get_mipmap_count(image_ID);
    Images::UID new_image_ID = Images::create(Images::get_name(image_ID), new_format, Math::Vector3ui(width, height, depth), mipmap_count);

    // TODO Specialize for most common formats.
    for (unsigned int m = 0; m < mipmap_count; ++m) {
        unsigned int pixel_count = Images::get_pixel_count(image_ID, m);
        for (unsigned int p = 0; p < pixel_count; ++p) {
            RGBA pixel = Images::get_pixel(image_ID, p, m);
            Images::set_pixel(new_image_ID, pixel, p, m);
        }
    }

    return new_image_ID;
}

} // NS ImageUtils

} // NS Assets
} // NS Cogwheel