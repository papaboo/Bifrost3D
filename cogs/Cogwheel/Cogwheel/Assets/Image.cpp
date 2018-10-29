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
Core::ChangeSet<Images::Changes, Images::UID> Images::m_changes;

void Images::allocate(unsigned int capacity) {
    if (is_allocated())
        return;

    m_UID_generator = UIDGenerator(capacity);
    capacity = m_UID_generator.capacity();

    m_metainfo = new MetaInfo[capacity];
    m_pixels = new PixelData[capacity];
    m_changes = Core::ChangeSet<Changes, UID>(capacity);

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
    m_changes.resize(0);
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

    const unsigned int copyable_elements = new_capacity < old_capacity ? new_capacity : old_capacity;
    m_metainfo = resize_and_copy_array(m_metainfo, new_capacity, copyable_elements);
    m_pixels = resize_and_copy_array(m_pixels, new_capacity, copyable_elements);
    m_changes.resize(new_capacity);
}

void Images::reserve(unsigned int new_capacity) {
    unsigned int old_capacity = capacity();
    m_UID_generator.reserve(new_capacity);
    reserve_image_data(m_UID_generator.capacity(), old_capacity);
}

bool Images::has(Images::UID image_ID) {
    return m_UID_generator.has(image_ID) && m_changes.get_changes(image_ID) != Change::Destroyed;
}

static inline Images::PixelData allocate_pixels(PixelFormat format, unsigned int pixel_count) {
    switch (format) {
    case PixelFormat::I8:
        return new unsigned char[pixel_count];
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

Images::UID Images::create3D(const std::string& name, PixelFormat format, float gamma, Vector3ui size, unsigned int mipmap_count) {
    assert(m_metainfo != nullptr);
    assert(m_pixels != nullptr);
    assert(mipmap_count > 0u);

    unsigned int old_capacity = m_UID_generator.capacity();
    UID id = m_UID_generator.generate();
    if (old_capacity != m_UID_generator.capacity())
        // The capacity has changed and the size of all arrays need to be adjusted.
        reserve_image_data(m_UID_generator.capacity(), old_capacity);

    // Only apply gamma to images that store colors.
    if (format == PixelFormat::I8)
        gamma = 1.0f;

    MetaInfo& metainfo = m_metainfo[id];
    metainfo.name = name;
    metainfo.pixel_format = format;
    metainfo.gamma = gamma;
    metainfo.width = size.x;
    metainfo.height = size.y;
    metainfo.depth = size.z;
    unsigned int total_pixel_count = 0u;
    unsigned int mip_count = 0u;
    while (mip_count != mipmap_count) {
        unsigned int mip_pixel_count = Images::get_width(id, mip_count) * Images::get_height(id, mip_count) * Images::get_depth(id, mip_count);
        total_pixel_count += mip_pixel_count;
        ++mip_count;
        if (mip_pixel_count == 1u)
            break;
    }
    metainfo.mipmap_count = mip_count;
    metainfo.is_mipmapable = false;
    m_pixels[id] = allocate_pixels(format, total_pixel_count);
    m_changes.set_change(id, Change::Created);

    return id;
}

Images::UID Images::create2D(const std::string& name, PixelFormat format, float gamma, Math::Vector2ui size, PixelData& pixels) {
    assert(m_metainfo != nullptr);
    assert(m_pixels != nullptr);

    unsigned int old_capacity = m_UID_generator.capacity();
    UID id = m_UID_generator.generate();
    if (old_capacity != m_UID_generator.capacity())
        // The capacity has changed and the size of all arrays need to be adjusted.
        reserve_image_data(m_UID_generator.capacity(), old_capacity);

    // Only apply gamma to images that store colors.
    if (format == PixelFormat::I8)
        gamma = 1.0f;

    MetaInfo& metainfo = m_metainfo[id];
    metainfo.name = name;
    metainfo.pixel_format = format;
    metainfo.gamma = gamma;
    metainfo.width = size.x;
    metainfo.height = size.y;
    metainfo.depth = 1u;

    metainfo.mipmap_count = 1u;
    metainfo.is_mipmapable = false;
    m_pixels[id] = pixels; pixels = nullptr; // Take ownership of pixels.
    m_changes.set_change(id, Change::Created);

    return id;
}

void Images::destroy(Images::UID image_ID) {
    if (m_UID_generator.erase(image_ID)) {
        delete[] m_pixels[image_ID]; m_pixels[image_ID] = nullptr;
        m_changes.set_change(image_ID, Change::Destroyed);
    }
}

void Images::set_mipmapable(Images::UID image_ID, bool value) { 
    // Only set as mipmapable if no mipmaps exist.
    value &= m_metainfo[image_ID].mipmap_count == 1;

    if (m_metainfo[image_ID].is_mipmapable == value)
        return;

    m_metainfo[image_ID].is_mipmapable = value;

    m_changes.add_change(image_ID, Change::Mipmapable);
}

Images::PixelData Images::get_pixels(Images::UID image_ID, int mipmap_level) {
    char* pixel_data = (char*)m_pixels[image_ID];
    int bytes_pr_pixel = size_of(get_pixel_format(image_ID));
    for (int l = 0; l < mipmap_level; ++l) {
        pixel_data += get_pixel_count(image_ID, l) * bytes_pr_pixel;
    }
    return pixel_data;
}

static RGBA get_nonlinear_pixel(Images::UID image_ID, unsigned int index) {
    Images::PixelData pixels = Images::get_pixels(image_ID);
    switch (Images::get_pixel_format(image_ID)) {
    case PixelFormat::I8: {
        unsigned char* pixel = ((unsigned char*)pixels) + index;
        float intensity = pixel[0] / 255.0f;
        return RGBA(intensity, intensity, intensity, intensity);
    }
    case PixelFormat::RGB24: {
        unsigned char* pixel = ((unsigned char*)pixels) + index * 3;
        return RGBA(pixel[0] / 255.0f, pixel[1] / 255.0f, pixel[2] / 255.0f, 1.0f);
    }
    case PixelFormat::RGBA32: {
        unsigned char* pixel = ((unsigned char*)pixels) + index * 4;
        return RGBA(pixel[0] / 255.0f, pixel[1] / 255.0f, pixel[2] / 255.0f, pixel[3] / 255.0f);
    }
    case PixelFormat::RGB_Float: {
        float* pixel = ((float*)pixels) + index * 3;
        return RGBA(pixel[0], pixel[1], pixel[2], 1.0f);
    }
    case PixelFormat::RGBA_Float: {
        float* pixel = ((float*)pixels) + index * 4;
        return RGBA(pixel[0], pixel[1], pixel[2], pixel[3]);
    }
    case PixelFormat::Unknown:
        return RGBA::red();
    }
    return RGBA::red();
}

RGBA Images::get_pixel(Images::UID image_ID, unsigned int index, unsigned int mipmap_level) {
    assert(index < Images::get_pixel_count(image_ID, mipmap_level));

    while (mipmap_level)
        index += Images::get_width(image_ID, --mipmap_level);
    RGBA nonlinear_pixel = get_nonlinear_pixel(image_ID, index);
    return gammacorrect(nonlinear_pixel, Images::get_gamma(image_ID));
}

RGBA Images::get_pixel(Images::UID image_ID, Vector2ui index, unsigned int mipmap_level) {
    assert(index.x < Images::get_width(image_ID, mipmap_level));
    assert(index.y < Images::get_height(image_ID, mipmap_level));

    Image image = image_ID;
    unsigned int pixel_index = index.x + image.get_width(mipmap_level) * index.y;

    while (mipmap_level) {
        --mipmap_level;
        pixel_index += image.get_width(mipmap_level) * image.get_height(mipmap_level);
    }
    RGBA nonlinear_pixel = get_nonlinear_pixel(image_ID, pixel_index);
    return gammacorrect(nonlinear_pixel, Images::get_gamma(image_ID));
}

RGBA Images::get_pixel(Images::UID image_ID, Vector3ui index, unsigned int mipmap_level) {
    assert(index.x < Images::get_width(image_ID, mipmap_level));
    assert(index.y < Images::get_height(image_ID, mipmap_level));
    assert(index.z < Images::get_depth(image_ID, mipmap_level));

    Image image = image_ID;
    unsigned int pixel_index = index.x + image.get_width(mipmap_level) * (index.y + image.get_height(mipmap_level) * index.z);
    while (mipmap_level) {
        --mipmap_level;
        pixel_index += image.get_width(mipmap_level) * image.get_height(mipmap_level) * image.get_depth(mipmap_level);
    }
    RGBA nonlinear_pixel = get_nonlinear_pixel(image_ID, pixel_index);
    return gammacorrect(nonlinear_pixel, Images::get_gamma(image_ID));
}

static void set_linear_pixel(Images::UID image_ID, RGBA color, unsigned int index) {
    color = gammacorrect(color, 1.0f / Images::get_gamma(image_ID));
    Images::PixelData pixels = Images::get_pixels(image_ID);
    switch (Images::get_pixel_format(image_ID)) {
    case PixelFormat::I8: {
        unsigned char* pixel = ((unsigned char*)pixels) + index;
        pixel[0] = unsigned char(clamp(color.r * 255.0f, 0.0f, 255.0f));
        break;
    }
    case PixelFormat::RGB24: {
        unsigned char* pixel = ((unsigned char*)pixels) + index * 3;
        pixel[0] = unsigned char(clamp(color.r * 255.0f, 0.0f, 255.0f));
        pixel[1] = unsigned char(clamp(color.g * 255.0f, 0.0f, 255.0f));
        pixel[2] = unsigned char(clamp(color.b * 255.0f, 0.0f, 255.0f));
        break;
    }
    case PixelFormat::RGBA32: {
        unsigned char* pixel = ((unsigned char*)pixels) + index * 4;
        pixel[0] = unsigned char(clamp(color.r * 255.0f, 0.0f, 255.0f));
        pixel[1] = unsigned char(clamp(color.g * 255.0f, 0.0f, 255.0f));
        pixel[2] = unsigned char(clamp(color.b * 255.0f, 0.0f, 255.0f));
        pixel[3] = unsigned char(clamp(color.a * 255.0f, 0.0f, 255.0f));
        break;
    }
    case PixelFormat::RGB_Float: {
        float* pixel = ((float*)pixels) + index * 3;
        pixel[0] = color.r;
        pixel[1] = color.g;
        pixel[2] = color.b;
        break;
    }
    case PixelFormat::RGBA_Float: {
        float* pixel = ((float*)pixels) + index * 4;
        pixel[0] = color.r;
        pixel[1] = color.g;
        pixel[2] = color.b;
        pixel[3] = color.a;
        break;
    }
    case PixelFormat::Unknown:
        ;
    }
}

void Images::set_pixel(Images::UID image_ID, RGBA color, unsigned int index, unsigned int mipmap_level) {
    assert(index < Images::get_pixel_count(image_ID, mipmap_level));

    Image image = image_ID;
    unsigned int width = image.get_width();
    while (mipmap_level)
        index += image.get_width(--mipmap_level);
    set_linear_pixel(image_ID, color, index);
    m_changes.add_change(image_ID, Change::PixelsUpdated);
}

void Images::set_pixel(Images::UID image_ID, RGBA color, Vector2ui index, unsigned int mipmap_level) {
    assert(index.x < Images::get_width(image_ID, mipmap_level));
    assert(index.y < Images::get_height(image_ID, mipmap_level));

    Image image = image_ID;
    unsigned int pixel_index = index.x + image.get_width(mipmap_level) * index.y;
    while (mipmap_level) {
        --mipmap_level;
        pixel_index += image.get_width(mipmap_level) * image.get_height(mipmap_level);
    }
    set_linear_pixel(image_ID, color, pixel_index);
    m_changes.add_change(image_ID, Change::PixelsUpdated);
}

void Images::set_pixel(Images::UID image_ID, RGBA color, Vector3ui index, unsigned int mipmap_level) {
    assert(index.x < Images::get_width(image_ID, mipmap_level));
    assert(index.y < Images::get_height(image_ID, mipmap_level));
    assert(index.z < Images::get_depth(image_ID, mipmap_level));

    Image image = image_ID;
    unsigned int pixel_index = index.x + image.get_width(mipmap_level) * (index.y + image.get_height(mipmap_level) * index.z);
    while (mipmap_level) {
        --mipmap_level;
        pixel_index += image.get_width(mipmap_level) * image.get_height(mipmap_level) * image.get_depth(mipmap_level);
    }
    set_linear_pixel(image_ID, color, pixel_index);
    m_changes.add_change(image_ID, Change::PixelsUpdated);
}


//*****************************************************************************
// Image Utilities
//*****************************************************************************

namespace ImageUtils {

Images::UID change_format(Images::UID image_ID, PixelFormat new_format, float new_gamma) {
    Image image = image_ID;
    unsigned int mipmap_count = image.get_mipmap_count();
    Vector3ui size = Vector3ui(image.get_width(), image.get_height(), image.get_depth());
    Images::UID new_image_ID = Images::create3D(image.get_name(), new_format, new_gamma, size, mipmap_count);

    for (unsigned int m = 0; m < mipmap_count; ++m)
        for (unsigned int z = 0; z < image.get_depth(m); ++z)
            for (unsigned int y = 0; y < image.get_height(m); ++y)
                #pragma omp parallel for schedule(dynamic, 16)
                for (int x = 0; x < int(image.get_width(m)); ++x) {
                    Vector3ui index = Vector3ui(x, y, z);
                    RGBA pixel = image.get_pixel(index, m);
                    Images::set_pixel(new_image_ID, pixel, index, m);
                }

    Images::set_mipmapable(new_image_ID, image.is_mipmapable());
    return new_image_ID;
}

void fill_mipmap_chain(Images::UID image_ID) {
    // assert that depth is 1, since 3D textures are not supported.

    // Future work: Optimize for the most used data formats.
    Image image = image_ID;
    for (unsigned int m = 0; m < image.get_mipmap_count() - 1; ++m) {
        for (unsigned int y = 0; y + 1 < image.get_height(m); y += 2) { // TODO Doesn't work with 1D textures does it?
            for (unsigned int x = 0; x + 1 < image.get_width(m); x += 2) {

                RGBA lower_left = image.get_pixel(Vector2ui(x, y), m);
                RGBA lower_right = image.get_pixel(Vector2ui(x + 1, y), m);
                RGBA upper_left = image.get_pixel(Vector2ui(x, y + 1), m);
                RGBA upper_right = image.get_pixel(Vector2ui(x + 1, y + 1), m);

                RGB new_rgb = (lower_left.rgb() + lower_right.rgb() + upper_left.rgb() + upper_right.rgb()) * 0.25f;
                float new_alpha = (lower_left.a + lower_right.a + upper_left.a + upper_right.a) * 0.25f;
                image.set_pixel(RGBA(new_rgb, new_alpha), Vector2ui(x / 2, y / 2), m + 1);
            }

            // If uneven number of columns, then add the last column into the last column of the next mipmap level.
            if (image.get_width(m) & 0x1) {
                RGBA left = image.get_pixel(Vector2ui(image.get_width(m + 1) - 1, y / 2), m + 1);
                RGBA lower_right = image.get_pixel(Vector2ui(image.get_width(m) - 1, y), m);
                RGBA upper_right = image.get_pixel(Vector2ui(image.get_width(m) - 1, y + 1), m);
                RGB rgb = (left.rgb() * 4.0f + lower_right.rgb() + upper_right.rgb()) / 6.0f;
                float alpha = (left.a * 4.0f + lower_right.a + upper_right.a) / 6.0f;
                image.set_pixel(RGBA(rgb, alpha), Vector2ui((image.get_width(m + 1) - 1), y / 2), m + 1);
            }
        }

        // If uneven number of rows, then add the last row into the last row of the next mipmap level.
        if (image.get_height(m) & 0x1) {
            bool uneven_column_count = image.get_width(m) & 0x1;
            unsigned int regular_columns = image.get_width(m) - (uneven_column_count ? 3u : 0u);
            for (unsigned int x = 0; x < regular_columns; x += 2) {
                RGBA lower = image.get_pixel(Vector2ui(x / 2, image.get_height(m + 1) - 1), m + 1);
                RGBA upper_left = image.get_pixel(Vector2ui(x, image.get_height(m) - 1), m);
                RGBA upper_right = image.get_pixel(Vector2ui(x + 1, image.get_height(m) - 1), m);
                RGB rgb = (lower.rgb() * 4.0f + upper_left.rgb() + upper_right.rgb()) / 6.0f;
                float alpha = (lower.a * 4.0f + upper_left.a + upper_right.a) / 6.0f;
                image.set_pixel(RGBA(rgb, alpha), Vector2ui(x / 2, (image.get_height(m + 1) - 1)), m + 1);
            }

            // If both the row and column count are uneven, then we still need to blend 3 edge pixels into the next mipmap pixel
            if (uneven_column_count) {
                RGBA lower = image.get_pixel(Vector2ui(image.get_width(m + 1) - 1, image.get_height(m + 1) - 1), m + 1);
                RGBA upper_left = image.get_pixel(Vector2ui(image.get_width(m) - 3, image.get_height(m) - 1), m);
                RGBA upper_middle = image.get_pixel(Vector2ui(image.get_width(m) - 2, image.get_height(m) - 1), m);
                RGBA upper_right = image.get_pixel(Vector2ui(image.get_width(m) - 1, image.get_height(m) - 1), m);
                RGB rgb = (lower.rgb() * 6.0f + upper_left.rgb() + upper_middle.rgb() + upper_right.rgb()) / 9.0f;
                float alpha = (lower.a * 6.0f + upper_left.a + upper_middle.a + upper_right.a) / 9.0f;
                image.set_pixel(RGBA(rgb, alpha), Vector2ui(image.get_width(m + 1) - 1, image.get_height(m + 1) - 1), m + 1);
            }
        }
    }
}

void compute_summed_area_table(Images::UID image_ID, RGBA* sat_result) {
    Image img = image_ID;
    unsigned int width = img.get_width(), height = img.get_height();

    // Initialize high precision buffer.
    // TODO Instead of preallocating a huge Vector4d array, two rows (current and previous) would do.
    Vector4d* sat = new Vector4d[width * height];

    auto RGBA_to_vector4d = [](RGBA rgba) -> Vector4d { return Vector4d(rgba.r, rgba.g, rgba.b, rgba.a); };

    // Fill the lower row and left column.
    sat[0] = RGBA_to_vector4d(img.get_pixel(Vector2ui(0, 0)));
    for (unsigned int x = 1; x < width; ++x)
        sat[x] = RGBA_to_vector4d(img.get_pixel(Vector2ui(x, 0))) + sat[x-1];
    for (unsigned int y = 1; y < height; ++y)
        sat[y * width] = RGBA_to_vector4d(img.get_pixel(Vector2ui(0, y))) + sat[(y - 1)  * width];

    for (unsigned int y = 1; y < height; ++y)
        for (unsigned int x = 1; x < width; ++x) {
            Vector4d pixel = RGBA_to_vector4d(img.get_pixel(Vector2ui(x, y)));
            sat[x + y * width] = pixel + sat[x + (y - 1)  * width] + sat[(x - 1) + y  * width] - sat[(x - 1) + (y - 1)  * width];
        }

    for (unsigned int y = 0; y < height; ++y)
        for (unsigned int x = 0; x < width; ++x) {
            Vector4d v = sat[x + y * width];
            sat_result[x + y * width] = { float(v.x), float(v.y), float(v.z), float(v.w) };
        }

    delete[] sat;
}

} // NS ImageUtils

} // NS Assets
} // NS Cogwheel