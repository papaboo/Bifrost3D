// Cogwheel image.
// ---------------------------------------------------------------------------
// Copyright (C) 2015-2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _COGWHEEL_ASSETS_IMAGE_H_
#define _COGWHEEL_ASSETS_IMAGE_H_

#include <Cogwheel/Core/Bitmask.h>
#include <Cogwheel/Core/ChangeSet.h>
#include <Cogwheel/Core/UniqueIDGenerator.h>
#include <Cogwheel/Math/Color.h>
#include <Cogwheel/Math/Utils.h>
#include <Cogwheel/Math/Vector.h>

#include <string>

namespace Cogwheel {
namespace Assets {

enum class PixelFormat {
    Unknown = 0,
    I8,
    RGB24,
    RGBA32,
    RGB_Float,
    RGBA_Float,
};

inline int size_of(PixelFormat format) {
    switch (format) {
    case PixelFormat::RGBA32: return 4;
    case PixelFormat::RGBA_Float: return 16;
    case PixelFormat::RGB24: return 3;
    case PixelFormat::RGB_Float: return 12;
    case PixelFormat::I8: return 1;
    case PixelFormat::Unknown:
    default:
        return 0;
    }
}

inline int channel_count(PixelFormat format) {
    switch (format) {
    case PixelFormat::RGBA32:
    case PixelFormat::RGBA_Float:
        return 4;
    case PixelFormat::RGB24:
    case PixelFormat::RGB_Float:
        return 3;
    case PixelFormat::I8:
        return 1;
    case PixelFormat::Unknown:
    default:
        return 0;
    }
}

//----------------------------------------------------------------------------
// Cogwheel image container.
// Images are indexed from the lower left corner to the top right one.
// E.g. (0, 0) is in the lower left corner.
// Future work:
// * A for_each that applies a lambda to all pixels. Maybe specialize it 
//   for floats and bytes and profile if that speeds up anything.
// * set_pixels.
// * set_pixels_rect.
// * Cubemap support.
//----------------------------------------------------------------------------
class Images final {
public:

    typedef Core::TypedUIDGenerator<Images> UIDGenerator;
    typedef UIDGenerator::UID UID;
    typedef UIDGenerator::ConstIterator ConstUIDIterator;

    typedef void* PixelData;

    static bool is_allocated() { return m_metainfo != nullptr; }
    static void allocate(unsigned int capacity);
    static void deallocate();

    static inline unsigned int capacity() { return m_UID_generator.capacity(); }
    static void reserve(unsigned int new_capacity);
    static bool has(Images::UID image_ID);

    static Images::UID create3D(const std::string& name, PixelFormat format, float gamma, Math::Vector3ui size, unsigned int mipmap_count = 1);
    static Images::UID create2D(const std::string& name, PixelFormat format, float gamma, Math::Vector2ui size, unsigned int mipmap_count = 1) {
        return create3D(name, format, gamma, Math::Vector3ui(size.x, size.y, 1u), mipmap_count);
    }
    static Images::UID create1D(const std::string& name, PixelFormat format, float gamma, unsigned int width, unsigned int mipmap_count = 1) {
        return create3D(name, format, gamma, Math::Vector3ui(width, 1u, 1u), mipmap_count);
    }

    static void destroy(Images::UID image_ID);

    static inline ConstUIDIterator begin() { return m_UID_generator.begin(); }
    static inline ConstUIDIterator end() { return m_UID_generator.end(); }
    static inline Core::Iterable<ConstUIDIterator> get_iterable() { return Core::Iterable<ConstUIDIterator>(begin(), end()); }

    static inline std::string get_name(Images::UID image_ID) { return m_metainfo[image_ID].name; }
    static inline void set_name(Images::UID image_ID, const std::string& name) { m_metainfo[image_ID].name = name; }

    static inline PixelFormat get_pixel_format(Images::UID image_ID) { return m_metainfo[image_ID].pixel_format; }
    static inline float get_gamma(Images::UID image_ID) { return m_metainfo[image_ID].gamma; }
    static void set_gamma(Images::UID image_ID, float gamma) { m_metainfo[image_ID].gamma = gamma; }
    static inline unsigned int get_mipmap_count(Images::UID image_ID) { return m_metainfo[image_ID].mipmap_count; }
    static inline unsigned int get_width(Images::UID image_ID, unsigned int mipmap_level = 0) { return Math::max(1u, m_metainfo[image_ID].width >> mipmap_level); }
    static inline unsigned int get_height(Images::UID image_ID, unsigned int mipmap_level = 0) { return Math::max(1u, m_metainfo[image_ID].height >> mipmap_level); }
    static inline unsigned int get_depth(Images::UID image_ID, unsigned int mipmap_level = 0) { return Math::max(1u, m_metainfo[image_ID].depth >> mipmap_level); }
    static inline unsigned int get_pixel_count(Images::UID image_ID, unsigned int mipmap_level = 0) {
        return get_width(image_ID, mipmap_level) * get_height(image_ID, mipmap_level) * get_depth(image_ID, mipmap_level);
    }

    // Returns true if mipmaps can be auto generated for the texture.
    static inline bool is_mipmapable(Images::UID image_ID) { return m_metainfo[image_ID].is_mipmapable; }
    static void set_mipmapable(Images::UID image_ID, bool value);

    static PixelData get_pixels(Images::UID image_ID, int mipmap_level = 0);
    template <typename T>
    static T* get_pixels(Images::UID image_ID, int mipmap_level = 0) {
        assert(sizeof(T) == size_of(get_pixel_format(image_ID)));
        return (T*)get_pixels(image_ID, mipmap_level);
    }

    static Math::RGBA get_pixel(Images::UID image_ID, unsigned int index, unsigned int mipmap_level = 0);
    static Math::RGBA get_pixel(Images::UID image_ID, Math::Vector2ui index, unsigned int mipmap_level = 0);
    static Math::RGBA get_pixel(Images::UID image_ID, Math::Vector3ui index, unsigned int mipmap_level = 0);
    static void set_pixel(Images::UID image_ID, Math::RGBA rgba, unsigned int index, unsigned int mipmap_level = 0);
    static void set_pixel(Images::UID image_ID, Math::RGBA rgba, Math::Vector2ui index, unsigned int mipmap_level = 0);
    static void set_pixel(Images::UID image_ID, Math::RGBA rgba, Math::Vector3ui index, unsigned int mipmap_level = 0);

    //-------------------------------------------------------------------------
    // Changes since last game loop tick.
    //-------------------------------------------------------------------------
    enum class Change : unsigned char {
        None = 0,
        Created = 1,
        Destroyed = 2,
        PixelsUpdated = 4,
        Mipmapable = 8,
        All = Created | Destroyed | PixelsUpdated | Mipmapable
    }; 
    typedef Core::Bitmask<Change> Changes;

    static inline Changes get_changes(Images::UID image_ID) { return m_changes.get_changes(image_ID); }

    typedef std::vector<UID>::iterator ChangedIterator;
    static Core::Iterable<ChangedIterator> get_changed_images() { return m_changes.get_changed_resources(); }

    static void reset_change_notifications() { m_changes.reset_change_notifications(); }

private:
    static void reserve_image_data(unsigned int new_capacity, unsigned int old_capacity);

    struct MetaInfo {
        std::string name;
        unsigned int width;
        unsigned int height;
        unsigned int depth;
        unsigned int mipmap_count;
        PixelFormat pixel_format;
        float gamma;
        bool is_mipmapable;
    };

    static UIDGenerator m_UID_generator;
    static MetaInfo* m_metainfo;
    static PixelData* m_pixels;
    static Core::ChangeSet<Changes, UID> m_changes;
};

// ---------------------------------------------------------------------------
// Images::UID wrapper convinience class.
// ---------------------------------------------------------------------------
class Image final {
public:
    // -----------------------------------------------------------------------
    // Constructors and destructors.
    // -----------------------------------------------------------------------
    Image() : m_ID(Images::UID::invalid_UID()) {}
    Image(Images::UID id) : m_ID(id) {}

    inline const Images::UID get_ID() const { return m_ID; }
    inline bool exists() const { return Images::has(m_ID); }

    inline bool operator==(Image rhs) const { return m_ID == rhs.m_ID; }
    inline bool operator!=(Image rhs) const { return m_ID != rhs.m_ID; }

    // -----------------------------------------------------------------------
    // Getters and setters.
    // -----------------------------------------------------------------------
    inline std::string get_name() { return Images::get_name(m_ID); }
    inline void set_name(const std::string& name) { Images::set_name(m_ID, name); }

    inline PixelFormat get_pixel_format() { return Images::get_pixel_format(m_ID); }
    inline float get_gamma() { return Images::get_gamma(m_ID); }
    inline bool is_mipmapable() { return Images::is_mipmapable(m_ID); }
    inline void set_mipmapable(bool value) { Images::set_mipmapable(m_ID, value); }
    inline unsigned int get_mipmap_count() { return Images::get_mipmap_count(m_ID); }
    inline unsigned int get_width(unsigned int mipmap_level = 0) { return Images::get_width(m_ID, mipmap_level); }
    inline unsigned int get_height(unsigned int mipmap_level = 0) { return Images::get_height(m_ID, mipmap_level); }
    inline unsigned int get_depth(unsigned int mipmap_level = 0) { return Images::get_depth(m_ID, mipmap_level); }
    inline unsigned int get_pixel_count(unsigned int mipmap_level = 0) { return Images::get_pixel_count(m_ID, mipmap_level); }

    inline Images::PixelData get_pixels(unsigned int mipmap_level = 0) { return Images::get_pixels(m_ID, mipmap_level); }
    template <typename T>
    inline T* get_pixels(int mipmap_level = 0) { return Images::get_pixels<T>(m_ID, mipmap_level); }

    inline Math::RGBA get_pixel(unsigned int index, unsigned int mipmap_level = 0) { return Images::get_pixel(m_ID, index, mipmap_level); }
    inline Math::RGBA get_pixel(Math::Vector2ui index, unsigned int mipmap_level = 0) { return Images::get_pixel(m_ID, index, mipmap_level); }
    inline Math::RGBA get_pixel(Math::Vector3ui index, unsigned int mipmap_level = 0) { return Images::get_pixel(m_ID, index, mipmap_level); }
    inline void set_pixel(Math::RGBA rgba, unsigned int index, unsigned int mipmap_level = 0) { Images::set_pixel(m_ID, rgba, index, mipmap_level); }
    inline void set_pixel(Math::RGBA rgba, Math::Vector2ui index, unsigned int mipmap_level = 0) { Images::set_pixel(m_ID, rgba, index, mipmap_level); }
    inline void set_pixel(Math::RGBA rgba, Math::Vector3ui index, unsigned int mipmap_level = 0) { Images::set_pixel(m_ID, rgba, index, mipmap_level); }

    //-------------------------------------------------------------------------
    // Changes since last game loop tick.
    //-------------------------------------------------------------------------
    inline Images::Changes get_changes() { return Images::get_changes(m_ID); }

private:
    Images::UID m_ID;
};

namespace ImageUtils {

Images::UID change_format(Images::UID image_ID, PixelFormat new_format, float new_gamma);

inline Images::UID change_format(Images::UID image_ID, PixelFormat new_format) {
    return change_format(image_ID, new_format, Images::get_gamma(image_ID));
}

void fill_mipmap_chain(Images::UID image_ID);

void compute_summed_area_table(Images::UID image_ID, Math::RGBA* sat_result);

inline Math::RGBA* compute_summed_area_table(Images::UID image_ID) {
    Math::RGBA* sat = new Math::RGBA[Images::get_width(image_ID) * Images::get_height(image_ID)];
    compute_summed_area_table(image_ID, sat);
    return sat;
}

} // NS ImageUtils

} // NS Assets
} // NS Cogwheel

#endif // _COGWHEEL_ASSETS_IMAGE_H_