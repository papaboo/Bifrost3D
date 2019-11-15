// Bifrost image.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _BIFROST_ASSETS_IMAGE_H_
#define _BIFROST_ASSETS_IMAGE_H_

#include <Bifrost/Core/Bitmask.h>
#include <Bifrost/Core/ChangeSet.h>
#include <Bifrost/Core/UniqueIDGenerator.h>
#include <Bifrost/Math/Color.h>
#include <Bifrost/Math/Utils.h>
#include <Bifrost/Math/Vector.h>

#include <string>

namespace Bifrost {
namespace Assets {

enum class PixelFormat {
    Unknown = 0,
    Alpha8,
    Metallic8 = Alpha8,
    Roughness8 = Alpha8,
    Intensity8, // Uses the red channel when getting and setting pixels. Alpha is always one when getting a pixel. Green and blue are undefined.
    RGB24,
    RGBA32,
    Intensity_Float, // Uses the red channel when getting and setting pixels. Alpha is always one when getting a pixel. Green and blue are undefined.
    RGB_Float,
    RGBA_Float,
};

inline int size_of(PixelFormat format) {
    switch (format) {
    case PixelFormat::Alpha8:
    case PixelFormat::Intensity8: return 1;
    case PixelFormat::RGB24: return 3;
    case PixelFormat::RGBA32:
    case PixelFormat::Intensity_Float:
        return 4;
    case PixelFormat::RGB_Float: return 12;
    case PixelFormat::RGBA_Float: return 16;
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
    case PixelFormat::Alpha8:
    case PixelFormat::Intensity8:
    case PixelFormat::Intensity_Float:
        return 1;
    case PixelFormat::Unknown:
    default:
        return 0;
    }
}

//----------------------------------------------------------------------------
// Bifrost image container.
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

    static Images::UID create2D(const std::string& name, PixelFormat format, float gamma, Math::Vector2ui size, PixelData& pixels);

    static void destroy(Images::UID image_ID);

    static inline ConstUIDIterator begin() { return m_UID_generator.begin(); }
    static inline ConstUIDIterator end() { return m_UID_generator.end(); }
    static inline Core::Iterable<ConstUIDIterator> get_iterable() { return { begin(), end() }; }

    static inline const std::string& get_name(Images::UID image_ID) { return m_metainfo[image_ID].name; }
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

    template <typename Operation>
    static void iterate_pixels(Images::UID image_ID, Operation pixel_operation) {
        int pixel_count = get_pixel_count(image_ID);
        for (int i = 0; i < pixel_count; ++i) {
            RGBA pixel = get_pixel(image_ID, i);
            pixel_operation(pixel);
        }
    }

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

    inline Images::UID get_ID() { return m_ID; }
    inline const Images::UID get_ID() const { return m_ID; }
    inline bool exists() const { return Images::has(m_ID); }

    inline bool operator==(Image rhs) const { return m_ID == rhs.m_ID; }
    inline bool operator!=(Image rhs) const { return m_ID != rhs.m_ID; }

    // -----------------------------------------------------------------------
    // Getters and setters.
    // -----------------------------------------------------------------------
    inline const std::string& get_name() const { return Images::get_name(m_ID); }
    inline void set_name(const std::string& name) { Images::set_name(m_ID, name); }

    inline PixelFormat get_pixel_format() const { return Images::get_pixel_format(m_ID); }
    inline float get_gamma() const { return Images::get_gamma(m_ID); }
    inline bool is_mipmapable() const { return Images::is_mipmapable(m_ID); }
    inline void set_mipmapable(bool value) { Images::set_mipmapable(m_ID, value); }
    inline unsigned int get_mipmap_count() const { return Images::get_mipmap_count(m_ID); }
    inline unsigned int get_width(unsigned int mipmap_level = 0) const { return Images::get_width(m_ID, mipmap_level); }
    inline unsigned int get_height(unsigned int mipmap_level = 0) const { return Images::get_height(m_ID, mipmap_level); }
    inline unsigned int get_depth(unsigned int mipmap_level = 0) const { return Images::get_depth(m_ID, mipmap_level); }
    inline unsigned int get_pixel_count(unsigned int mipmap_level = 0) const { return Images::get_pixel_count(m_ID, mipmap_level); }

    inline Images::PixelData get_pixels(unsigned int mipmap_level = 0) { return Images::get_pixels(m_ID, mipmap_level); }
    inline const Images::PixelData get_pixels(unsigned int mipmap_level = 0) const { return Images::get_pixels(m_ID, mipmap_level); }
    template <typename T>
    inline T* get_pixels(int mipmap_level = 0) { return Images::get_pixels<T>(m_ID, mipmap_level); }
    template <typename T>
    inline const T* const get_pixels(int mipmap_level = 0) const { return Images::get_pixels<T>(m_ID, mipmap_level); }

    inline Math::RGBA get_pixel(unsigned int index, unsigned int mipmap_level = 0) const { return Images::get_pixel(m_ID, index, mipmap_level); }
    inline Math::RGBA get_pixel(Math::Vector2ui index, unsigned int mipmap_level = 0) const { return Images::get_pixel(m_ID, index, mipmap_level); }
    inline Math::RGBA get_pixel(Math::Vector3ui index, unsigned int mipmap_level = 0) const { return Images::get_pixel(m_ID, index, mipmap_level); }
    inline void set_pixel(Math::RGBA rgba, unsigned int index, unsigned int mipmap_level = 0) { Images::set_pixel(m_ID, rgba, index, mipmap_level); }
    inline void set_pixel(Math::RGBA rgba, Math::Vector2ui index, unsigned int mipmap_level = 0) { Images::set_pixel(m_ID, rgba, index, mipmap_level); }
    inline void set_pixel(Math::RGBA rgba, Math::Vector3ui index, unsigned int mipmap_level = 0) { Images::set_pixel(m_ID, rgba, index, mipmap_level); }

    template <typename Operation>
    inline void iterate_pixels(Operation pixel_operation) { Images::iterate_pixels(m_ID, pixel_operation); }

    //-------------------------------------------------------------------------
    // Changes since last game loop tick.
    //-------------------------------------------------------------------------
    inline Images::Changes get_changes() const { return Images::get_changes(m_ID); }

private:
    Images::UID m_ID;
};

namespace ImageUtils {

template <typename T>
inline Images::UID change_format(Images::UID image_ID, PixelFormat new_format, float new_gamma, T process_pixel) {
    Image image = image_ID;
    unsigned int mipmap_count = image.get_mipmap_count();
    auto size = Math::Vector3ui(image.get_width(), image.get_height(), image.get_depth());
    Images::UID new_image_ID = Images::create3D(image.get_name(), new_format, new_gamma, size, mipmap_count);

    for (unsigned int m = 0; m < mipmap_count; ++m)
        #pragma omp parallel for schedule(dynamic, 16)
        for (int p = 0; p < int(image.get_pixel_count(m)); ++p) {
            auto pixel = image.get_pixel(p, m);
            Images::set_pixel(new_image_ID, process_pixel(pixel), p, m);
        }

    Images::set_mipmapable(new_image_ID, image.is_mipmapable());
    return new_image_ID;
}

inline Images::UID change_format(Images::UID image_ID, PixelFormat new_format, float new_gamma) {
    return change_format(image_ID, new_format, new_gamma, [](Math::RGBA c) -> Math::RGBA { return c; } );
}

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
} // NS Bifrost

#endif // _BIFROST_ASSETS_IMAGE_H_
