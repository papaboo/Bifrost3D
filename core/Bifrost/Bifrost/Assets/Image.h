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

//----------------------------------------------------------------------------
// Pixel formats and helper functions.
//----------------------------------------------------------------------------
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

inline bool has_alpha(PixelFormat format) {
    return format == PixelFormat::Alpha8 || format == PixelFormat::RGBA32 || format == PixelFormat::RGBA_Float;
}

//----------------------------------------------------------------------------
// Image ID
//----------------------------------------------------------------------------
class Images;
typedef Core::TypedUIDGenerator<Images> ImageIDGenerator;
typedef ImageIDGenerator::UID ImageID;

//----------------------------------------------------------------------------
// Bifrost image container.
// Images are indexed from the lower left corner to the top right one.
// E.g. (0, 0) is in the lower left corner.
// Future work:
// * Replace gamma by an is_sRGB bool/flag. We only ever use gamma 2.2 anyway. Then we can also precompute sRGB <-> linear tables for faster encoding and decoding.
// * A for_each that applies a lambda to all pixels. Maybe specialize it 
//   for floats and bytes and profile if that speeds up anything.
// * set_pixels.
// * set_pixels_rect.
// * Cubemap support.
//----------------------------------------------------------------------------
class Images final {
public:
    using Iterator = ImageIDGenerator::ConstIterator;

    typedef void* PixelData;

    static bool is_allocated() { return m_metainfo != nullptr; }
    static void allocate(unsigned int capacity);
    static void deallocate();

    static inline unsigned int capacity() { return m_UID_generator.capacity(); }
    static void reserve(unsigned int new_capacity);
    static bool has(ImageID image_ID);

    static ImageID create(const std::string& name, PixelFormat format, float gamma, Math::Vector3ui size, unsigned int mipmap_count = 1);
    static ImageID create(const std::string& name, PixelFormat format, float gamma, Math::Vector3ui size, PixelData& pixels);

    static void destroy(ImageID image_ID);

    static inline Iterator begin() { return m_UID_generator.begin(); }
    static inline Iterator end() { return m_UID_generator.end(); }
    static inline Core::Iterable<Iterator> get_iterable() { return { begin(), end() }; }

    static inline std::string get_name(ImageID image_ID) { return m_metainfo[image_ID].name; }
    static inline void set_name(ImageID image_ID, const std::string& name) { m_metainfo[image_ID].name = name; }

    static inline PixelFormat get_pixel_format(ImageID image_ID) { return m_metainfo[image_ID].pixel_format; }
    static inline float get_gamma(ImageID image_ID) { return m_metainfo[image_ID].gamma; }
    static void set_gamma(ImageID image_ID, float gamma) { m_metainfo[image_ID].gamma = gamma; }
    static inline unsigned int get_mipmap_count(ImageID image_ID) { return m_metainfo[image_ID].mipmap_count; }
    static inline unsigned int get_width(ImageID image_ID, unsigned int mipmap_level = 0) { return Math::max(1u, m_metainfo[image_ID].width >> mipmap_level); }
    static inline unsigned int get_height(ImageID image_ID, unsigned int mipmap_level = 0) { return Math::max(1u, m_metainfo[image_ID].height >> mipmap_level); }
    static inline unsigned int get_depth(ImageID image_ID, unsigned int mipmap_level = 0) { return Math::max(1u, m_metainfo[image_ID].depth >> mipmap_level); }
    static inline Math::Vector2ui get_size_2D(ImageID image_ID, unsigned int mipmap_level = 0) { return Math::Vector2ui(get_width(image_ID, mipmap_level), get_height(image_ID, mipmap_level)); }
    static inline Math::Vector3ui get_size_3D(ImageID image_ID, unsigned int mipmap_level = 0) { return Math::Vector3ui(get_size_2D(image_ID, mipmap_level), get_depth(image_ID, mipmap_level)); }
    static inline unsigned int get_pixel_count(ImageID image_ID, unsigned int mipmap_level = 0) {
        return get_width(image_ID, mipmap_level) * get_height(image_ID, mipmap_level) * get_depth(image_ID, mipmap_level);
    }
    static inline unsigned int get_total_pixel_count(ImageID image_ID) { return m_metainfo[image_ID].total_pixel_count; }

    // Returns true if mipmaps can be auto generated for the texture.
    static inline bool is_mipmapable(ImageID image_ID) { return m_metainfo[image_ID].is_mipmapable; }
    static void set_mipmapable(ImageID image_ID, bool value);

    static PixelData get_pixels(ImageID image_ID, int mipmap_level = 0);
    template <typename T>
    static T* get_pixels(ImageID image_ID, int mipmap_level = 0) {
        assert(sizeof(T) == size_of(get_pixel_format(image_ID)));
        return (T*)get_pixels(image_ID, mipmap_level);
    }

    static Math::RGBA get_pixel(ImageID image_ID, unsigned int index, unsigned int mipmap_level = 0);
    static Math::RGBA get_pixel(ImageID image_ID, Math::Vector2ui index, unsigned int mipmap_level = 0);
    static Math::RGBA get_pixel(ImageID image_ID, Math::Vector3ui index, unsigned int mipmap_level = 0);
    static void set_pixel(ImageID image_ID, Math::RGBA rgba, unsigned int index, unsigned int mipmap_level = 0);
    static void set_pixel(ImageID image_ID, Math::RGBA rgba, Math::Vector2ui index, unsigned int mipmap_level = 0);
    static void set_pixel(ImageID image_ID, Math::RGBA rgba, Math::Vector3ui index, unsigned int mipmap_level = 0);

    template <typename Operation>
    static void iterate_pixels(ImageID image_ID, Operation pixel_operation) {
        int pixel_count = get_pixel_count(image_ID);
        for (int i = 0; i < pixel_count; ++i) {
            RGBA pixel = get_pixel(image_ID, i);
            pixel_operation(pixel);
        }
    }

    static void change_format(ImageID image_ID, PixelFormat new_format, float new_gamma);

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

    static inline Changes get_changes(ImageID image_ID) { return m_changes.get_changes(image_ID); }

    typedef std::vector<ImageID>::iterator ChangedIterator;
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
        unsigned int total_pixel_count;
        PixelFormat pixel_format;
        float gamma;
        bool is_mipmapable;
    };

    static ImageIDGenerator m_UID_generator;
    static MetaInfo* m_metainfo;
    static PixelData* m_pixels;
    static Core::ChangeSet<Changes, ImageID> m_changes;
};

// ---------------------------------------------------------------------------
// ImageID wrapper convinience class.
// ---------------------------------------------------------------------------
class Image final {
public:
    // -----------------------------------------------------------------------
    // Constructors and destructors.
    // -----------------------------------------------------------------------
    Image() : m_ID(ImageID::invalid_UID()) {}
    Image(ImageID id) : m_ID(id) {}

    static Image create3D(const std::string& name, PixelFormat format, float gamma, Math::Vector3ui size, unsigned int mipmap_count = 1) {
        return Images::create(name, format, gamma, size, mipmap_count);
    }
    static Image create2D(const std::string& name, PixelFormat format, float gamma, Math::Vector2ui size, unsigned int mipmap_count = 1) {
        return Images::create(name, format, gamma, Math::Vector3ui(size, 1u), mipmap_count);
    }
    static Image create1D(const std::string& name, PixelFormat format, float gamma, unsigned int width, unsigned int mipmap_count = 1) {
        return Images::create(name, format, gamma, Math::Vector3ui(width, 1u, 1u), mipmap_count);
    }

    static ImageID create2D(const std::string& name, PixelFormat format, float gamma, Math::Vector2ui size, Images::PixelData& pixels) {
        return Images::create(name, format, gamma, Math::Vector3ui(size, 1u), pixels);
    }

    static Image invalid() { return ImageID::invalid_UID(); }

    inline void destroy() { Images::destroy(m_ID); }
    inline bool exists() const { return Images::has(m_ID); }
    inline ImageID get_ID() const { return m_ID; }

    inline bool operator==(Image rhs) const { return m_ID == rhs.m_ID; }
    inline bool operator!=(Image rhs) const { return m_ID != rhs.m_ID; }

    // -----------------------------------------------------------------------
    // Getters and setters.
    // -----------------------------------------------------------------------
    inline std::string get_name() const { return Images::get_name(m_ID); }
    inline void set_name(const std::string& name) { Images::set_name(m_ID, name); }

    inline PixelFormat get_pixel_format() const { return Images::get_pixel_format(m_ID); }
    inline float get_gamma() const { return Images::get_gamma(m_ID); }
    inline bool is_mipmapable() const { return Images::is_mipmapable(m_ID); }
    inline void set_mipmapable(bool value) { Images::set_mipmapable(m_ID, value); }
    inline unsigned int get_mipmap_count() const { return Images::get_mipmap_count(m_ID); }
    inline unsigned int get_width(unsigned int mipmap_level = 0) const { return Images::get_width(m_ID, mipmap_level); }
    inline unsigned int get_height(unsigned int mipmap_level = 0) const { return Images::get_height(m_ID, mipmap_level); }
    inline unsigned int get_depth(unsigned int mipmap_level = 0) const { return Images::get_depth(m_ID, mipmap_level); }
    inline Math::Vector2ui get_size_2D(unsigned int mipmap_level = 0) const { return Images::get_size_2D(m_ID, mipmap_level); }
    inline Math::Vector3ui get_size_3D(unsigned int mipmap_level = 0) const { return Images::get_size_3D(m_ID, mipmap_level); }
    inline unsigned int get_pixel_count(unsigned int mipmap_level = 0) const { return Images::get_pixel_count(m_ID, mipmap_level); }
    inline unsigned int get_total_pixel_count() const { return Images::get_total_pixel_count(m_ID); }

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

    inline void change_format(PixelFormat new_format, float new_gamma) { Images::change_format(m_ID, new_format, new_gamma); }

    void clear();
    void clear(Math::RGBA clear_color);

    //-------------------------------------------------------------------------
    // Changes since last game loop tick.
    //-------------------------------------------------------------------------
    inline Images::Changes get_changes() const { return Images::get_changes(m_ID); }

private:
    ImageID m_ID;
};

namespace ImageUtils {

template <typename T>
inline Image copy_with_new_format(Image image, PixelFormat new_format, float new_gamma, T process_pixel) {
    unsigned int mipmap_count = image.get_mipmap_count();
    Math::Vector3ui size = image.get_size_3D();
    Image new_image = Image::create3D(image.get_name(), new_format, new_gamma, size, mipmap_count);

    for (unsigned int m = 0; m < mipmap_count; ++m)
        #pragma omp parallel for schedule(dynamic, 16)
        for (int p = 0; p < int(image.get_pixel_count(m)); ++p) {
            auto pixel = image.get_pixel(p, m);
            new_image.set_pixel(process_pixel(pixel), p, m);
        }

    new_image.set_mipmapable(image.is_mipmapable());
    return new_image;
}

inline Image copy_with_new_format(Image image, PixelFormat new_format, float new_gamma) {
    return copy_with_new_format(image, new_format, new_gamma, [](Math::RGBA c) -> Math::RGBA { return c; } );
}

inline Image copy_with_new_format(Image image, PixelFormat new_format) {
    return copy_with_new_format(image, new_format, image.get_gamma());
}

void fill_mipmap_chain(Image image);

void compute_summed_area_table(Image image, Math::RGBA* sat_result);

inline Math::RGBA* compute_summed_area_table(Image image) {
    Math::RGBA* sat = new Math::RGBA[image.get_width() * image.get_height()];
    compute_summed_area_table(image, sat);
    return sat;
}

Image combine_tint_roughness(const Image tint, const Image roughness, int roughness_channel = 3);

} // NS ImageUtils

} // NS Assets
} // NS Bifrost

#endif // _BIFROST_ASSETS_IMAGE_H_
