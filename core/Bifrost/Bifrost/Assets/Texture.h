// Bifrost texture.
//---------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
//---------------------------------------------------------------------------------------------

#ifndef _BIFROST_ASSETS_TEXTURE_H_
#define _BIFROST_ASSETS_TEXTURE_H_

#include <Bifrost/Assets/Image.h>
#include <Bifrost/Core/Bitmask.h>
#include <Bifrost/Core/ChangeSet.h>
#include <Bifrost/Core/Iterable.h>
#include <Bifrost/Core/UniqueIDGenerator.h>

namespace Bifrost {
namespace Assets {

enum class WrapMode {
    Clamp,
    Repeat
};

enum class MagnificationFilter {
    None,
    Linear
};

enum class MinificationFilter {
    None,
    Linear,
    Trilinear
};

//----------------------------------------------------------------------------
// Texture ID
//----------------------------------------------------------------------------
class Textures;
typedef Core::TypedUIDGenerator<Textures> TextureIDGenerator;
typedef TextureIDGenerator::UID TextureID;

//-------------------------------------------------------------------------------------------------
// Bifrost texture container.
// Future work:
// * Cubemap support.
//-------------------------------------------------------------------------------------------------
class Textures final {
public:

    enum class Type {
        OneD,
        TwoD,
        ThreeD
    };

    using Iterator = TextureIDGenerator::ConstIterator;

    static bool is_allocated() { return m_samplers != nullptr; }
    static void allocate(unsigned int capacity);
    static void deallocate();

    static inline unsigned int capacity() { return m_UID_generator.capacity(); }
    static void reserve(unsigned int new_capacity);
    static bool has(TextureID texture_ID) { return m_UID_generator.has(texture_ID); }

    static TextureID create2D(ImageID, MagnificationFilter magnification_filter = MagnificationFilter::Linear, MinificationFilter minification_filter = MinificationFilter::Linear, WrapMode wrapmode_U = WrapMode::Repeat, WrapMode wrapmode_V = WrapMode::Repeat);
    static void destroy(TextureID texture_ID);

    static inline Iterator begin() { return m_UID_generator.begin(); }
    static inline Iterator end() { return m_UID_generator.end(); }
    static inline Core::Iterable<Iterator> get_iterable() { return Core::Iterable<Iterator>(begin(), end()); }

    static inline ImageID get_image_ID(TextureID texture_ID) { return m_samplers[texture_ID].image_ID; }
    static inline Type get_type(TextureID texture_ID) { return m_samplers[texture_ID].type; }
    static inline MagnificationFilter get_magnification_filter(TextureID texture_ID) { return m_samplers[texture_ID].magnification_filter; }
    static inline MinificationFilter get_minification_filter(TextureID texture_ID) { return m_samplers[texture_ID].minification_filter; }
    static inline WrapMode get_wrapmode_U(TextureID texture_ID) { return m_samplers[texture_ID].wrapmode_U; }
    static inline WrapMode get_wrapmode_V(TextureID texture_ID) { return m_samplers[texture_ID].wrapmode_V; }
    static inline WrapMode get_wrapmode_W(TextureID texture_ID) { return m_samplers[texture_ID].wrapmode_W; }

    //---------------------------------------------------------------------------------------------
    // Changes since last game loop tick.
    //---------------------------------------------------------------------------------------------
    enum class Change : unsigned char {
        None      = 0u,
        Created   = 1u << 0u,
        Destroyed = 1u << 1u,
        All       = Created | Destroyed
    }; 
    typedef Core::Bitmask<Change> Changes;
    
    static inline Changes get_changes(TextureID texture_ID) { return m_changes.get_changes(texture_ID); }

    typedef std::vector<TextureID>::iterator ChangedIterator;
    static Core::Iterable<ChangedIterator> get_changed_textures() { return m_changes.get_changed_resources(); }

    static void reset_change_notifications() { m_changes.reset_change_notifications(); }

private:
    static void reserve_image_data(unsigned int new_capacity, unsigned int old_capacity);

    // NOTE All of this but the ID can be stored in a single int.
    struct Sampler {
        ImageID image_ID;
        Type type;
        MagnificationFilter magnification_filter;
        MinificationFilter minification_filter;
        WrapMode wrapmode_U;
        WrapMode wrapmode_V;
        WrapMode wrapmode_W;
    };

    static TextureIDGenerator m_UID_generator;
    static Sampler* m_samplers;
    static Core::ChangeSet<Changes, TextureID> m_changes;
};

//-------------------------------------------------------------------------------------------------
// Texture ID wrapper.
//-------------------------------------------------------------------------------------------------
class Texture {
public:
    //---------------------------------------------------------------------------------------------
    // Constructors and destructors.
    //---------------------------------------------------------------------------------------------
    Texture() : m_ID(TextureID::invalid_UID()) {}
    Texture(TextureID id) : m_ID(id) {}

    inline const TextureID get_ID() const { return m_ID; }
    inline bool exists() const { return Textures::has(m_ID); }

    inline bool operator==(Texture rhs) const { return m_ID == rhs.m_ID; }
    inline bool operator!=(Texture rhs) const { return m_ID != rhs.m_ID; }

    //---------------------------------------------------------------------------------------------
    // Getters and setters.
    //---------------------------------------------------------------------------------------------
    inline Image get_image() { return Image(Textures::get_image_ID(m_ID)); }
    inline const Image get_image() const { return Image(Textures::get_image_ID(m_ID)); }
    inline Textures::Type get_type() const { return Textures::get_type(m_ID); }
    inline MagnificationFilter get_magnification_filter() const { return Textures::get_magnification_filter(m_ID); }
    inline MinificationFilter get_minification_filter() const { return Textures::get_minification_filter(m_ID); }
    inline WrapMode get_wrapmode_U() const { return Textures::get_wrapmode_U(m_ID); }
    inline WrapMode get_wrapmode_V() const { return Textures::get_wrapmode_V(m_ID); }
    inline WrapMode get_wrapmode_W() const { return Textures::get_wrapmode_W(m_ID); }

    //---------------------------------------------------------------------------------------------
    // Changes since last game loop tick.
    //---------------------------------------------------------------------------------------------
    inline Textures::Changes get_changes() const { return Textures::get_changes(m_ID); }

private:
    TextureID m_ID;
};

//-------------------------------------------------------------------------------------------------
// Texture sampling
//-------------------------------------------------------------------------------------------------
Math::RGBA sample2D(TextureID texture_ID, Math::Vector2f texcoord, int mipmap_level = 0);

} // NS Assets
} // NS Bifrost

#endif // _BIFROST_ASSETS_TEXTURE_H_
