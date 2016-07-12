// Cogwheel texture.
// ---------------------------------------------------------------------------
// Copyright (C) 2015-2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _COGWHEEL_ASSETS_TEXTURE_H_
#define _COGWHEEL_ASSETS_TEXTURE_H_

#include <Cogwheel/Assets/Image.h>
#include <Cogwheel/Core/Iterable.h>
#include <Cogwheel/Core/UniqueIDGenerator.h>

#include <vector>

namespace Cogwheel {
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
// Cogwheel texture container.
// Future work:
// * Cubemap support.
//----------------------------------------------------------------------------
class Textures final {
public:

    enum class Type {
        OneD,
        TwoD,
        ThreeD
    };

    typedef Core::TypedUIDGenerator<Textures> UIDGenerator;
    typedef UIDGenerator::UID UID;
    typedef UIDGenerator::ConstIterator ConstUIDIterator;

    static bool is_allocated() { return m_samplers != nullptr; }
    static void allocate(unsigned int capacity);
    static void deallocate();

    static inline unsigned int capacity() { return m_UID_generator.capacity(); }
    static void reserve(unsigned int new_capacity);
    static bool has(Textures::UID texture_ID);

    static Textures::UID create2D(Images::UID, MagnificationFilter magnification_filter = MagnificationFilter::Linear, MinificationFilter minification_filter = MinificationFilter::Linear, WrapMode wrapmode_U = WrapMode::Repeat, WrapMode wrapmode_V = WrapMode::Repeat);
    static void destroy(Textures::UID texture_ID);

    static inline ConstUIDIterator begin() { return m_UID_generator.begin(); }
    static inline ConstUIDIterator end() { return m_UID_generator.end(); }
    static inline Core::Iterable<ConstUIDIterator> get_iterable() { return Core::Iterable<ConstUIDIterator>(begin(), end()); }

    static inline Images::UID get_image_ID(Textures::UID texture_ID) { return m_samplers[texture_ID].image_ID; }
    static inline Type get_type(Textures::UID texture_ID) { return m_samplers[texture_ID].type; }
    static inline MagnificationFilter get_magnification_filter(Textures::UID texture_ID) { return m_samplers[texture_ID].magnification_filter; }
    static inline MinificationFilter get_minification_filter(Textures::UID texture_ID) { return m_samplers[texture_ID].minification_filter; }
    static inline WrapMode get_wrapmode_U(Textures::UID texture_ID) { return m_samplers[texture_ID].wrapmode_U; }
    static inline WrapMode get_wrapmode_V(Textures::UID texture_ID) { return m_samplers[texture_ID].wrapmode_V; }
    static inline WrapMode get_wrapmode_W(Textures::UID texture_ID) { return m_samplers[texture_ID].wrapmode_W; }

    //-------------------------------------------------------------------------
    // Changes since last game loop tick.
    //-------------------------------------------------------------------------
    struct Changes {
        static const unsigned char None = 0u;
        static const unsigned char Created = 1u << 0u;
        static const unsigned char Destroyed = 1u << 1u;
        static const unsigned char All = Created | Destroyed;
    }; 
    
    static inline unsigned char get_changes(Textures::UID texture_ID) { return m_changes[texture_ID]; }
    static inline bool has_changes(Textures::UID texture_ID, unsigned char change_bitmask = Changes::All) {
        return (m_changes[texture_ID] & change_bitmask) != Changes::None;
    }

    typedef std::vector<UID>::iterator ChangedIterator;
    static Core::Iterable<ChangedIterator> get_changed_textures() {
        return Core::Iterable<ChangedIterator>(m_textures_changed.begin(), m_textures_changed.end());
    }

    static void reset_change_notifications();

private:
    static void reserve_image_data(unsigned int new_capacity, unsigned int old_capacity);

    // NOTE All of this but the ID can be stored in a single int.
    struct Sampler {
        Images::UID image_ID;
        Type type;
        MagnificationFilter magnification_filter;
        MinificationFilter minification_filter;
        WrapMode wrapmode_U;
        WrapMode wrapmode_V;
        WrapMode wrapmode_W;
    };

    static UIDGenerator m_UID_generator;
    static Sampler* m_samplers;
    static unsigned char* m_changes; // Bitmask of changes.
    static std::vector<UID> m_textures_changed;
};

class TextureND {
public:
    // -----------------------------------------------------------------------
    // Constructors and destructors.
    // -----------------------------------------------------------------------
    TextureND() : m_ID(Textures::UID::invalid_UID()) {}
    TextureND(Textures::UID id) : m_ID(id) {}

    inline const Textures::UID get_ID() const { return m_ID; }
    inline bool exists() const { return Textures::has(m_ID); }

    inline bool operator==(TextureND rhs) const { return m_ID == rhs.m_ID; }
    inline bool operator!=(TextureND rhs) const { return m_ID != rhs.m_ID; }

    // -----------------------------------------------------------------------
    // Getters and setters.
    // -----------------------------------------------------------------------
    inline Image get_image() { return Image(Textures::get_image_ID(m_ID)); }
    inline Textures::Type get_type() { return Textures::get_type(m_ID); }
    inline MagnificationFilter get_magnification_filter() { return Textures::get_magnification_filter(m_ID); }
    inline MinificationFilter get_minification_filter() { return Textures::get_minification_filter(m_ID); }
    inline WrapMode get_wrapmode_U() { return Textures::get_wrapmode_U(m_ID); }
    inline WrapMode get_wrapmode_V() { return Textures::get_wrapmode_V(m_ID); }
    inline WrapMode get_wrapmode_W() { return Textures::get_wrapmode_W(m_ID); }

    //-------------------------------------------------------------------------
    // Changes since last game loop tick.
    //-------------------------------------------------------------------------
    inline unsigned char get_changes() { return Textures::get_changes(m_ID); }
    inline bool has_changes(unsigned char change_bitmask = Textures::Changes::All) { return Textures::has_changes(m_ID, change_bitmask); }

private:
    Textures::UID m_ID;
};

Math::RGBA sample2D(Textures::UID texture_ID, Math::Vector2f texcoord);

} // NS Assets
} // NS Cogwheel

#endif // _COGWHEEL_ASSETS_TEXTURE_H_