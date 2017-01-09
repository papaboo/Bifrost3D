// Cogwheel bitmask using enum class.
// ---------------------------------------------------------------------------
// Copyright (C) 2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _COGWHEEL_CORE_BITMASK_H_
#define _COGWHEEL_CORE_BITMASK_H_

#include <type_traits>

namespace Cogwheel {
namespace Core {

// ---------------------------------------------------------------------------
// Typesafe bitmask templated with an enumeration.
// ---------------------------------------------------------------------------
template <typename E>
struct Bitmask final {
public:
    typedef typename std::underlying_type<E>::type T;
    
private:
    T m_mask;

public:

    // -----------------------------------------------------------------------
    // Constructors.
    // -----------------------------------------------------------------------
    Bitmask() { }
    Bitmask(E v) : m_mask(T(v)) { }

    // -----------------------------------------------------------------------
    // Modifiers.
    // -----------------------------------------------------------------------
    Bitmask<E>& operator=(E v) { m_mask = T(v); return *this; }
    Bitmask<E>& operator&=(E v) { m_mask &= T(v); return *this; }
    Bitmask<E>& operator|=(E v) { m_mask |= T(v); return *this; }

    // -----------------------------------------------------------------------
    // Accessors.
    // -----------------------------------------------------------------------
    bool none_set() const { return m_mask == T(0); }
    bool not_set(E v) const { return ((T)v & m_mask) == T(0); }
    bool some_set(E v) const { return ((T)v & m_mask) != T(0); }
    bool is_set(E v) const { return ((T)v & m_mask) == T(v); }

    // -----------------------------------------------------------------------
    // Comparisons.
    // -----------------------------------------------------------------------
    bool operator==(E v) const { return T(v) == m_mask; }
    bool operator==(Bitmask v) const { return v.m_mask == m_mask; }
    bool operator!=(E v) const { return T(v) != m_mask; }
    bool operator!=(Bitmask v) const { return v.m_mask != m_mask; }

    // -----------------------------------------------------------------------
    // Raw data access.
    // -----------------------------------------------------------------------
    T raw() { return m_mask; }
};

} // NS Core
} // NS Cogwheel

#endif // _COGWHEEL_CORE_ARRAY_H_