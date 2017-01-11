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
    typedef E Flags;

private:
    T m_mask;

public:

    // -----------------------------------------------------------------------
    // Constructors.
    // -----------------------------------------------------------------------
    Bitmask() { }
    Bitmask(E v) : m_mask(T(v)) { }
    Bitmask(T mask) : m_mask(mask) { }

    // -----------------------------------------------------------------------
    // Modifiers.
    // -----------------------------------------------------------------------
    Bitmask<E>& operator=(E v) { m_mask = T(v); return *this; }
    Bitmask<E>& operator&=(E v) { m_mask &= T(v); return *this; }
    Bitmask<E>& operator|=(E v) { m_mask |= T(v); return *this; }
    Bitmask<E>& operator^=(E v) { m_mask ^= T(v); return *this; }

    // -----------------------------------------------------------------------
    // Bit operations.
    // -----------------------------------------------------------------------
    Bitmask<E> operator&(E v) const { return Bitmask(m_mask & T(v)); }
    Bitmask<E> operator|(E v) const { return Bitmask(m_mask | T(v)); }
    Bitmask<E> operator^(E v) const { return Bitmask(m_mask ^ T(v)); }

    // -----------------------------------------------------------------------
    // Accessors.
    // -----------------------------------------------------------------------
    bool none_set() const { return m_mask == T(0); }
    bool not_set(E v) const { return (T(v) & m_mask) == T(0); }
    bool all_set(E v1, E v2) const { T v = (T(v1) | T(v2);  return v & m_mask) == v; }
    bool is_set(E v) const { return (T(v) & m_mask) != T(0); }
    bool any_set(E v) const { return (T(v) & m_mask) != T(0); }
    bool any_set(E v1, E v2) const { return ((T(v1) | T(v2)) & m_mask) != T(0); }

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

// -----------------------------------------------------------------------
// Bit operations.
// -----------------------------------------------------------------------


// -----------------------------------------------------------------------
// Comparisons.
// -----------------------------------------------------------------------
template <typename E>
inline bool operator==(E lhs, Bitmask<E> rhs) { return rhs == lhs; }
template <typename E>
inline bool operator!=(E lhs, Bitmask<E> rhs) { return rhs != lhs; }

} // NS Core
} // NS Cogwheel

#endif // _COGWHEEL_CORE_BITMASK_H_