// Bifrost bitmask using enum class.
// ------------------------------------------------------------------------------------------------
// Copyright (C) 2016, Bifrost. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. 
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _BIFROST_CORE_BITMASK_H_
#define _BIFROST_CORE_BITMASK_H_

#include <type_traits>

namespace Bifrost {
namespace Core {

// ------------------------------------------------------------------------------------------------
// Typesafe bitmask templated with an enumeration.
// ------------------------------------------------------------------------------------------------
template <typename E>
struct Bitmask final {
public:
    typedef typename std::underlying_type<E>::type T;
    typedef E Flags;

private:
    T m_mask;

public:

    // --------------------------------------------------------------------------------------------
    // Constructors.
    // --------------------------------------------------------------------------------------------
    Bitmask() = default;
    Bitmask(E v) : m_mask(T(v)) { }
    Bitmask(T mask) : m_mask(mask) { }
    Bitmask(std::initializer_list<E> flags) : m_mask(0) {  for (E flag : flags) m_mask |= T(flag); }

    // --------------------------------------------------------------------------------------------
    // Modifiers.
    // --------------------------------------------------------------------------------------------
    inline Bitmask<E>& operator=(E v) { m_mask = T(v); return *this; }
    inline Bitmask<E>& operator&=(E v) { m_mask &= T(v); return *this; }
    inline Bitmask<E>& operator&=(Bitmask v) { m_mask &= v.m_mask; return *this; }
    inline Bitmask<E>& operator|=(E v) { m_mask |= T(v); return *this; }
    inline Bitmask<E>& operator|=(Bitmask v) { m_mask |= v.m_mask; return *this; }
    inline Bitmask<E>& operator^=(E v) { m_mask ^= T(v); return *this; }
    inline Bitmask<E>& operator^=(Bitmask v) { m_mask ^= v.m_mask; return *this; }

    // --------------------------------------------------------------------------------------------
    // Bit operations.
    // --------------------------------------------------------------------------------------------
    inline Bitmask<E> operator&(E v) const { return Bitmask(m_mask & T(v)); }
    inline Bitmask<E> operator&(Bitmask v) const { return Bitmask(m_mask & v.m_mask); }
    inline Bitmask<E> operator|(E v) const { return Bitmask(m_mask | T(v)); }
    inline Bitmask<E> operator|(Bitmask v) const { return Bitmask(m_mask | v.m_mask); }
    inline Bitmask<E> operator^(E v) const { return Bitmask(m_mask ^ T(v)); }
    inline Bitmask<E> operator^(Bitmask v) const { return Bitmask(m_mask ^ v.m_mask); }

    // --------------------------------------------------------------------------------------------
    // Accessors.
    // --------------------------------------------------------------------------------------------
    inline bool none_set() const { return m_mask == T(0); }
    inline bool not_set(E v) const { return (T(v) & m_mask) == T(0); }
    inline bool all_set(E v1, E v2) const { T v = (T(v1) | T(v2);  return v & m_mask) == v; }
    inline bool is_set(E v) const { return (T(v) & m_mask) != T(0); }
    inline bool any_set(E v) const { return (T(v) & m_mask) != T(0); }
    inline bool any_set(E v1, E v2) const { return ((T(v1) | T(v2)) & m_mask) != T(0); }

    // --------------------------------------------------------------------------------------------
    // Comparisons.
    // --------------------------------------------------------------------------------------------
    inline bool operator==(E v) const { return T(v) == m_mask; }
    inline bool operator==(Bitmask v) const { return v.m_mask == m_mask; }
    inline bool operator!=(E v) const { return T(v) != m_mask; }
    inline bool operator!=(Bitmask v) const { return v.m_mask != m_mask; }
    inline operator bool() const { return m_mask != T(0); }

    // --------------------------------------------------------------------------------------------
    // Raw data access.
    // --------------------------------------------------------------------------------------------
    inline T raw() { return m_mask; }
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
} // NS Bifrost

#endif // _BIFROST_CORE_BITMASK_H_
