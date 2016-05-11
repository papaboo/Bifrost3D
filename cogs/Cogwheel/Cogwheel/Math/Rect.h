// Cogwheel rectangle.
// ----------------------------------------------------------------------------
// Copyright (C) 2015, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ----------------------------------------------------------------------------

#ifndef _COGWHEEL_MATH_RECT_H_
#define _COGWHEEL_MATH_RECT_H_

#include <cstring>
#include <sstream>

namespace Cogwheel {
namespace Math {

//----------------------------------------------------------------------------
// Implementation of a templated rectangle.
//----------------------------------------------------------------------------
template <typename T>
struct Rect final {
public:
    typedef T value_type;

    //*************************************************************************
    // Public members
    //*************************************************************************
    T x;
    T y;
    T width;
    T height;

    Rect() { }
    Rect(T x, T y, T width, T height)
        : x(x), y(y), width(width), height(height) { }

    inline Vector2<T> get_offset() const { return Vector2<T>(x, y); }
    inline Vector2<T> get_size() const { return Vector2<T>(width, height); }
    inline Vector2<T> get_min() const { return Vector2<T>(x, y); }
    inline Vector2<T> get_max() const { return Vector2<T>(x+width, y+height); }

    //*************************************************************************
    // Comparison operators.
    //*************************************************************************
    inline bool operator==(Rect<T> rhs) const {
        return memcmp(this, &rhs, sizeof(rhs)) == 0;
    }
    inline bool operator!=(Rect<T> rhs) const {
        return memcmp(this, &rhs, sizeof(rhs)) != 0;
    }

    inline std::string to_string() const {
        std::ostringstream out;
        out << "[x: " << x << ", y: " << y << ", width: " << width << ", height: " << height << "]";
        return out.str();
    }
};

//*****************************************************************************
// Typedefs.
//*****************************************************************************

typedef Rect<double> Rectd;
typedef Rect<float> Rectf;
typedef Rect<int> Recti;

} // NS Math
} // NS Cogwheel

// Convenience function that appends a rectangle's string representation to an ostream.
template<class T>
inline std::ostream& operator<<(std::ostream& s, Cogwheel::Math::Rect<T> v) {
    return s << v.to_string();
}

#endif // _COGWHEEL_MATH_RECT_H_