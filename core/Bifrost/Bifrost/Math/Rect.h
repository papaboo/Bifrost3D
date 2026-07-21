// Bifrost rectangle.
// ----------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ----------------------------------------------------------------------------

#ifndef _BIFROST_MATH_RECT_H_
#define _BIFROST_MATH_RECT_H_

#include <Bifrost/Core/Defines.h>
#include <Bifrost/Math/Vector.h>

#ifndef GPU_COMPILATION
#include <sstream>
#endif

namespace Bifrost::Math {

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

    Rect() = default;
    GPU_ENABLED Rect(T x, T y, T width, T height)
        : x(x), y(y), width(width), height(height) { }
    template <typename U>
    GPU_ENABLED explicit Rect(Rect<U> other)
        : x(T(other.x)), y(T(other.y)), width(T(other.width)), height(T(other.height)) { }

    __always_inline__ GPU_ENABLED Vector2<T> get_offset() const { return Vector2<T>(x, y); }
    __always_inline__ GPU_ENABLED Vector2<T> get_size() const { return Vector2<T>(width, height); }
    __always_inline__ GPU_ENABLED Vector2<T> get_min() const { return Vector2<T>(x, y); }
    __always_inline__ GPU_ENABLED Vector2<T> get_max() const { return Vector2<T>(x+width, y+height); }

    //*************************************************************************
    // Comparison operators.
    //*************************************************************************
    __always_inline__ GPU_ENABLED bool operator==(Rect<T> rhs) const {
        return x == rhs.x && y == rhs.y && width == rhs.width && height == rhs.height;
    }
    __always_inline__ GPU_ENABLED bool operator!=(Rect<T> rhs) const {
        return x != rhs.x || y != rhs.y || width != rhs.width || height != rhs.height;
    }

#ifndef GPU_COMPILATION
    inline std::string to_string() const {
        std::ostringstream out;
        out << "[x: " << x << ", y: " << y << ", width: " << width << ", height: " << height << "]";
        return out.str();
    }
#endif
};

//*****************************************************************************
// Typedefs.
//*****************************************************************************

typedef Rect<double> Rectd;
typedef Rect<float> Rectf;
typedef Rect<int> Recti;

} // NS Bifrost::Math

// Convenience function that appends a rectangle's string representation to an ostream.
#ifndef GPU_COMPILATION
template<class T>
__always_inline__ std::ostream& operator<<(std::ostream& s, Bifrost::Math::Rect<T> v) {
    return s << v.to_string();
}
#endif

#endif // _BIFROST_MATH_RECT_H_
