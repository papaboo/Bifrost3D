// Bifrost triangle.
// ----------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ----------------------------------------------------------------------------

#ifndef _BIFROST_MATH_TRIANGLE_H_
#define _BIFROST_MATH_TRIANGLE_H_

#include <Bifrost/Core/Defines.h>
#include <Bifrost/Math/Vector.h>

#ifndef GPU_COMPILATION
#include <sstream>
#endif

namespace Bifrost::Math {

//----------------------------------------------------------------------------
// Implementation of a triangle.
//----------------------------------------------------------------------------
template <typename T>
struct Triangle final {
public:
    //*************************************************************************
    // Public members
    //*************************************************************************
    Vector3<T> v0, v1, v2;

    Triangle() = default;
    GPU_ENABLED Triangle(Vector3<T> v0, Vector3<T> v1, Vector3<T> v2)
        : v0(v0), v1(v1), v2(v2) {}
    template <typename U>
    GPU_ENABLED explicit Triangle(Triangle<U> other)
        : v0(Vector3<T>(other.v0)), v1(Vector3<T>(other.v1)), v2(Vector3<T>(other.v2)) {}

    // Unnormalized up direction of the triangle.
    _inline_all_archs_ Vector3<T> get_up() const { return get_up(v0, v1, v2); }
    _inline_all_archs_ Vector3<T> get_normal() const { return normalize(get_up()); }
    _inline_all_archs_ T get_surface_area() const { return magnitude(cross(v1 - v0, v2 - v0)) / 2; }
    _inline_all_archs_ Vector3<T> get_min() const { return min(v0, min(v1, v2)); }
    _inline_all_archs_ Vector3<T> get_max() const { return max(v0, max(v1, v2)); }

    _inline_all_archs_ static Vector3<T> get_up(Vector3<T> v0, Vector3<T> v1, Vector3<T> v2) { return cross(v1 - v0, v2 - v0); }
    _inline_all_archs_ static Vector3<T> get_normal(Vector3<T> v0, Vector3<T> v1, Vector3<T> v2) { return normalize(get_up(v0, v1, v2)); }

    //*************************************************************************
    // Comparison operators.
    //*************************************************************************
    _inline_all_archs_ bool operator==(Triangle<T> rhs) const {
        return v0 == rhs.v0 && v1 == rhs.v1 && v2 == rhs.v2;
    }
    _inline_all_archs_ bool operator!=(Triangle<T> rhs) const {
        return v0 != rhs.v0 || v1 != rhs.v1 || v2 != rhs.v2;
    }

#ifndef GPU_COMPILATION
    inline std::string to_string() const {
        std::ostringstream out;
        out << "[v0: " << v0 << ", v1: " << v1 << ", v2: " << v2 << "]";
        return out.str();
    }
#endif
};

//*****************************************************************************
// Typedefs.
//*****************************************************************************

typedef Triangle<float> Trianglef;
typedef Triangle<double> Triangled;

} // NS Bifrost::Math

// Convenience function that appends a triangle's string representation to an ostream.
#ifndef GPU_COMPILATION
template<class T>
__always_inline__ std::ostream& operator<<(std::ostream& s, Bifrost::Math::Triangle<T> v) {
    return s << v.to_string();
}
#endif

#endif // _BIFROST_MATH_TRIANGLE_H_
