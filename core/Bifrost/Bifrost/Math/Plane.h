// Bifrost plane.
// ------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _BIFROST_MATH_PLANE_H_
#define _BIFROST_MATH_PLANE_H_

#include <Bifrost/Core/Defines.h>
#include <Bifrost/Math/Vector.h>

#ifndef GPU_COMPILATION
#include <sstream>
#endif

namespace Bifrost::Math {

// ------------------------------------------------------------------------------------------------
// Plane implementation.
// The normal of the plane stored in [a, b, c] is assumed to be normalized.
// ------------------------------------------------------------------------------------------------
struct Plane final {
public:
    // --------------------------------------------------------------------------------------------
    // Public members
    // --------------------------------------------------------------------------------------------
    float a, b, c, d;

    // --------------------------------------------------------------------------------------------
    // Constructors.
    // --------------------------------------------------------------------------------------------
    Plane() = default;
    GPU_ENABLED Plane(float a, float b, float c, float d) : a(a), b(b), c(c), d(d) { }

    __always_inline__ GPU_ENABLED static Plane from_point_normal(Vector3f point, Vector3f normal) {
        float d = -dot(point, normal);
        return Plane(normal.x, normal.y, normal.z, d);
    }

    __always_inline__ GPU_ENABLED static Plane from_point_direction(Vector3f point, Vector3f direction) {
        return from_point_normal(point, normalize(direction));
    }

    // --------------------------------------------------------------------------------------------
    // Comparison operators.
    // --------------------------------------------------------------------------------------------
    __always_inline__ GPU_ENABLED bool operator==(Plane rhs) const {
        return a == rhs.a && b == rhs.b && c == rhs.c && d == rhs.d;
    }
    __always_inline__ GPU_ENABLED bool operator!=(Plane rhs) const {
        return a != rhs.a || b != rhs.b || c != rhs.c || d != rhs.d;
    }

    // --------------------------------------------------------------------------------------------
    // Getters.
    // --------------------------------------------------------------------------------------------
    __always_inline__ GPU_ENABLED Vector3f get_normal() const { return { a, b, c }; }

    // --------------------------------------------------------------------------------------------
    // To string.
    // --------------------------------------------------------------------------------------------
#ifndef GPU_COMPILATION
    inline std::string to_string() const {
        std::ostringstream out;
        out << "[a: " << a << ", b: " << b << ", c: " << c << ", d: " << d << "]";
        return out.str();
    }
#endif
};

} // NS Bifrost::Math

// Convenience function that appends an AABB's string representation to an ostream.
#ifndef GPU_COMPILATION
__always_inline__ std::ostream& operator<<(std::ostream& s, Bifrost::Math::Plane v) {
    return s << v.to_string();
}
#endif

#endif // _BIFROST_MATH_PLANE_H_
