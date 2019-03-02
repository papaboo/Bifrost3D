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

#include <cstring>
#include <sstream>

namespace Bifrost {
namespace Math {

// ------------------------------------------------------------------------------------------------
// Plane implementation.
// The normal of the plane stored in [a, b, c] is assumed to be normalized.
// ------------------------------------------------------------------------------------------------
struct Plane final {
public:
    // --------------------------------------------------------------------------------------------
    // Public members
    // --------------------------------------------------------------------------------------------
    float a;
    float b;
    float c;
    float d;

    // --------------------------------------------------------------------------------------------
    // Constructors.
    // --------------------------------------------------------------------------------------------
    Plane() = default;
    Plane(float a, float b, float c, float d)
        : a(a), b(b), c(c), d(d) {
    }

    __always_inline__ static Plane from_point_normal(Vector3f point, Vector3f normal) {
        float d = -dot(point, normal);
        return Plane(normal.x, normal.y, normal.z, d);
    }

    __always_inline__ static Plane from_point_direction(Vector3f point, Vector3f direction) {
        return from_point_normal(point, normalize(direction));
    }

    // --------------------------------------------------------------------------------------------
    // Comparison operators.
    // --------------------------------------------------------------------------------------------
    __always_inline__ bool operator==(Plane rhs) const {
        return memcmp(this, &rhs, sizeof(rhs)) == 0;
    }
    __always_inline__ bool operator!=(Plane rhs) const {
        return memcmp(this, &rhs, sizeof(rhs)) != 0;
    }

    // --------------------------------------------------------------------------------------------
    // Getters.
    // --------------------------------------------------------------------------------------------
    __always_inline__ Vector3f get_normal() const { return Vector3f(a, b, c); }

    // --------------------------------------------------------------------------------------------
    // To string.
    // --------------------------------------------------------------------------------------------
    inline std::string to_string() const {
        std::ostringstream out;
        out << "[a: " << a << ", b: " << b << ", c: " << c << ", d: " << d << "]";
        return out.str();
    }
};

} // NS Math
} // NS Bifrost

// Convenience function that appends an AABB's string representation to an ostream.
__always_inline__ std::ostream& operator<<(std::ostream& s, Bifrost::Math::Plane v) {
    return s << v.to_string();
}

#endif // _BIFROST_MATH_PLANE_H_
