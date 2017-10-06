// Cogwheel plane.
// ------------------------------------------------------------------------------------------------
// Copyright (C) 2017, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _COGWHEEL_MATH_PLANE_H_
#define _COGWHEEL_MATH_PLANE_H_

#include <Cogwheel/Math/Vector.h>

#include <cstring>
#include <sstream>

namespace Cogwheel {
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
    Plane() {}
    Plane(float a, float b, float c, float d)
        : a(a), b(b), c(c), d(d) {
    }

    inline static Plane from_point_normal(Vector3f point, Vector3f normal) {
        float d = -dot(point, normal);
        return Plane(normal.x, normal.y, normal.z, d);
    }

    inline static Plane from_point_direction(Vector3f point, Vector3f direction) {
        return from_point_normal(point, normalize(direction));
    }
    
    // --------------------------------------------------------------------------------------------
    // Comparison operators.
    // --------------------------------------------------------------------------------------------
    inline bool operator==(Plane rhs) const {
        return memcmp(this, &rhs, sizeof(rhs)) == 0;
    }
    inline bool operator!=(Plane rhs) const {
        return memcmp(this, &rhs, sizeof(rhs)) != 0;
    }

    // --------------------------------------------------------------------------------------------
    // Getters.
    // --------------------------------------------------------------------------------------------
    inline Vector3f get_normal() const { return Vector3f(a, b, c); }

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
} // NS Cogwheel

// Convenience function that appends an AABB's string representation to an ostream.
inline std::ostream& operator<<(std::ostream& s, Cogwheel::Math::Plane v) {
    return s << v.to_string();
}

#endif // _COGWHEEL_MATH_PLANE_H_