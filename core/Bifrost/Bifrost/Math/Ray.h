// Bifrost ray.
// ----------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ----------------------------------------------------------------------------

#ifndef _BIFROST_MATH_RAY_H_
#define _BIFROST_MATH_RAY_H_

#include <Bifrost/Core/Defines.h>
#include <Bifrost/Math/Vector.h>

#include <cstring>
#include <sstream>

namespace Bifrost {
namespace Math {

//----------------------------------------------------------------------------
// Implementation of a 3 dimensional ray.
//----------------------------------------------------------------------------
struct Ray final {
public:
    //*************************************************************************
    // Public members
    //*************************************************************************
    Vector3f origin;
    Vector3f direction;

    Ray() = default;
    Ray(Vector3f origin, Vector3f direction)
        : origin(origin), direction(direction) { }

    //*************************************************************************
    // Comparison operators.
    //*************************************************************************
    __always_inline__ bool operator==(Ray rhs) const {
        return memcmp(this, &rhs, sizeof(rhs)) == 0;
    }
    __always_inline__ bool operator!=(Ray rhs) const {
        return memcmp(this, &rhs, sizeof(rhs)) != 0;
    }

    __always_inline__ Vector3f position_at(float t) const {
        return origin + direction * t;
    }

    inline std::string to_string() const {
        std::ostringstream out;
        out << "[origin: " << origin << ", direction: " << direction << "]";
        return out.str();
    }
};

} // NS Math
} // NS Bifrost

// Convenience function that appends a ray's string representation to an ostream.
__always_inline__ std::ostream& operator<<(std::ostream& s, Bifrost::Math::Ray v) {
    return s << v.to_string();
}

#endif // _BIFROST_MATH_RAY_H_
