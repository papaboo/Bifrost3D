// Bifrost axis-aliged bounding box.
// ----------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ----------------------------------------------------------------------------

#ifndef _BIFROST_MATH_AABB_H_
#define _BIFROST_MATH_AABB_H_

#include <Bifrost/Core/Defines.h>
#include <Bifrost/Math/Vector.h>

#include <cstring>
#include <sstream>

namespace Bifrost {
namespace Math {

//----------------------------------------------------------------------------
// Implementation of an axis-aligned bounding box.
//----------------------------------------------------------------------------
struct AABB final {
public:
    //*************************************************************************
    // Public members
    //*************************************************************************
    Vector3f minimum;
    Vector3f maximum;

    AABB() = default;
    AABB(Vector3f minimum, Vector3f maximum)
        : minimum(minimum), maximum(maximum) {
    }

    static __always_inline__ AABB invalid() {
        return AABB(Vector3f(std::numeric_limits<float>::infinity()), Vector3f(-std::numeric_limits<float>::infinity()));
    }

    //*************************************************************************
    // Comparison operators.
    //*************************************************************************
    __always_inline__ bool operator==(AABB rhs) const {
        return memcmp(this, &rhs, sizeof(rhs)) == 0;
    }
    __always_inline__ bool operator!=(AABB rhs) const {
        return memcmp(this, &rhs, sizeof(rhs)) != 0;
    }

    __always_inline__ void grow_to_contain(Vector3f point) {
        minimum = min(minimum, point);
        maximum = max(maximum, point);
    }

    __always_inline__ void grow_to_contain(AABB aabb) {
        minimum = min(minimum, aabb.minimum);
        maximum = max(maximum, aabb.maximum);
    }

    __always_inline__ Vector3f center() const {
        return (maximum + minimum) * 0.5f;
    }

    __always_inline__ Vector3f size() const {
        return maximum - minimum;
    }

    inline std::string to_string() const {
        std::ostringstream out;
        out << "[minimum: " << minimum << ", maximum: " << maximum << "]";
        return out.str();
    }
};

} // NS Math
} // NS Bifrost

// Convenience function that appends an AABB's string representation to an ostream.
__always_inline__ std::ostream& operator<<(std::ostream& s, Bifrost::Math::AABB v) {
    return s << v.to_string();
}

#endif // _BIFROST_MATH_AABB_H_
