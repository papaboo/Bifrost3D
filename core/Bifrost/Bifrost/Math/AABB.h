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

#ifndef GPU_COMPILATION
#include <sstream>
#endif

namespace Bifrost::Math {

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
    GPU_ENABLED AABB(Vector3f minimum, Vector3f maximum)
        : minimum(minimum), maximum(maximum) {
    }

    static __always_inline__ GPU_ENABLED AABB invalid() {
        return AABB(Vector3f(std::numeric_limits<float>::infinity()), Vector3f(-std::numeric_limits<float>::infinity()));
    }

    //*************************************************************************
    // Comparison operators.
    //*************************************************************************
    __always_inline__ GPU_ENABLED bool operator==(AABB rhs) const {
        return minimum == rhs.minimum && maximum == rhs.maximum;
    }
    __always_inline__ GPU_ENABLED bool operator!=(AABB rhs) const {
        return minimum != rhs.minimum || maximum != rhs.maximum;
    }

    __always_inline__ GPU_ENABLED void grow_to_contain(Vector3f point) {
        minimum = min(minimum, point);
        maximum = max(maximum, point);
    }

    __always_inline__ GPU_ENABLED void grow_to_contain(AABB aabb) {
        minimum = min(minimum, aabb.minimum);
        maximum = max(maximum, aabb.maximum);
    }

    __always_inline__ GPU_ENABLED Vector3f center() const {
        return (maximum + minimum) * 0.5f;
    }

    __always_inline__ GPU_ENABLED Vector3f size() const {
        return maximum - minimum;
    }

    __always_inline__ GPU_ENABLED Vector3f closest_point_on_surface(Vector3f point) const {
        return max(minimum, min(maximum, point));
    }

#ifndef GPU_COMPILATION
    inline std::string to_string() const {
        std::ostringstream out;
        out << "[minimum: " << minimum << ", maximum: " << maximum << "]";
        return out.str();
    }
#endif
};

} // NS Bifrost::Math

// Convenience function that appends an AABB's string representation to an ostream.
#ifndef GPU_COMPILATION
__always_inline__ std::ostream& operator<<(std::ostream& s, Bifrost::Math::AABB v) {
    return s << v.to_string();
}
#endif

#endif // _BIFROST_MATH_AABB_H_
