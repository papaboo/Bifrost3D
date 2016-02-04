// Cogwheel axis-aliged bounding box.
// ----------------------------------------------------------------------------
// Copyright (C) 2015, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ----------------------------------------------------------------------------

#ifndef _COGWHEEL_MATH_AABB_H_
#define _COGWHEEL_MATH_AABB_H_

#include <Cogwheel/Math/Vector.h>

#include <cstring>
#include <sstream>

namespace Cogwheel {
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

    AABB() {}
    AABB(Vector3f minimum, Vector3f maximum)
        : minimum(minimum), maximum(maximum) {
    }

    //*************************************************************************
    // Comparison operators.
    //*************************************************************************
    inline bool operator==(AABB rhs) const {
        return memcmp(this, &rhs, sizeof(rhs)) == 0;
    }
    inline bool operator!=(AABB rhs) const {
        return memcmp(this, &rhs, sizeof(rhs)) != 0;
    }

    inline void include_point(Vector3f point) {
        minimum = min(minimum, point);
        maximum = max(maximum, point);
    }

    inline Vector3f size() const {
        return maximum - minimum;
    }

    std::string to_string() const {
        std::ostringstream out;
        out << "[minimum: " << minimum << ", maximum: " << maximum << "]";
        return out.str();
    }
};

} // NS Math
} // NS Cogwheel

// Convenience function that appends an AABB's string representation to an ostream.
inline std::ostream& operator<<(std::ostream& s, Cogwheel::Math::AABB v) {
    return s << v.to_string();
}

#endif // _COGWHEEL_MATH_AABB_H_