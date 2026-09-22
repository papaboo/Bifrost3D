// Bifrost disk.
// ----------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ----------------------------------------------------------------------------

#ifndef _BIFROST_MATH_DISK_H_
#define _BIFROST_MATH_DISK_H_

#include <Bifrost/Math/Constants.h>
#include <Bifrost/Math/Vector.h>

#ifndef GPU_COMPILATION
#include <sstream>
#endif

namespace Bifrost::Math {

//----------------------------------------------------------------------------
// Implementation of a disk.
//----------------------------------------------------------------------------
struct Disk final {
public:
    //*************************************************************************
    // Public members
    //*************************************************************************
    Vector3f center;
    float radius;
    Vector3f normal;

    Disk() = default;
    GPU_ENABLED Disk(Vector3f center, float radius, Vector3f normal)
        : center(center), radius(radius), normal(normal) {}

    _inline_all_archs_ float get_surface_area() const { return PI<float>() * radius * radius; }

    _inline_all_archs_ static float get_surface_area(Vector3f center, float radius) { return PI<float>() * radius * radius; }

    //*************************************************************************
    // Comparison operators.
    //*************************************************************************
    _inline_all_archs_ bool operator==(Disk rhs) const {
        return center == rhs.center && radius == rhs.radius && normal == rhs.normal;
    }
    _inline_all_archs_ bool operator!=(Disk rhs) const {
        return center != rhs.center || radius != rhs.radius || normal != rhs.normal;
    }

#ifndef GPU_COMPILATION
    inline std::string to_string() const {
        std::ostringstream out;
        out << "[center: " << center << ", radius: " << radius << ", normal: " << normal << "]";
        return out.str();
    }
#endif
};

} // NS Bifrost::Math

// Convenience function that appends a triangle's string representation to an ostream.
#ifndef GPU_COMPILATION
template<class T>
__always_inline__ std::ostream& operator<<(std::ostream& s, Bifrost::Math::Disk v) {
    return s << v.to_string();
}
#endif

#endif // _BIFROST_MATH_TRIANGLE_H_
