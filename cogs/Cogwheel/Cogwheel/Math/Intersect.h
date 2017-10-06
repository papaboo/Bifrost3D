// Mathematical primitive intersections.
// ------------------------------------------------------------------------------------------------
// Copyright (C) 2017, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _COGWHEEL_MATH_INTERSECT_H_
#define _COGWHEEL_MATH_INTERSECT_H_

#include <Cogwheel/Math/Plane.h>
#include <Cogwheel/Math/Ray.h>

namespace Cogwheel {
namespace Math {

// https://www.siggraph.org/education/materials/HyperGraph/raytrace/rayplane_intersection.htm
inline float intersect(Ray ray, Plane plane) {
    return -(dot(plane.get_normal(), ray.origin) + plane.d) / dot(plane.get_normal(), ray.direction);
}

} // NS Math
} // NS Cogwheel

#endif // _COGWHEEL_MATH_PLANE_H_