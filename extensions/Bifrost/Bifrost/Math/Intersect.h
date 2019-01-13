// Mathematical primitive intersections.
// ------------------------------------------------------------------------------------------------
// Copyright (C) 2017, Bifrost. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _BIFROST_MATH_INTERSECT_H_
#define _BIFROST_MATH_INTERSECT_H_

#include <Bifrost/Math/Plane.h>
#include <Bifrost/Math/Ray.h>

namespace Bifrost {
namespace Math {

// https://www.siggraph.org/education/materials/HyperGraph/raytrace/rayplane_intersection.htm
inline float intersect(Ray ray, Plane plane) {
    return -(dot(plane.get_normal(), ray.origin) + plane.d) / dot(plane.get_normal(), ray.direction);
}

} // NS Math
} // NS Bifrost

#endif // _BIFROST_MATH_PLANE_H_
