// Mathematical primitive intersections.
// ------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _BIFROST_MATH_INTERSECT_H_
#define _BIFROST_MATH_INTERSECT_H_

#include <Bifrost/Core/Defines.h>
#include <Bifrost/Math/Plane.h>
#include <Bifrost/Math/Ray.h>
#include <Bifrost/Math/Triangle.h>

namespace Bifrost::Math::Intersect {

constexpr float no_hit_value = INFINITY;

_inline_all_archs_ bool valid_hit(float distance) { return abs(distance) != INFINITY; } // NAN safe check
_inline_all_archs_ bool no_hit(float distance) { return !valid_hit(distance); }

// https://www.siggraph.org/education/materials/HyperGraph/raytrace/rayplane_intersection.htm
_inline_all_archs_ float ray_plane(Ray ray, Plane plane) {
    float distance = -(dot(plane.get_normal(), ray.origin) + plane.d) / dot(plane.get_normal(), ray.direction);
    bool hit_behind_ray = distance < 0.0f;
    return hit_behind_ray ? no_hit_value : distance;
}

struct TriangleHit {
    float distance;
    Vector3f barycentric_coords;

    _inline_all_archs_ static TriangleHit none() {
        TriangleHit hit = {};
        hit.distance = no_hit_value;
        return hit;
    }

    _inline_all_archs_ bool hit() const { return valid_hit(distance); }
};

// Mueller-Trumbore ray/triangle intersection.
// Returns the barycentric coordinates of the intersection. NaN in case of no intersection.
_inline_all_archs_ TriangleHit ray_triangle(Ray ray, Vector3f positions[3], bool two_sided = true) {

    Vector3f edge1 = positions[1] - positions[0];
    Vector3f edge2 = positions[2] - positions[0];
    Vector3f ray_cross_edge2 = cross(ray.direction, edge2);
    float determinant = dot(edge1, ray_cross_edge2);

    if (two_sided ? determinant == 0 : determinant <= 0)
        // Output degenerate barycentric coords as the ray is parallel to the triangle plane.
        return TriangleHit::none();

    float inv_determinant = 1.0 / determinant;
    Vector3f s = ray.origin - positions[0];
    float u = inv_determinant * dot(s, ray_cross_edge2);

    Vector3f s_cross_edge1 = cross(s, edge1);
    float v = inv_determinant * dot(ray.direction, s_cross_edge1);

    float distance = inv_determinant * dot(edge2, s_cross_edge1);

    Vector3f barycentric_coords = Vector3f(1 - u - v, u, v);

    // Ray intersected the triangle if the barycentric coordinates are valid and the triangle is in front of the ray.
    bool valid_u = u >= 0 && u <= 1;
    bool valid_v = v >= 0 && u + v <= 1;
    bool valid_distance = distance >= 0.0;
    bool intersected = valid_u && valid_v && valid_distance;
    return { intersected ? distance : no_hit_value, barycentric_coords };
}

_inline_all_archs_ TriangleHit ray_triangle(Ray ray, Trianglef triangle, bool two_sided = true) {
    return ray_triangle(ray, &triangle.v0, two_sided);
}

} // NS Bifrost::Math::intersect

#endif // _BIFROST_MATH_PLANE_H_
