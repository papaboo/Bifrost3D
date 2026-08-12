// Test Bifrost primitive intersections.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _BIFROST_MATH_INTERSECT_TEST_H_
#define _BIFROST_MATH_INTERSECT_TEST_H_

#include <Bifrost/Math/Intersect.h>

#include <Expects.h>

namespace Bifrost::Math {

GTEST_TEST(Math_Intersect, ray_plane) {
    // Define test plane at origo, with surface normal pointing along +z.
    auto plane = Plane::from_point_normal(Vector3f::zero(), { 0, 0, 1 });

    { // Ray one unit from the plane and pointing directly at it.
        auto ray = Ray({ 0, 0, 1 }, { 0, 0, -1 });
        float distance = Intersect::ray_plane(ray, plane);

        EXPECT_FLOAT_EQ(1.0f, distance);
    }

    { // Ray four units from the plane, point at it at an angle. The direction is chosen such that the distance the ray travels is 5.
        auto ray = Ray({ 0, 0, 4 }, normalize(Vector3f(0, 3, -4)));
        float distance = Intersect::ray_plane(ray, plane);

        EXPECT_FLOAT_EQ(5.0f, distance);
    }

    { // Ray perpendicular to the plane misses.
        auto ray = Ray({ 0, 0, 1 }, { 0, 1, 0 });
        float distance = Intersect::ray_plane(ray, plane);

        EXPECT_TRUE(Intersect::no_hit(distance));
    }

    { // Do not report hits behind the ray
        auto ray = Ray({ 0, 0, 1 }, { 0, 0, 1 });
        float distance = Intersect::ray_plane(ray, plane);

        EXPECT_TRUE(Intersect::no_hit(distance));
    }
}

GTEST_TEST(Math_Intersect, ray_triangle_hits) {
    // Define a large triangle spanning the positive xy-quadrant.
    Vector3f v0 = { 0, 0, 0 };
    Vector3f v1 = { 20, 0, 0 };
    Vector3f v2 = { 0, 20, 0 };
    Vector3f triangle[3] = { v0, v1, v2 };

    { // Ray one unit from the triangle and pointing directly at it.
        auto ray = Ray({ 0, 0, 1 }, { 0, 0, -1 });
        auto hit = Intersect::ray_triangle(ray, triangle);

        EXPECT_TRUE(hit.hit());
        EXPECT_FLOAT_EQ(1.0f, hit.distance);
    }

    { // Ray four units from the plane, point at it at an angle. The direction is chosen such that the distance the ray travels is 5.
        auto ray = Ray({ 1, 1, 4 }, normalize(Vector3f(0, 3, -4)));
        Vector3f ray_pos = ray.position_at(5.0f);
        Vector3f ray_pos_hit = ray.position_at(1.0f);

        auto hit = Intersect::ray_triangle(ray, triangle);

        EXPECT_TRUE(hit.hit());
        EXPECT_FLOAT_EQ(5.0f, hit.distance);
    }
}

GTEST_TEST(Math_Intersect, ray_triangle_miss_behind_ray) {
    // Define a large triangle spanning the positive xy-quadrant.
    Vector3f v0 = { 0, 0, 0 };
    Vector3f v1 = { 20, 0, 0 };
    Vector3f v2 = { 0, 20, 0 };
    Vector3f triangle[3] = { v0, v1, v2 };

    { // Ray one unit from the triangle and pointing away from it should miss, even if the line defined by the ray still inersects the triangle.
        auto ray = Ray({ 0, 0, 1 }, { 0, 0, -1 });
        auto hit = Intersect::ray_triangle(ray, triangle);
        EXPECT_TRUE(hit.hit());

        // Flip ray so it points away from the triangle and the ray could intersect the triangle behind origin.
        ray.direction = -ray.direction;
        hit = Intersect::ray_triangle(ray, triangle);
        EXPECT_FALSE(hit.hit());
    }
}

GTEST_TEST(Math_Intersect, ray_triangle_can_miss_from_behind_triangle) {
    // Define a large triangle spanning the positive xy-quadrant.
    Vector3f v0 = { 0, 0, 0 };
    Vector3f v1 = { 20, 0, 0 };
    Vector3f v2 = { 0, 20, 0 };
    Vector3f triangle[3] = { v0, v1, v2 };

    { // Ray from behind triangle can hit the triangle if two_sided is true, but not if it's false.
        auto ray = Ray({ 0, 0, -1 }, { 0, 0, 1 });
        auto hit = Intersect::ray_triangle(ray, triangle, true);
        EXPECT_TRUE(hit.hit());

        hit = Intersect::ray_triangle(ray, triangle, false);
        EXPECT_FALSE(hit.hit());
    }
}

} // NS Bifrost::Math

#endif // _BIFROST_MATH_INTERSECT_TEST_H_
