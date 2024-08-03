// OptiX primitive intersections.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _OPTIXRENDERER_INTERSECT_H_
#define _OPTIXRENDERER_INTERSECT_H_

#include <OptiXRenderer/Types.h>

#include <optixu/optixu_math_namespace.h>

namespace OptiXRenderer {
namespace Intersect {

// Intersection of ray and sphere.
// Returns the distance to the sphere or negative if no hit.
// Source: Ray Tracing Gems 1, chapter 7, Precision Improvements for Ray / Sphere Intersection 
// and https://www.shadertoy.com/view/WdXfR2. The second precision improvement from Ray Tracing Gems 1 isn't included.
__inline_all__ float ray_sphere(optix::float3 ray_origin, optix::float3 ray_direction, optix::float3 sphere_center, float sphere_radius) {
    optix::float3 direction_to_sphere = ray_origin - sphere_center;
    float b = optix::dot(direction_to_sphere, ray_direction);
    float radius_squared = sphere_radius * sphere_radius;
    optix::float3 fbd = direction_to_sphere - b * ray_direction;
    float d = radius_squared - optix::dot(fbd, fbd);
    if (d > 0.0)
        return -b - sqrt(d);
    else
        return nanf("");
}

__inline_all__ float ray_sphere(const optix::Ray& ray, Sphere sphere) {
    return ray_sphere(ray.origin, ray.direction, sphere.center, sphere.radius);
}

__inline_all__ float ray_plane(optix::float3 ray_origin, optix::float3 ray_direction, optix::float3 plane_point, optix::float3 plane_normal) {
    float d = optix::dot(plane_normal, plane_point);
    float n_dot_o = optix::dot(plane_normal, ray_origin);
    float n_dot_d = optix::dot(plane_normal, ray_direction);
    return (d - n_dot_o) / n_dot_d;
}

// https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-plane-and-ray-disk-intersection
__inline_all__ float ray_disk(optix::float3 ray_origin, optix::float3 ray_direction, optix::float3 disk_center, optix::float3 disk_normal, float disk_radius) {
    float distance_to_plane = ray_plane(ray_origin, ray_direction, disk_center, disk_normal);

    optix::float3 plane_intersection = ray_origin + ray_direction * distance_to_plane;
    optix::float3 v = plane_intersection - disk_center;
    float distance_squared = optix::dot(v, v);
    if (distance_squared <= disk_radius * disk_radius && distance_to_plane >= 0.0f)
        return distance_to_plane;
    else
        return nanf("");
}

__inline_all__ float ray_disk(const optix::Ray& ray, Disk disk) {
    return ray_disk(ray.origin, ray.direction, disk.center, disk.normal, disk.radius);
}

__inline_all__ float point_distance_to_plane(optix::float3 point, optix::float3 plane_point, optix::float3 plane_normal) {
    float d = optix::dot(plane_normal, plane_point);
    float n_dot_o = optix::dot(plane_normal, point);
    return d - n_dot_o;
}

} // NS Intersect
} // NS OptiXRenderer

#endif // _OPTIXRENDERER_INTERSECT_H_