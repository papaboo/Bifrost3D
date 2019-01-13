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
__inline_all__ float ray_sphere(const optix::float3& ray_origin, const optix::float3& ray_direction, const optix::float3& sphere_center, float sphere_radius) {
    optix::float3 direction_to_sphere = ray_origin - sphere_center;
    float b = optix::dot(direction_to_sphere, ray_direction);
    float c = optix::dot(direction_to_sphere, direction_to_sphere) - sphere_radius * sphere_radius;
    float disc = b * b - c;
    if (disc > 0.0f)
        return -b - sqrtf(disc);
    else
        return -1e30f;

}

__inline_all__ float ray_sphere(const optix::Ray& ray, const Sphere& sphere) {
    return ray_sphere(ray.origin, ray.direction, sphere.center, sphere.radius);
}

} // NS Intersect
} // NS OptiXRenderer

#endif // _OPTIXRENDERER_INTERSECT_H_