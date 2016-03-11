// OptiX primitive intersections.
// ---------------------------------------------------------------------------
// Copyright (C) 2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _OPTIXRENDERER_INTERSECT_H_
#define _OPTIXRENDERER_INTERSECT_H_

#include <OptiXRenderer/Types.h>

#include <optixu/optixu_math_namespace.h>

namespace OptiXRenderer {
namespace Intersect {

// Intersection of ray and sphere.
// Returns the distance to the sphere or negative if no hit.
__inline_all__ float ray_sphere(const optix::Ray& ray, const Sphere& sphere) {
    optix::float3 o = ray.origin - sphere.center;
    float b = optix::dot(o, ray.direction);
    float c = optix::dot(o, o) - sphere.radius * sphere.radius;
    float disc = b * b - c;
    if (disc > 0.0f)
        return -b - sqrtf(disc);
    else
        return -1e30f;
}

} // NS Intersect
} // NS OptiXRenderer

#endif // _OPTIXRENDERER_INTERSECT_H_