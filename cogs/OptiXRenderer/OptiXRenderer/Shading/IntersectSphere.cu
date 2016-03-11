// OptiX sphere intersection program.
// ---------------------------------------------------------------------------
// Copyright (C) 2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------
// Inspired by the OptiX samples.
// ---------------------------------------------------------------------------

#include <OptiXRenderer/Intersect.h>
#include <OptiXRenderer/Shading/Utils.h>

#include <optix.h>
#include <optixu/optixu_aabb.h>
#include <optixu/optixu_math.h>

using namespace optix;
using namespace OptiXRenderer;

rtDeclareVariable(Sphere, sphere, , );

rtDeclareVariable(Ray, ray, rtCurrentRay, );

rtDeclareVariable(float2, texcoord, attribute texcoord, );
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, );
rtDeclareVariable(float3, shading_normal, attribute shading_normal, );

RT_PROGRAM void intersect(int) {
    float t = Intersect::ray_sphere(ray, sphere);
    if (t > 0.0f && rtPotentialIntersection(t)) {
        float3 intersection_point = t * ray.direction + ray.origin;
        float inv_radius = 1.0f / sphere.radius;
        shading_normal = geometric_normal = (intersection_point - sphere.center) * inv_radius;
        float s = (shading_normal.z + 1.0f) * 0.5f;
        float t = (atan2(shading_normal.y, shading_normal.x) + PIf) * 0.5f / PIf;
        texcoord = make_float2(s, t);
        rtReportIntersection(0);
    }
}

RT_PROGRAM void bounds(int, float result[6]) {
    optix::Aabb* aabb = (optix::Aabb*)result;

    aabb->m_min = sphere.center - sphere.radius;
    aabb->m_max = sphere.center + sphere.radius;
}
