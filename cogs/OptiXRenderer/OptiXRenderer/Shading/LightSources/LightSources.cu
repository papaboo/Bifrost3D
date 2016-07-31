// OptiX light source intersection programs.
// ---------------------------------------------------------------------------
// Copyright (C) 2015-2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#include <OptiXRenderer/Intersect.h>
#include <OptiXRenderer/Shading/LightSources/SphereLightImpl.h>

#include <optix.h>
#include <optixu/optixu_aabb.h>
#include <optixu/optixu_math.h>

using namespace OptiXRenderer;
using namespace optix;

rtDeclareVariable(Ray, ray, rtCurrentRay, );

rtBuffer<Light, 1> g_lights;

// Encode light index in geometric_normal.x
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, );
rtDeclareVariable(float3, shading_normal, attribute shading_normal, );

//=============================================================================
// Sphere light intersection programs.
//=============================================================================
RT_PROGRAM void intersect(int primitive_index) {

    // Only sphere lights can be intersected.
    if (g_lights[primitive_index].get_type() != Light::Sphere) return;

    const SphereLight& light = g_lights[primitive_index].sphere;

    float t = Intersect::ray_sphere(ray, Sphere::make(light.position, light.radius));
    if (t > 0.0f && rtPotentialIntersection(t)) {
        float3 intersection_point = t * ray.direction + ray.origin;
        float inv_radius = 1.0f / light.radius;
        shading_normal = (intersection_point - light.position) * inv_radius;
        geometric_normal.x = __int_as_float(primitive_index);
        rtReportIntersection(0);
    }
}

RT_PROGRAM void bounds(int primitive_index, float result[6]) {
    optix::Aabb* aabb = (optix::Aabb*)result;

    if (g_lights[primitive_index].get_type() != Light::Sphere) {
        aabb->invalidate();
        return;
    }

    const SphereLight& light = g_lights[primitive_index].sphere;

    if (light.radius > 0.0f) {
        aabb->m_min = light.position - light.radius;
        aabb->m_max = light.position + light.radius;
    } else
        aabb->invalidate();
}