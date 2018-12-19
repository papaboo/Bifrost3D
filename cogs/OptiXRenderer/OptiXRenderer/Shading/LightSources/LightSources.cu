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

rtDeclareVariable(SceneStateGPU, g_scene, , );

// Encode light index in geometric_normal.x
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, );
rtDeclareVariable(float3, shading_normal, attribute shading_normal, );

//=============================================================================
// Sphere light intersection programs.
//=============================================================================
RT_PROGRAM void intersect(int primitive_index) {

    const Light& light = rtBufferId<Light, 1>(g_scene.light_buffer_ID)[primitive_index];

    // Only sphere lights can be intersected.
    if (light.get_type() != Light::Sphere) return;

    const SphereLight& sphere_light = light.sphere;

    float t = Intersect::ray_sphere(ray, Sphere::make(sphere_light.position, sphere_light.radius));
    if (t > 0.0f && rtPotentialIntersection(t)) {
        float3 intersection_point = t * ray.direction + ray.origin;
        float inv_radius = 1.0f / sphere_light.radius;
        shading_normal = (intersection_point - sphere_light.position) * inv_radius;
        geometric_normal.x = __int_as_float(primitive_index);
        rtReportIntersection(0);
    }
}

RT_PROGRAM void bounds(int primitive_index, float result[6]) {
    optix::Aabb* aabb = (optix::Aabb*)result;

    const Light& light = rtBufferId<Light, 1>(g_scene.light_buffer_ID)[primitive_index];
    if (light.get_type() != Light::Sphere) {
        aabb->invalidate();
        return;
    }

    const SphereLight& sphere_light = light.sphere;

    if (sphere_light.radius > 0.0f) {
        aabb->m_min = sphere_light.position - sphere_light.radius;
        aabb->m_max = sphere_light.position + sphere_light.radius;
    } else
        aabb->invalidate();
}