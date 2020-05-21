// OptiX light source intersection programs.
// ---------------------------------------------------------------------------
// Copyright (C) 2015-2016, Bifrost. See AUTHORS.txt for authors
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

    const Light& light = g_scene.light_buffer[primitive_index];

    float t = -1e30f;
    float3 local_shading_normal;
    // Only lights with a spherical geometry can be intersected.
    if (light.get_type() == Light::Sphere) {
        const SphereLight& sphere_light = light.sphere;
        t = Intersect::ray_sphere(ray, Sphere::make(sphere_light.position, sphere_light.radius));
        float3 intersection_point = t * ray.direction + ray.origin;
        float inv_radius = 1.0f / sphere_light.radius;
        local_shading_normal = (intersection_point - sphere_light.position) * inv_radius;

    } else if (light.get_type() == Light::Spot) {
        const SpotLight spot_light = light.spot;
        t = Intersect::ray_disk(ray, Disk::make(spot_light.position, spot_light.direction, spot_light.radius));
        local_shading_normal = spot_light.direction;
    }

    if (t > 0.0f && rtPotentialIntersection(t)) {
        shading_normal = local_shading_normal;
        geometric_normal.x = __int_as_float(primitive_index);
        rtReportIntersection(0);
    }
}

RT_PROGRAM void bounds(int primitive_index, float result[6]) {
    optix::Aabb* aabb = (optix::Aabb*)result;

    const Light& light = g_scene.light_buffer[primitive_index];
    if (light.get_type() != Light::Sphere && light.get_type() != Light::Spot) {
        aabb->invalidate();
        return;
    }

    // Light is either a sphere light or a spot light.
    // TODO Tighter bounds around disk?
    bool is_sphere_light = light.get_type() == Light::Sphere;
    float radius = is_sphere_light ? light.sphere.radius : light.spot.radius;
    if (radius > 0.0f) {
        optix::float3 position = is_sphere_light ? light.sphere.position : light.spot.position;
        aabb->m_min = position - radius;
        aabb->m_max = position + radius;
    } else
        aabb->invalidate();
}