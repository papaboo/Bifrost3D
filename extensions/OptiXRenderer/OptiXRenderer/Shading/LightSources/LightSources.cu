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

rtDeclareVariable(float3, intersection_point, attribute intersection_point, );
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, );
rtDeclareVariable(float3, shading_normal, attribute shading_normal, );
rtDeclareVariable(unsigned int, primitive_index, attribute primitive_index, );

//=============================================================================
// Sphere light intersection programs.
//=============================================================================
RT_PROGRAM void intersect(int prim_index) {
    const Light& light = g_scene.light_buffer[prim_index];

    // Only area lights can be intersected.
    float t = -1e30f;
    if (light.get_type() == Light::Sphere) {
        const SphereLight sphere_light = light.sphere;
        t = Intersect::ray_sphere(ray, Sphere::make(sphere_light.position, sphere_light.radius));
    } else if (light.get_type() == Light::Spot) {
        const SpotLight spot_light = light.spot;
        t = Intersect::ray_disk(ray, Disk::make(spot_light.position, spot_light.direction, spot_light.radius));
    }

    if (rtPotentialIntersection(t)) {
        primitive_index = prim_index;

        float3 coarse_intersection_point = t * ray.direction + ray.origin;
        if (light.get_type() == Light::Sphere) {
            const SphereLight sphere_light = light.sphere;
            shading_normal = normalize(coarse_intersection_point - sphere_light.position);

            // Computing the intersection point using origin + t * direction can be unstable if t is large.
            // To avoid this the intersection point is recomputed wrt the shading normal,
            // to ensure that the intersection point is as close to the sphere surface as possible.
            intersection_point = sphere_light.position + sphere_light.radius * shading_normal;
        } else if (light.get_type() == Light::Spot) {
            const SpotLight spot_light = light.spot;
            shading_normal = spot_light.direction;

            // Computing the intersection point using origin + t * direction can be unstable if t is large.
            // So the coarse but close intersection point is projected on to the plane.
            float t_fine = Intersect::point_distance_to_plane(coarse_intersection_point, spot_light.position, spot_light.direction);
            intersection_point = coarse_intersection_point + t_fine * spot_light.direction;
        }

        geometric_normal = shading_normal;

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