// OptiX light source intersection programs.
// ---------------------------------------------------------------------------
// Copyright (C) 2015-2016, Bifrost. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#include <OptiXRenderer/Intersect.h>

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
        t = Intersect::ray_sphere(ray, Sphere::make(to_float3(sphere_light.get_position()), sphere_light.get_radius()));
    } else if (light.get_type() == Light::Disk) {
        const Bifrost::Math::Disk disk = light.disk.get_surface();
        t = Intersect::ray_disk(ray, Disk::make(to_float3(disk.center), to_float3(disk.normal), disk.radius));
    } else if (light.get_type() == Light::Spot) {
        const SpotLight spot_light = light.spot;
        t = Intersect::ray_disk(ray, Disk::make(to_float3(spot_light.position), to_float3(spot_light.direction), spot_light.radius));
    }

    if (rtPotentialIntersection(t)) {
        primitive_index = prim_index;

        float3 coarse_intersection_point = t * ray.direction + ray.origin;
        if (light.get_type() == Light::Sphere) {
            const SphereLight sphere_light = light.sphere;
            float3 light_center = to_float3(sphere_light.get_position());
            shading_normal = normalize(coarse_intersection_point - light_center);

            // Computing the intersection point using origin + t * direction can be unstable if t is large.
            // To avoid this the intersection point is recomputed wrt the shading normal,
            // to ensure that the intersection point is as close to the sphere surface as possible.
            intersection_point = light_center + sphere_light.get_radius() * shading_normal;
        } else if (light.get_type() == Light::Disk) {
            const Bifrost::Math::Disk disk = light.disk.get_surface();
            shading_normal = to_float3(disk.normal);

            // Computing the intersection point using origin + t * direction can be unstable if t is large.
            // So the coarse but close intersection point is projected on to the plane.
            float t_fine = Intersect::point_distance_to_plane(coarse_intersection_point, to_float3(disk.center), to_float3(disk.normal));
            intersection_point = coarse_intersection_point + t_fine * to_float3(disk.normal);
        } else if (light.get_type() == Light::Spot) {
            const SpotLight spot_light = light.spot;
            shading_normal = to_float3(spot_light.direction);

            // Computing the intersection point using origin + t * direction can be unstable if t is large.
            // So the coarse but close intersection point is projected on to the plane.
            float t_fine = Intersect::point_distance_to_plane(coarse_intersection_point, to_float3(spot_light.position), to_float3(spot_light.direction));
            intersection_point = coarse_intersection_point + t_fine * to_float3(spot_light.direction);
        }

        geometric_normal = shading_normal;

        rtReportIntersection(0);
    }
}

RT_PROGRAM void bounds(int primitive_index, float result[6]) {
    const Light& light = g_scene.light_buffer[primitive_index];
    optix::float3 position;
    float radius = 0.0f;
    if (light.get_type() == Light::Sphere) {
        position = to_float3(light.sphere.get_position());
        radius = light.sphere.get_radius();
    } else if (light.get_type() == Light::Disk) {
        position = to_float3(light.disk.get_surface().center);
        radius = light.disk.get_surface().radius;
    } else if (light.get_type() == Light::Spot) {
        position = to_float3(light.spot.position);
        radius = light.spot.radius;
    }

    // TODO Tighter bounds around disk?
    optix::Aabb* aabb = (optix::Aabb*)result;
    if (radius > 0.0f) {
        aabb->m_min = position - radius;
        aabb->m_max = position + radius;
    } else
        aabb->invalidate();
}