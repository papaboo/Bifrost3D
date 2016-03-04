// OptiX path tracing ray generation and miss program.
// ---------------------------------------------------------------------------
// Copyright (C) 2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#include <OptiXRenderer/Types.h>
#include <OptiXRenderer/Shading/LightSources/PointLightImpl.h>

#include <optix.h>

using namespace OptiXRenderer;
using namespace optix;

// Ray params
rtDeclareVariable(Ray, ray, rtCurrentRay, );
rtDeclareVariable(float, t_hit, rtIntersectionDistance, );
rtDeclareVariable(MonteCarloPRD, monte_carlo_PRD, rtPayload, );

// Scene params
rtDeclareVariable(rtObject, g_scene_root, , );
rtDeclareVariable(float, g_scene_epsilon, , );
rtBuffer<PointLight, 1> g_lights;
rtDeclareVariable(int, g_light_count, , );

// Material params
rtDeclareVariable(float3, g_color, , );

//----------------------------------------------------------------------------
// Closest hit program for monte carlo sampling rays.
//----------------------------------------------------------------------------

rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, );
rtDeclareVariable(float3, shading_normal, attribute shading_normal, );
rtDeclareVariable(float2, texcoord, attribute texcoord, );

RT_PROGRAM void closest_hit() {
    // const float3 world_geometric_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometric_normal));
    const float3 world_shading_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
    const float3 forward_shading_normal = dot(world_shading_normal, -ray.direction) >= 0.0f ? world_shading_normal : -world_shading_normal;

    const float3 intersection_point = ray.direction * t_hit + ray.origin;

    for (int i = 0; i < g_light_count; ++i) {
        const PointLight& light = g_lights[i];
        LightSample light_sample = LightSources::sample_radiance(light, intersection_point, make_float2(0.0f, 0.0f));
        float N_dot_L = dot(forward_shading_normal, light_sample.direction);
        light_sample.radiance *= abs(N_dot_L);

        const float3 bsdf_response = g_color / PIf;
        if (dot(forward_shading_normal, light_sample.direction) >= 0.0f) {
            ShadowPRD shadow_PRD = { 1.0f, 1.0f, 1.0f };
            // TODO Always offset slightly along the geometric normal?
            // float3 origin_offset = world_geometric_normal * g_scene_epsilon * (dot(world_geometric_normal, light_sample.direction) >= 0.0f ? 1.0f : -1.0f);
            Ray shadow_ray(intersection_point, light_sample.direction, unsigned int(RayTypes::Shadow), g_scene_epsilon, light_sample.distance - g_scene_epsilon);
            rtTrace(g_scene_root, shadow_ray, shadow_PRD);

            monte_carlo_PRD.radiance += light_sample.radiance * bsdf_response * shadow_PRD.attenuation;
        }
    }
}

//----------------------------------------------------------------------------
// Any hit program for monte carlo shadow rays.
//----------------------------------------------------------------------------

rtDeclareVariable(ShadowPRD, shadow_PRD, rtPayload, );

RT_PROGRAM void shadow_any_hit() {
    shadow_PRD.attenuation = make_float3(0.0f);
    rtTerminateRay();
}

//=============================================================================
// Closest hit programs for monte carlo light sources.
//=============================================================================

RT_PROGRAM void light_closest_hit() {

    int light_index = __float_as_int(geometric_normal.x);

    // This should be sampled by specular rays right now!
    const PointLight& light = g_lights[light_index];
    const float power_normalizer = 4.0f * PIf * PIf * light.radius * light.radius;
    const float3 light_radiance = light.power / power_normalizer;
    monte_carlo_PRD.radiance += monte_carlo_PRD.throughput * light_radiance;
}