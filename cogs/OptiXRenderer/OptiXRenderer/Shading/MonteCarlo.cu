// OptiX path tracing ray generation and miss program.
// ---------------------------------------------------------------------------
// Copyright (C) 2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#include <OptiXRenderer/Shading/ShadingModels/DefaultShading.h>
#include <OptiXRenderer/Shading/LightSources/SphereLightImpl.h>
#include <OptiXRenderer/TBN.h>
#include <OptiXRenderer/Types.h>

#include <optix.h>

using namespace optix;
using namespace OptiXRenderer;
using namespace OptiXRenderer::Shading::ShadingModels;

// Ray params
rtDeclareVariable(Ray, ray, rtCurrentRay, );
rtDeclareVariable(float, t_hit, rtIntersectionDistance, );
rtDeclareVariable(MonteCarloPRD, monte_carlo_PRD, rtPayload, );

// Scene params
rtDeclareVariable(rtObject, g_scene_root, , );
rtDeclareVariable(float, g_scene_epsilon, , );
rtBuffer<SphereLight, 1> g_lights;
rtDeclareVariable(int, g_light_count, , );

// Material params
rtBuffer<Material, 1> g_materials;
rtDeclareVariable(int, material_index, , );

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

    const TBN world_shading_tbn = TBN(forward_shading_normal);

    // Store intersection point and wo in PRD.
    monte_carlo_PRD.position = ray.direction * t_hit + ray.origin;
    monte_carlo_PRD.direction = world_shading_tbn * -ray.direction;

    const Material& material_parameter = g_materials[material_index];
    DefaultShading material = DefaultShading(material_parameter);

    // Sample light sources.
    for (int i = 0; i < g_light_count; ++i) {
        const SphereLight& light = g_lights[i];
        LightSample light_sample = LightSources::sample_radiance(light, monte_carlo_PRD.position, monte_carlo_PRD.rng.sample2f());
        float N_dot_L = dot(world_shading_tbn.get_normal(), light_sample.direction);
        light_sample.radiance *= abs(N_dot_L) / light_sample.PDF;

        const float3 shading_light_direction = world_shading_tbn * light_sample.direction;
        const float3 bsdf_response = material.evaluate(monte_carlo_PRD.direction, shading_light_direction);
        if (N_dot_L >= 0.0f) {
            ShadowPRD shadow_PRD = { 1.0f, 1.0f, 1.0f };
            // TODO Always offset slightly along the geometric normal?
            // float3 origin_offset = world_geometric_normal * g_scene_epsilon * (dot(world_geometric_normal, light_sample.direction) >= 0.0f ? 1.0f : -1.0f);
            Ray shadow_ray(monte_carlo_PRD.position, light_sample.direction, unsigned int(RayTypes::Shadow), g_scene_epsilon, light_sample.distance - g_scene_epsilon);
            rtTrace(g_scene_root, shadow_ray, shadow_PRD);

            monte_carlo_PRD.radiance += monte_carlo_PRD.throughput * light_sample.radiance * bsdf_response * shadow_PRD.attenuation;
        }
    }

    // Sample material.
    BSDFSample bsdf_sample = material.sample_one_BSDF(monte_carlo_PRD.direction, monte_carlo_PRD.rng.sample3f());
    monte_carlo_PRD.direction = bsdf_sample.direction * world_shading_tbn;
    monte_carlo_PRD.bsdf_sample_pdf = bsdf_sample.PDF;
    if (!bsdf_sample.is_valid())
        monte_carlo_PRD.throughput = make_float3(0.0f);
    else
        monte_carlo_PRD.throughput *= bsdf_sample.weight * (abs(bsdf_sample.direction.z) / bsdf_sample.PDF); // f * ||cos(theta)|| / pdf
    monte_carlo_PRD.bounces += 1u;
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

    if (monte_carlo_PRD.bounces == 0) {
        // This should only be sampled by rays leaving specular BRDFs right now!
        int light_index = __float_as_int(geometric_normal.x);
        const SphereLight& light = g_lights[light_index];

        const float power_normalizer = 4.0f * PIf * PIf * light.radius * light.radius;
        const float3 light_radiance = light.power / power_normalizer;
        monte_carlo_PRD.radiance += monte_carlo_PRD.throughput * light_radiance;
    }

    monte_carlo_PRD.throughput = make_float3(0.0f);
}