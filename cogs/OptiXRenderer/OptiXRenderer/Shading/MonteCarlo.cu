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

__inline_dev__ void closest_hit_not_MIS() {
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

        // Inline the material response into the light sample's contribution.
        const float3 shading_light_direction = world_shading_tbn * light_sample.direction;
        const float3 bsdf_response = material.evaluate(monte_carlo_PRD.direction, shading_light_direction);// TODO Extend material and BRDFs with methods for evaluating contribution and PDF at the same time.
        light_sample.radiance *= bsdf_response;

        if (light_sample.radiance.x > 0.0f || light_sample.radiance.y > 0.0f || light_sample.radiance.z > 0.0f) {
            ShadowPRD shadow_PRD = { light_sample.radiance };
            Ray shadow_ray(monte_carlo_PRD.position, light_sample.direction, unsigned int(RayTypes::Shadow), g_scene_epsilon, light_sample.distance - g_scene_epsilon);
            rtTrace(g_scene_root, shadow_ray, shadow_PRD);

            monte_carlo_PRD.radiance += monte_carlo_PRD.throughput * shadow_PRD.attenuation;
        }
    }

    // Sample material.
    BSDFSample bsdf_sample = material.sample_all(monte_carlo_PRD.direction, monte_carlo_PRD.rng.sample3f());
    monte_carlo_PRD.direction = bsdf_sample.direction * world_shading_tbn;
    monte_carlo_PRD.bsdf_MIS_PDF = 0.0f; // bsdf_sample.PDF;
    monte_carlo_PRD.path_PDF *= bsdf_sample.PDF;
    if (!is_PDF_valid(bsdf_sample.PDF))
        monte_carlo_PRD.throughput = make_float3(0.0f);
    else
        monte_carlo_PRD.throughput *= bsdf_sample.weight * (abs(bsdf_sample.direction.z) / bsdf_sample.PDF); // f * ||cos(theta)|| / pdf
    monte_carlo_PRD.bounces += 1u;
}

__inline_dev__ void closest_hit_MIS() {
    // const float3 world_geometric_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometric_normal));
    const float3 world_shading_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
    const float3 forward_shading_normal = dot(world_shading_normal, -ray.direction) >= 0.0f ? world_shading_normal : -world_shading_normal;

    const TBN world_shading_tbn = TBN(forward_shading_normal);

    // Store intersection point and wo in PRD.
    monte_carlo_PRD.position = ray.direction * t_hit + ray.origin;
    monte_carlo_PRD.direction = world_shading_tbn * -ray.direction;

    const Material& material_parameter = g_materials[material_index];
    DefaultShading material = DefaultShading(material_parameter);

    // Sample a light source.
    bool apply_MIS = false;
    if (g_light_count != 0) {
        int light_index = min(g_light_count - 1, int(monte_carlo_PRD.rng.sample1f() * g_light_count));
        const SphereLight& light = g_lights[light_index];
        LightSample light_sample = LightSources::sample_radiance(light, monte_carlo_PRD.position, monte_carlo_PRD.rng.sample2f());
        light_sample.radiance *= g_light_count; // Scale up radiance to account for only sampling one light.

        // Inline the material response into the light sample's radiance.
        const float3 shading_light_direction = world_shading_tbn * light_sample.direction;
        const float3 bsdf_response = material.evaluate(monte_carlo_PRD.direction, shading_light_direction);// TODO Extend material and BRDFs with methods for evaluating contribution and PDF at the same time.
        light_sample.radiance *= bsdf_response;

        float N_dot_L = dot(world_shading_tbn.get_normal(), light_sample.direction);
        light_sample.radiance *= abs(N_dot_L) / light_sample.PDF;

        // Apply MIS weights if the light isn't a delta function and if a new material ray will be spawned, i.e. it isn't the final bounce.
        bool delta_light = LightSources::is_delta_light(light, monte_carlo_PRD.position);
        apply_MIS = !delta_light && monte_carlo_PRD.bounces < 4;
        if (apply_MIS) {
            float bsdf_PDF = material.PDF(monte_carlo_PRD.direction, shading_light_direction);
            float mis_weight = RNG::power_heuristic(light_sample.PDF, bsdf_PDF); // TODO Check if the BSDF material PDF is valid. If it isn't we then disable MIS intirely? Or set contribution to black?
            rtPrintf("light PDF %f, bsdf PDF %f, MIS weight %f\n", light_sample.PDF, bsdf_PDF, mis_weight);

            light_sample.radiance *= mis_weight;
        }

        if (light_sample.radiance.x > 0.0f || light_sample.radiance.y > 0.0f || light_sample.radiance.z > 0.0f) {
            ShadowPRD shadow_PRD = { light_sample.radiance };
            Ray shadow_ray(monte_carlo_PRD.position, light_sample.direction, unsigned int(RayTypes::Shadow), g_scene_epsilon, light_sample.distance - g_scene_epsilon);
            rtTrace(g_scene_root, shadow_ray, shadow_PRD);

            monte_carlo_PRD.radiance += monte_carlo_PRD.throughput * shadow_PRD.attenuation;
        }
    }

    // Sample material.
    BSDFSample bsdf_sample = material.sample_all(monte_carlo_PRD.direction, monte_carlo_PRD.rng.sample3f());
    monte_carlo_PRD.direction = bsdf_sample.direction * world_shading_tbn;
    monte_carlo_PRD.bsdf_MIS_PDF = bsdf_sample.PDF;
    monte_carlo_PRD.path_PDF *= bsdf_sample.PDF;
    if (!is_PDF_valid(bsdf_sample.PDF))
        monte_carlo_PRD.throughput = make_float3(0.0f);
    else
        monte_carlo_PRD.throughput *= bsdf_sample.weight * (abs(bsdf_sample.direction.z) / bsdf_sample.PDF); // f * ||cos(theta)|| / pdf
    monte_carlo_PRD.bounces += 1u;
}

// rtDeclareVariable(uint2, g_launch_index, rtLaunchIndex, );
// rtBuffer<float4, 2>  g_accumulation_buffer;

RT_PROGRAM void closest_hit() {
    // if (g_launch_index.x * 2 < g_accumulation_buffer.size().x)
        // closest_hit_not_MIS();
    // else
        closest_hit_MIS();
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
    const SphereLight& light = g_lights[light_index];

    float3 light_radiance = LightSources::evaluate(light, ray.origin, ray.direction);

    bool next_event_estimated = monte_carlo_PRD.bounces != 0;
    bool apply_MIS = monte_carlo_PRD.bsdf_MIS_PDF > 0.0f;
    if (apply_MIS) {
        // Calculate MIS weight and scale the radiance by it.
        const float light_PDF = LightSources::PDF(light, ray.origin, ray.direction);
        float mis_weight = is_PDF_valid(light_PDF) ? RNG::power_heuristic(monte_carlo_PRD.bsdf_MIS_PDF, light_PDF) : 0.0f;
        light_radiance *= mis_weight;
    } else if (next_event_estimated)
        // Previous bounce used next even estimation, but did not calculate MIS, so don't apply light contribution.
        // TODO Could this be handled by setting bsdf_MIS_PDF to 0 instead? Wait until we have a specular BRDF implementation.
        light_radiance = make_float3(0.0f);

    monte_carlo_PRD.radiance += monte_carlo_PRD.throughput * light_radiance;

    monte_carlo_PRD.throughput = make_float3(0.0f);
}