// OptiX path tracing ray generation and miss program.
// ---------------------------------------------------------------------------
// Copyright (C) 2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#include <OptiXRenderer/Shading/ShadingModels/DefaultShading.h>
#include <OptiXRenderer/Shading/LightSources/LightImpl.h>
#include <OptiXRenderer/TBN.h>
#include <OptiXRenderer/Types.h>

#include <optix.h>

using namespace optix;
using namespace OptiXRenderer;
using namespace OptiXRenderer::Shading::ShadingModels;

// Ray parameters.
rtDeclareVariable(Ray, ray, rtCurrentRay, );
rtDeclareVariable(float, t_hit, rtIntersectionDistance, );
rtDeclareVariable(MonteCarloPayload, monte_carlo_payload, rtPayload, );

// Scene parameters.
rtDeclareVariable(rtObject, g_scene_root, , );
rtDeclareVariable(float, g_scene_epsilon, , );
rtBuffer<Light, 1> g_lights;
rtDeclareVariable(int, g_light_count, , );
rtDeclareVariable(int, g_max_bounce_count, , );
rtDeclareVariable(int, g_accumulations, , );

// Material parameters.
rtBuffer<Material, 1> g_materials;
rtDeclareVariable(int, material_index, , );

//----------------------------------------------------------------------------
// Closest hit program for monte carlo sampling rays.
//----------------------------------------------------------------------------

rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, );
rtDeclareVariable(float3, shading_normal, attribute shading_normal, );
rtDeclareVariable(float2, texcoord, attribute texcoord, );

//-----------------------------------------------------------------------------
// Light sampling.
//-----------------------------------------------------------------------------

// Sample a single light source, evaluates the material's response to the light and 
// stores the combined response in the light source's radiance member.
__inline_dev__ LightSample sample_single_light(const DefaultShading& material, const TBN& world_shading_tbn) {
    int light_index = min(g_light_count - 1, int(monte_carlo_payload.rng.sample1f() * g_light_count));
    const Light& light = g_lights[light_index];
    LightSample light_sample = LightSources::sample_radiance(light, monte_carlo_payload.position, monte_carlo_payload.rng.sample2f());
    light_sample.radiance *= g_light_count; // Scale up radiance to account for only sampling one light.

    float N_dot_L = dot(world_shading_tbn.get_normal(), light_sample.direction_to_light);
    light_sample.radiance *= abs(N_dot_L) / light_sample.PDF;

    // Apply MIS weights if the light isn't a delta function and if a new material ray will be spawned, i.e. it isn't the final bounce.
    const float3 shading_light_direction = world_shading_tbn * light_sample.direction_to_light;
    BSDFResponse bsdf_response = material.evaluate_with_PDF(monte_carlo_payload.direction, shading_light_direction);
    bool delta_light = LightSources::is_delta_light(light, monte_carlo_payload.position);
    bool apply_MIS = !delta_light && monte_carlo_payload.bounces < g_max_bounce_count;
    if (apply_MIS) { // TODO Try using math instead and profile using test scene.
        float mis_weight = RNG::power_heuristic(light_sample.PDF, bsdf_response.PDF);
        light_sample.radiance *= mis_weight;
    } else
        // BIAS Nearly specular materials and delta lights will lead to insane fireflies, so we clamp them here.
        bsdf_response.weight = fminf(bsdf_response.weight, make_float3(32.0f));

    light_sample.radiance = clamp_light_contribution_by_path_PDF(light_sample.radiance, monte_carlo_payload.clamped_path_PDF, g_accumulations);

    // Inline the material response into the light sample's radiance.
    light_sample.radiance *= bsdf_response.weight;

    return light_sample;
}

// Take multiple light samples and from that set pick one based on the contribution of the light scaled by the material.
// Basic Resampled importance sampling: http://scholarsarchive.byu.edu/cgi/viewcontent.cgi?article=1662&context=etd.
__inline_dev__ LightSample reestimated_light_samples(const DefaultShading& material, const TBN& world_shading_tbn, int samples) {
    LightSample light_sample = sample_single_light(material, world_shading_tbn);
    for (int s = 1; s < samples; ++s) {
        LightSample new_light_sample = sample_single_light(material, world_shading_tbn);
        float light_weight = sum(light_sample.radiance);
        float new_light_weight = sum(new_light_sample.radiance);
        float new_light_probability = new_light_weight / (light_weight + new_light_weight);
        if (monte_carlo_payload.rng.sample1f() < new_light_probability) {
            light_sample = new_light_sample;
            light_sample.radiance /= new_light_probability;
        } else
            light_sample.radiance /= 1.0f - new_light_probability;
    }
    light_sample.radiance /= samples;

    return light_sample;
}

//-----------------------------------------------------------------------------
// Closest hit integrator.
//-----------------------------------------------------------------------------

RT_PROGRAM void closest_hit() {
    // const float3 world_geometric_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometric_normal));
    const float3 world_shading_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
    const float3 forward_shading_normal = -dot(world_shading_normal, ray.direction) >= 0.0f ? world_shading_normal : -world_shading_normal;

    const TBN world_shading_tbn = TBN(forward_shading_normal);

    const Material& material_parameter = g_materials[material_index];
    const DefaultShading material = DefaultShading(material_parameter, texcoord);

    float coverage = material.coverage(texcoord);
    if (monte_carlo_payload.rng.sample1f() > coverage) {
        monte_carlo_payload.position = ray.direction * (t_hit + g_scene_epsilon) + ray.origin;
        return;
    }

    // Store intersection point and wo in payload.
    monte_carlo_payload.position = ray.direction * t_hit + ray.origin;
    monte_carlo_payload.direction = world_shading_tbn * -ray.direction;

#if ENABLE_NEXT_EVENT_ESTIMATION
    // Sample a light source.
    if (g_light_count != 0) {
        const LightSample light_sample = reestimated_light_samples(material, world_shading_tbn, 3);

        if (light_sample.radiance.x > 0.0f || light_sample.radiance.y > 0.0f || light_sample.radiance.z > 0.0f) {
            ShadowPayload shadow_payload = { light_sample.radiance };
            Ray shadow_ray(monte_carlo_payload.position, light_sample.direction_to_light, RayTypes::Shadow, g_scene_epsilon, light_sample.distance - g_scene_epsilon);
            rtTrace(g_scene_root, shadow_ray, shadow_payload);

            monte_carlo_payload.radiance += monte_carlo_payload.throughput * shadow_payload.attenuation;
        }
    }
#endif // ENABLE_NEXT_EVENT_ESTIMATION

    // Sample BSDF.
    BSDFSample bsdf_sample = material.sample_all(monte_carlo_payload.direction, monte_carlo_payload.rng.sample3f());
    monte_carlo_payload.direction = bsdf_sample.direction * world_shading_tbn;
    monte_carlo_payload.bsdf_MIS_PDF = bsdf_sample.PDF;
    monte_carlo_payload.path_PDF *= bsdf_sample.PDF;
    monte_carlo_payload.clamped_path_PDF *= fminf(bsdf_sample.PDF, 1.0f);
    if (!is_PDF_valid(bsdf_sample.PDF))
        monte_carlo_payload.throughput = make_float3(0.0f);
    else
        monte_carlo_payload.throughput *= bsdf_sample.weight * (abs(bsdf_sample.direction.z) / bsdf_sample.PDF); // f * ||cos(theta)|| / pdf
    monte_carlo_payload.bounces += 1u;
}

//----------------------------------------------------------------------------
// Any hit program for monte carlo shadow rays.
//----------------------------------------------------------------------------

rtDeclareVariable(ShadowPayload, shadow_payload, rtPayload, );

RT_PROGRAM void shadow_any_hit() {
    float coverage = DefaultShading::coverage(g_materials[material_index], texcoord);
    shadow_payload.attenuation *= 1.0f - coverage;
    if (shadow_payload.attenuation.x < 0.0000001f && shadow_payload.attenuation.y < 0.0000001f && shadow_payload.attenuation.z < 0.0000001f)
        rtTerminateRay();
}

//=============================================================================
// Closest hit programs for monte carlo light sources.
//=============================================================================

RT_PROGRAM void light_closest_hit() {

    int light_index = __float_as_int(geometric_normal.x);
    const SphereLight& light = g_lights[light_index].sphere;

    bool next_event_estimated = monte_carlo_payload.bounces != 0; // Was next event estimated at previous intersection.
    float3 light_radiance = LightSources::evaluate_intersection(light, ray.origin, ray.direction, 
                                                                monte_carlo_payload.bsdf_MIS_PDF, next_event_estimated);

    float3 scaled_radiance = clamp_light_contribution_by_path_PDF(light_radiance, monte_carlo_payload.clamped_path_PDF, g_accumulations);
    monte_carlo_payload.radiance += monte_carlo_payload.throughput * scaled_radiance;

    monte_carlo_payload.throughput = make_float3(0.0f);
}