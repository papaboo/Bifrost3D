// OptiX path tracing ray generation and miss program.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#include <OptiXRenderer/MonteCarlo.h>
#include <OptiXRenderer/Shading/ShadingModels/DefaultShading.h>
#include <OptiXRenderer/Shading/ShadingModels/DiffuseShading.h>
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

// Camera parameters
rtDeclareVariable(CameraStateGPU, g_camera_state, , );

// Scene parameters.
rtDeclareVariable(rtObject, g_scene_root, , );
rtDeclareVariable(SceneStateGPU, g_scene, , );

// Material parameters.
rtBuffer<Material, 1> g_materials;
rtDeclareVariable(int, material_index, , );

// Renderer config
rtBuffer<float4, 1> g_random_sample_offsets;

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
template <class ShadingModel>
__inline_dev__ LightSample sample_single_light(const ShadingModel& material, const TBN& world_shading_tbn, float3 random_sample) {
    int light_index = min(g_scene.light_count - 1, int(random_sample.z * g_scene.light_count));
    const Light& light = g_scene.light_buffer[light_index];
    LightSample light_sample = LightSources::sample_radiance(light, monte_carlo_payload.position, make_float2(random_sample));
    light_sample.radiance *= g_scene.light_count; // Scale up radiance to account for only sampling one light.

    float N_dot_L = dot(world_shading_tbn.get_normal(), light_sample.direction_to_light);
    light_sample.radiance *= abs(N_dot_L) / light_sample.PDF;

    // Apply MIS weights if the light isn't a delta function and if a new material ray will be spawned, i.e. it isn't the final bounce.
    const float3 shading_light_direction = world_shading_tbn * light_sample.direction_to_light;
    BSDFResponse bsdf_response = material.evaluate_with_PDF(monte_carlo_payload.direction, shading_light_direction);
    bool delta_light = LightSources::is_delta_light(light, monte_carlo_payload.position);
    bool apply_MIS = !delta_light && monte_carlo_payload.bounces < g_camera_state.max_bounce_count;
    if (apply_MIS)
        light_sample.radiance *= MonteCarlo::MIS_weight(light_sample.PDF, bsdf_response.PDF);
    else
        // BIAS Nearly specular materials and delta lights will lead to insane fireflies, so we clamp them here.
        bsdf_response.reflectance = fminf(bsdf_response.reflectance, make_float3(32.0f));

    // Inline the material response into the light sample's radiance.
    light_sample.radiance *= bsdf_response.reflectance;

    return light_sample;
}

// Take multiple light samples and from that set pick one based on the contribution of the light scaled by the material.
// Basic Resampled importance sampling: http://scholarsarchive.byu.edu/cgi/viewcontent.cgi?article=1662&context=etd.
template <class ShadingModel>
__inline_dev__ LightSample reestimated_light_samples(const ShadingModel& material, const TBN& world_shading_tbn) {
    float4 light_random_4f = monte_carlo_payload.rng.sample4f();
    float3 light_random_number = make_float3(light_random_4f);
    LightSample light_sample = sample_single_light(material, world_shading_tbn, light_random_number);

    int light_sample_count = g_scene.next_event_sample_count;
    for (int s = 1; s < light_sample_count; ++s) {
        // Grab the next light sample
        light_random_4f += g_random_sample_offsets[s];
        light_random_4f = light_random_4f - floor(light_random_4f);
        light_random_number = make_float3(light_random_4f);
        float use_new_light_decision = light_random_4f.w;

        // Sample another light and compute the probability of keeping it.
        LightSample new_light_sample = sample_single_light(material, world_shading_tbn, light_random_number);
        float light_weight = sum(light_sample.radiance);
        float new_light_weight = sum(new_light_sample.radiance);
        float new_light_probability = new_light_weight / (light_weight + new_light_weight);

        // Decide which light to keep and adjust the radiance.
        if (use_new_light_decision < new_light_probability) {
            light_sample = new_light_sample;
            light_sample.radiance /= new_light_probability;
        } else
            light_sample.radiance /= 1.0f - new_light_probability;
    }
    light_sample.radiance /= light_sample_count;

    return light_sample;
}

//-----------------------------------------------------------------------------
// Closest hit integrator.
//-----------------------------------------------------------------------------

template <typename MaterialCreator>
__inline_all__ void path_tracing_closest_hit() {
    const Material& material_parameter = g_materials[material_index];

    // Backside culling of non-thin-walled geometry.
    const float3 world_geometric_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometric_normal));
    bool hit_from_behind = dot(world_geometric_normal, ray.direction) >= 0.0f;
    bool ignore_intersection = hit_from_behind && !material_parameter.is_thin_walled();

    float4 bsdf_coverage_random_4f = monte_carlo_payload.rng.sample4f(); // Always draw coverage random number to have predictable RNG dimension usage whether the material is a cutout or not.
    float coverage_cutoff = bsdf_coverage_random_4f.w;
    float3 bsdf_random_uvs = make_float3(bsdf_coverage_random_4f);

    float coverage = material_parameter.get_coverage(texcoord);
    coverage_cutoff = material_parameter.is_cutout() ? material_parameter.coverage : coverage_cutoff;
    ignore_intersection |= coverage < coverage_cutoff;

    if (ignore_intersection) {
        // Advance the ray past the intersection by incrementing the min distance past the intersected distance.
        // This should be stable wrt floating point precision as origin and direction are not changed.
        monte_carlo_payload.ray_min_t = nextafterf(t_hit, INFINITY);
        return;
    }

    // Setup world shading normal and tangents.
    // If the surface is hit from behind then we flip the shading normal to the backside of the surface.
    float3 world_shading_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
    world_shading_normal = hit_from_behind ? -world_shading_normal : world_shading_normal;
    monte_carlo_payload.shading_normal = fix_backfacing_shading_normal(-ray.direction, world_shading_normal, 0.001f);
    const TBN world_shading_tbn = TBN(monte_carlo_payload.shading_normal);

    // Store geometry varyings.
    monte_carlo_payload.material_index = material_index; // Store material index after coverage check, since the material isn't actually used unless the coverage check succeeded.
    monte_carlo_payload.texcoord = texcoord;

    // Store intersection point and wo in payload.
    monte_carlo_payload.position = ray.direction * t_hit + ray.origin;
    monte_carlo_payload.direction = world_shading_tbn * -ray.direction;
    float abs_cos_theta_o = abs(monte_carlo_payload.direction.z);
    float pdf_regularization_hint = monte_carlo_payload.bsdf_MIS_PDF.PDF() * g_camera_state.path_regularization_PDF_scale;
    const MaterialCreator::material_type material = MaterialCreator::create(material_parameter, texcoord, abs_cos_theta_o, pdf_regularization_hint);

    // Deferred BSDF sampling.
    // The BSDF is sampled before tracing the shadow ray in order to avoid flushing world_shading_tbn and the material to the local stack when tracing the ray.
    BSDFSample bsdf_sample = material.sample(monte_carlo_payload.direction, bsdf_random_uvs);
    float3 next_payload_direction = bsdf_sample.direction * world_shading_tbn;
    float next_payload_MIS_PDF = bsdf_sample.PDF;
    float3 next_payload_throughput = make_float3(0.0f);
    if (is_PDF_valid(bsdf_sample.PDF))
       next_payload_throughput = monte_carlo_payload.throughput * bsdf_sample.reflectance * (abs(bsdf_sample.direction.z) / bsdf_sample.PDF); // f * ||cos(theta)|| / pdf

#if ENABLE_NEXT_EVENT_ESTIMATION
    // Sample a light source.
    if (g_scene.light_count != 0) {
        // Grab the RNG state before next even estimation to reset it afterwards which reduces the curse of dimensionality.
        // This is acceptable as the paths generated by BSDF sampling and next event estimation are independent.
        auto pre_NNE_rng_state = monte_carlo_payload.rng.get_state();

        monte_carlo_payload.light_sample = reestimated_light_samples(material, world_shading_tbn);
        monte_carlo_payload.light_sample.radiance *= monte_carlo_payload.throughput;

        monte_carlo_payload.rng.set_state(pre_NNE_rng_state);
    }
#endif // ENABLE_NEXT_EVENT_ESTIMATION

    // Apply deferred BSDF sample to the ray payload.
    monte_carlo_payload.ray_min_t = g_scene.ray_epsilon;
    monte_carlo_payload.direction = next_payload_direction;
    monte_carlo_payload.bsdf_MIS_PDF = MisPDF::from_PDF(next_payload_MIS_PDF);
    monte_carlo_payload.throughput = next_payload_throughput;
    monte_carlo_payload.bounces += 1u;
}

//----------------------------------------------------------------------------
// Path tracing closest hit programs for different shading models.
//----------------------------------------------------------------------------

struct DefaultMaterialCreator {
    typedef DefaultShading material_type;
    __inline_all__ static material_type create(const Material& material_params, optix::float2 texcoord, float abs_cos_theta_o, float max_PDF_hint) {
        return DefaultShading::initialize_with_max_PDF_hint(material_params, texcoord, abs_cos_theta_o, max_PDF_hint);
    }
};

RT_PROGRAM void default_closest_hit() {
    path_tracing_closest_hit<DefaultMaterialCreator>();
}

struct DiffuseMaterialCreator {
    typedef DiffuseShading material_type;
    __inline_all__ static material_type create(const Material& material_params, optix::float2 texcoord, float abs_cos_theta_o, float max_PDF_hint) {
        float3 tint = make_float3(material_params.get_tint_roughness(texcoord));
        return DiffuseShading(tint);
    }
};

RT_PROGRAM void diffuse_closest_hit() {
    path_tracing_closest_hit<DiffuseMaterialCreator>();
}

//----------------------------------------------------------------------------
// Any hit program for monte carlo shadow rays.
//----------------------------------------------------------------------------

rtDeclareVariable(ShadowPayload, shadow_payload, rtPayload, );

RT_PROGRAM void shadow_any_hit() {
    float coverage = g_materials[material_index].get_coverage(texcoord);
    shadow_payload.radiance *= 1.0f - coverage;
    if (shadow_payload.radiance.x < 0.0000001f && shadow_payload.radiance.y < 0.0000001f && shadow_payload.radiance.z < 0.0000001f) {
        shadow_payload.radiance = make_float3(0, 0, 0);
        rtTerminateRay();
    }
}

//=============================================================================
// Closest hit programs for monte carlo light sources.
//=============================================================================

RT_PROGRAM void light_closest_hit() {

    int light_index = __float_as_int(geometric_normal.x);
    bool next_event_estimated = monte_carlo_payload.bounces != 0; // Was next event estimated at previous intersection.

    const Light light = g_scene.light_buffer[light_index];
    float3 light_radiance = make_float3(0, 0, 0);
    if (light.get_type() == Light::Sphere)
        light_radiance = LightSources::evaluate_intersection(light.sphere, ray.origin, ray.direction,
                                                             monte_carlo_payload.bsdf_MIS_PDF, next_event_estimated);
    else if (light.get_type() == Light::Spot)
        light_radiance = LightSources::evaluate_intersection(light.spot, ray.origin, ray.direction,
                                                             monte_carlo_payload.bsdf_MIS_PDF, next_event_estimated);

    monte_carlo_payload.radiance += monte_carlo_payload.throughput * light_radiance;
    monte_carlo_payload.throughput = make_float3(0.0f);
    monte_carlo_payload.position = ray.direction * t_hit + ray.origin;
    monte_carlo_payload.shading_normal = shading_normal;
}