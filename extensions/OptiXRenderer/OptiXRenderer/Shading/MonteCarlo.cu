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
#include <OptiXRenderer/Shading/ShadingModels/TransmissiveShading.h>
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
rtDeclareVariable(SceneStateGPU, g_scene, , );

// Material parameters.
rtBuffer<Material, 1> g_materials;

// Model parameters
rtDeclareVariable(ModelState, model_state, , );

// Renderer config
rtBuffer<float4, 1> g_random_sample_offsets;

//----------------------------------------------------------------------------
// Closest hit program for monte carlo sampling rays.
//----------------------------------------------------------------------------

rtDeclareVariable(float3, intersection_point, attribute intersection_point, );
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, );
rtDeclareVariable(float3, shading_normal, attribute shading_normal, );
rtDeclareVariable(float2, texcoord, attribute texcoord, );
rtDeclareVariable(unsigned int, primitive_index, attribute primitive_index, );

//-----------------------------------------------------------------------------
// Light sampling.
//-----------------------------------------------------------------------------

// Sample a single light source, evaluates the material's response to the light and 
// stores the combined response in the light source's radiance member.
template <class ShadingModel>
__inline_dev__ LightSample sample_single_light(const ShadingModel& material, float3 wo, const TBN& world_shading_tbn, float3 random_sample) {
    int light_index = min(g_scene.light_count - 1, int(random_sample.z * g_scene.light_count));
    const Light& light = g_scene.light_buffer[light_index];
    LightSample light_sample = LightSources::sample_radiance(light, monte_carlo_payload.position, make_float2(random_sample));
    light_sample.radiance *= g_scene.light_count; // Scale up radiance to account for only sampling one light.

    float N_dot_L = dot(world_shading_tbn.get_normal(), light_sample.direction_to_light);
    light_sample.radiance *= abs(N_dot_L) / light_sample.PDF;

    // Apply MIS weights if the light isn't a delta function.
    const float3 shading_light_direction = world_shading_tbn * light_sample.direction_to_light;
    BSDFResponse bsdf_response = material.evaluate_with_PDF(wo, shading_light_direction);
    bool apply_MIS = !LightSources::is_delta_light(light, monte_carlo_payload.position);
    if (apply_MIS)
        // The light source connected to the final bounce will be scaled by the MIS weight as well, even though the BSDF sample isn't traced and thus the second sample scheme isn't used.
        // This is done as the MIS weight will still reduce variance from light sources that would be more easily sampled using the BSDf.
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
__inline_dev__ LightSample reestimated_light_samples(const ShadingModel& material, float3 wo, const TBN& world_shading_tbn) {
    // If there are no lights to sample in the scene then early out.
    if (g_scene.light_count == 0)
        return LightSample::none();

    float4 light_random_base = monte_carlo_payload.rng.sample4f();

    LightSample light_sample = LightSample::none();
    int light_sample_count = g_scene.next_event_sample_count;
    for (int s = 0; s < light_sample_count; ++s) {
        // The light sample random numbers from the base random number shifted by the stratified sample offsets.
        float4 light_random_4f = toroidal_shift(light_random_base, g_random_sample_offsets[s]);
        float3 light_random_number = make_float3(light_random_4f);
        float use_new_light_decision = light_random_4f.w;

        // Sample another light and compute the probability of keeping it.
        LightSample new_light_sample = sample_single_light(material, wo, world_shading_tbn, light_random_number);
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
    monte_carlo_payload.light_sample = LightSample::none();

    InstanceID instance_id = model_state.instance_id;
    PrimitiveID primitive_id = PrimitiveID::make(instance_id, primitive_index);

    // Ignore self intersection with the same primitive.
    if (primitive_id == monte_carlo_payload.primitive_id) {
        // Advance the ray past the intersection by incrementing the min distance past the intersected distance.
        // This is stable wrt floating point precision as origin and direction are not changed.
        monte_carlo_payload.ray_min_t = nextafterf(t_hit, INFINITY);
        return;
    }

    const Material& material_parameter = g_materials[model_state.material_index];

    // Backside culling of non-thin-walled geometry.
    float3 world_geometric_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometric_normal));
    bool hit_from_front = dot(world_geometric_normal, ray.direction) < 0.0f;
    bool backside_cull = !hit_from_front && !material_parameter.is_thin_walled();

    float4 bsdf_coverage_random_4f = monte_carlo_payload.rng.sample4f(); // Always draw coverage random number to have predictable RNG dimension usage whether the material is a cutout or not.
    float coverage_cutoff = bsdf_coverage_random_4f.w;
    float3 bsdf_random_uvs = make_float3(bsdf_coverage_random_4f);

    float coverage = material_parameter.get_coverage(texcoord);
    bool discard_from_coverage = coverage < coverage_cutoff;

    if (backside_cull || discard_from_coverage) {
        // Advance the ray past the intersection by incrementing the min distance past the intersected distance.
        // This is stable wrt floating point precision as origin and direction are not changed.
        monte_carlo_payload.ray_min_t = nextafterf(t_hit, INFINITY);
        return;
    }

    // Store geometry varyings and surface properties
    // Store them after successful sefl-intersection and coverage checks to avoid storing state of an ignored surface.
    monte_carlo_payload.material_index = model_state.material_index;
    monte_carlo_payload.texcoord = texcoord;
    monte_carlo_payload.primitive_id = primitive_id;

    // Setup normals and tangents.
    // If the surface is hit from behind then we flip the shading and geometric normal to the backside of the surface.
    world_geometric_normal = hit_from_front ? world_geometric_normal : -world_geometric_normal;
    float3 world_shading_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
    world_shading_normal = hit_from_front ? world_shading_normal : -world_shading_normal;
    world_shading_normal = fix_backfacing_shading_normal(-ray.direction, world_shading_normal, 0.002f);
    monte_carlo_payload.shading_normal = world_shading_normal;
    const TBN world_shading_tbn = TBN(world_shading_normal);

    // Store intersection point and wo in payload.
    // Compute world position by transforming the local intersection point to world space,
    // as that should be numerically more stable than picking a point on the ray, for large values of 't'.
    // Source: Ray Tracing Gems 1, chapter 6, A Fast and Robust Method for Avoiding Self-Intersection.
    float3 world_intersection_point = rtTransformPoint(RT_OBJECT_TO_WORLD, intersection_point);
    float3 world_wo = -ray.direction;
    float3 wo = world_shading_tbn * world_wo;

    float cos_theta = hit_from_front || material_parameter.is_thin_walled() ? wo.z : -wo.z;
    const auto material = MaterialCreator::create(material_parameter, texcoord, cos_theta);

    { // Next event estimation, sample the light sources directly.
        // Grab the RNG state before next even estimation to reset it afterwards which reduces the curse of dimensionality.
        // This is acceptable as the paths generated by BSDF sampling and next event estimation are independent.
        auto pre_NNE_rng_state = monte_carlo_payload.rng.get_state();

        monte_carlo_payload.light_sample = reestimated_light_samples(material, wo, world_shading_tbn);
        monte_carlo_payload.light_sample_origin = offset_ray_origin(
            world_intersection_point, monte_carlo_payload.light_sample.direction_to_light, world_geometric_normal);
        monte_carlo_payload.light_sample.radiance *= monte_carlo_payload.throughput;

        monte_carlo_payload.rng.set_state(pre_NNE_rng_state);
    }

    // BSDF sampling.
    BSDFSample bsdf_sample = material.sample(wo, bsdf_random_uvs);
    bool is_reflection = bsdf_sample.direction.z >= 0;
    monte_carlo_payload.direction = bsdf_sample.direction * world_shading_tbn;
    monte_carlo_payload.bsdf_MIS_PDF = MisPDF::from_PDF(bsdf_sample.PDF);
    if (is_PDF_valid(bsdf_sample.PDF))
        monte_carlo_payload.throughput *= bsdf_sample.reflectance * (abs(bsdf_sample.direction.z) / bsdf_sample.PDF); // f * ||cos(theta)|| / pdf
    else
        monte_carlo_payload.throughput = make_float3(0.0f);

    // Mirror the BSDF direction if the sampled BSDF direction points into the geometry.
    // This makes the integrator non-symmetric wrt wi and wo, but gives a decent
    // reflection vector in the case of a low-tesselated (cubical) sphere.
    // The reflection is handled independently of the ray throughput to avoid increasing
    // variance by changing cos_theta after sampling, but again, that makes it non-symmetrical.
    float cos_geometric_theta_i = dot(monte_carlo_payload.direction, world_geometric_normal);
    if (is_reflection ? cos_geometric_theta_i < 0.0f : cos_geometric_theta_i >= 0.0f)
        monte_carlo_payload.direction = reflect(monte_carlo_payload.direction, world_geometric_normal);

    // Offset the ray origin along the geometric normal in the sampled BSDF direction.
    monte_carlo_payload.position = offset_ray_origin(
        world_intersection_point, monte_carlo_payload.direction, world_geometric_normal);

    monte_carlo_payload.ray_min_t = 0.0f;
    monte_carlo_payload.bounces += 1u;

    // Disable multiple importance sampling if there are no valid light samples.
    if (!is_PDF_valid(monte_carlo_payload.light_sample.PDF))
        monte_carlo_payload.bsdf_MIS_PDF.disable_MIS();
}

//----------------------------------------------------------------------------
// Path tracing closest hit programs for different shading models.
//----------------------------------------------------------------------------

struct DefaultMaterialCreator {
    __inline_all__ static DefaultShading create(const Material& material_params, optix::float2 texcoord, float cos_theta_o) {
        float max_PDF_hint = monte_carlo_payload.bsdf_MIS_PDF.PDF() * g_camera_state.path_regularization_PDF_scale;
        return DefaultShading::initialize_with_max_PDF_hint(material_params, texcoord, cos_theta_o, max_PDF_hint);
    }
};

RT_PROGRAM void default_closest_hit() {
    path_tracing_closest_hit<DefaultMaterialCreator>();
}

struct DiffuseMaterialCreator {
    __inline_all__ static DiffuseShading create(const Material& material_params, optix::float2 texcoord, float cos_theta_o) {
        float4 tint_roughness = material_params.get_tint_roughness(texcoord);
        return DiffuseShading(make_float3(tint_roughness), tint_roughness.w);
    }
};

RT_PROGRAM void diffuse_closest_hit() {
    path_tracing_closest_hit<DiffuseMaterialCreator>();
}

struct TransmissiveMaterialCreator {
    __inline_all__ static TransmissiveShading create(const Material& material_params, optix::float2 texcoord, float cos_theta_o) {
        float max_PDF_hint = monte_carlo_payload.bsdf_MIS_PDF.PDF() * g_camera_state.path_regularization_PDF_scale;
        return TransmissiveShading::initialize_with_max_PDF_hint(material_params, texcoord, cos_theta_o, max_PDF_hint);
    }
};

RT_PROGRAM void transmissive_closest_hit() {
    path_tracing_closest_hit<TransmissiveMaterialCreator>();
}

//----------------------------------------------------------------------------
// Any hit program for monte carlo shadow rays.
//----------------------------------------------------------------------------

rtDeclareVariable(ShadowPayload, shadow_payload, rtPayload, );

RT_PROGRAM void shadow_any_hit() {
    float coverage = g_materials[model_state.material_index].get_coverage(texcoord);
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
    Light light = g_scene.light_buffer[primitive_index];
    float3 light_radiance = make_float3(0, 0, 0);
    if (light.get_type() == Light::Sphere)
        light_radiance = LightSources::evaluate_intersection(light.sphere, ray.origin, ray.direction,
                                                             monte_carlo_payload.bsdf_MIS_PDF);
    else if (light.get_type() == Light::Spot)
        light_radiance = LightSources::evaluate_intersection(light.spot, ray.origin, ray.direction,
                                                             monte_carlo_payload.bsdf_MIS_PDF);

    monte_carlo_payload.radiance += monte_carlo_payload.throughput * light_radiance;
    monte_carlo_payload.throughput = make_float3(0.0f);
    monte_carlo_payload.position = ray.direction * t_hit + ray.origin;
    monte_carlo_payload.shading_normal = shading_normal;
    monte_carlo_payload.primitive_id = PrimitiveID::make(InstanceID::analytical_light_sources(), primitive_index);
}