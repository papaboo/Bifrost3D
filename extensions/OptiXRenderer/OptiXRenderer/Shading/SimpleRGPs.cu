// Simple OptiX ray generation programs, such as path tracing, normal and albedo visualization
// ------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#include <OptiXRenderer/Shading/ShadingModels/DefaultShading.h>
#include <OptiXRenderer/Shading/ShadingModels/DiffuseShading.h>
#include <OptiXRenderer/Shading/ShadingModels/TransmissiveShading.h>
#include <OptiXRenderer/Shading/LightSources/LightImpl.h>
#include <OptiXRenderer/Types.h>
#include <OptiXRenderer/Utils.h>

#include <optix.h>
#include <optixu/optixu_matrix_namespace.h>

using namespace OptiXRenderer;
using namespace optix;

rtDeclareVariable(uint2, g_launch_index, rtLaunchIndex, );

rtDeclareVariable(CameraStateGPU, g_camera_state, , );
rtBuffer<Material, 1> g_materials;

// ------------------------------------------------------------------------------------------------
// Ray generation program utility functions.
// ------------------------------------------------------------------------------------------------

// Scene variables
rtDeclareVariable(rtObject, g_scene_root, , );
rtDeclareVariable(SceneStateGPU, g_scene, , );

__inline_dev__ float4 half_to_float(ushort4 xyzw) {
    return make_float4(__half2float(xyzw.x), __half2float(xyzw.y), __half2float(xyzw.z), __half2float(xyzw.w));
}

__inline_dev__ ushort4 float_to_half(float4 xyzw) {
    return make_ushort4(((__half_raw)__float2half_rn(xyzw.x)).x, ((__half_raw)__float2half_rn(xyzw.y)).x,
        ((__half_raw)__float2half_rn(xyzw.z)).x, ((__half_raw)__float2half_rn(xyzw.w)).x);
}

__inline_dev__ void fill_ray_info(float2 viewport_pos, const CameraStateGPU& camera_state_GPU,
                                  float3& origin, float3& direction) {

    float4 NDC_near_pos = make_float4(viewport_pos.x * 2.0f - 1.0f, viewport_pos.y * 2.0f - 1.0f, -1.0f, 1.0f);
    float4 scaled_near_world_pos = camera_state_GPU.inverse_view_projection_matrix * NDC_near_pos;
    origin = make_float3(scaled_near_world_pos) / scaled_near_world_pos.w;

    float4 NDC_far_pos = make_float4(NDC_near_pos.x, NDC_near_pos.y, 1.0f, 1.0f);
    float4 scaled_near_view_pos = camera_state_GPU.inverse_projection_matrix * NDC_far_pos;
    direction = normalize(camera_state_GPU.view_to_world_rotation * make_float3(scaled_near_view_pos));
}

__inline_dev__ MonteCarloPayload initialize_monte_carlo_payload(int x, int y, int image_width, int image_height,
    int accumulation_count, const CameraStateGPU& camera_state_GPU) {

    MonteCarloPayload payload = {};
    payload.throughput = make_float3(1.0f);
    payload.light_sample = LightSample::none();
    payload.bsdf_PDF = PDF::delta_dirac();

#if LCG_RNG
    payload.rng.set_state(__brev(RNG::teschner_hash(x, y) + accumulation_count));
#elif PRACTICAL_SOBOL_RNG
    payload.rng = RNG::PracticalScrambledSobol(x, y, accumulation_count);
#endif

    // Generate rays.
    RNG::LinearCongruential rng; rng.set_state(__brev(RNG::teschner_hash(x, y, accumulation_count)));
    float2 screen_pos = make_float2(x, y) + (accumulation_count == 0 ? make_float2(0.5f) : rng.sample2f());
    float2 viewport_pos = make_float2(screen_pos.x / float(image_width), screen_pos.y / float(image_height));
    fill_ray_info(viewport_pos, camera_state_GPU, payload.position, payload.direction);
    return payload;
}

template <typename Evaluator>
__inline_dev__ void accumulate(Evaluator evaluator) {
    const CameraStateGPU& camera_state = g_camera_state;
    const int accumulation_count = camera_state.accumulations;
    size_t2 screen_size = camera_state.accumulation_buffer.size();

    MonteCarloPayload payload = initialize_monte_carlo_payload(g_launch_index.x, g_launch_index.y,
        screen_size.x, screen_size.y, accumulation_count, camera_state);

    float3 radiance = evaluator(payload);

    auto accumulation_buffer = camera_state.accumulation_buffer;
#ifdef DOUBLE_PRECISION_ACCUMULATION_BUFFER
    double3 accumulated_radiance_d;
    if (accumulation_count != 0) {
        double3 prev_radiance = make_double3(accumulation_buffer[g_launch_index].x, accumulation_buffer[g_launch_index].y, accumulation_buffer[g_launch_index].z);
        accumulated_radiance_d = lerp_double(prev_radiance, make_double3(radiance.x, radiance.y, radiance.z), 1.0 / (accumulation_count + 1.0));
    } else
        accumulated_radiance_d = make_double3(radiance.x, radiance.y, radiance.z);
    accumulation_buffer[g_launch_index] = make_double4(accumulated_radiance_d.x, accumulated_radiance_d.y, accumulated_radiance_d.z, 1.0f);
    float3 accumulated_radiance = make_float3(accumulated_radiance_d.x, accumulated_radiance_d.y, accumulated_radiance_d.z);
#else
    float3 accumulated_radiance;
    if (accumulation_count != 0) {
        float3 prev_radiance = make_float3(accumulation_buffer[g_launch_index]);
        accumulated_radiance = lerp(prev_radiance, radiance, 1.0f / (accumulation_count + 1.0f));
    }
    else
        accumulated_radiance = radiance;
    accumulation_buffer[g_launch_index] = make_float4(accumulated_radiance, 1.0f);
#endif

    camera_state.output_buffer[g_launch_index] = float_to_half(make_float4(accumulated_radiance, 1.0f));
}

//-------------------------------------------------------------------------------------------------
// Path tracing ray generation program.
//-------------------------------------------------------------------------------------------------
__inline_dev__ void path_trace_single_bounce(MonteCarloPayload& payload) {
    payload.material_index = 0;
    Ray ray(payload.position, payload.direction, RayTypes::MonteCarlo, payload.ray_min_t);
    rtTrace(g_scene_root, ray, payload, RT_VISIBILITY_ALL, RT_RAY_FLAG_DISABLE_ANYHIT);

    // Trace shadow ray for light sample.
    const LightSample& light_sample = payload.light_sample;
    if (light_sample.radiance.x > 0 || light_sample.radiance.y > 0 || light_sample.radiance.z > 0) {
        ShadowPayload shadow_payload = { light_sample.radiance };
        Ray shadow_ray(payload.light_sample_origin, light_sample.direction_to_light, RayTypes::Shadow, 0.0f, light_sample.distance);
        rtTrace(g_scene_root, shadow_ray, shadow_payload, RT_VISIBILITY_ALL, RT_RAY_FLAG_DISABLE_CLOSESTHIT);

        payload.radiance += shadow_payload.radiance;
    }

    // Reset light sample
    payload.light_sample = LightSample::none();
}

RT_PROGRAM void path_tracing_RPG() {

    accumulate([](MonteCarloPayload payload) -> float3 {
        do {
            path_trace_single_bounce(payload);
        } while (payload.bounces <= g_camera_state.max_bounce_count && !is_black(payload.throughput));

        return payload.radiance;
    });
}

//-------------------------------------------------------------------------------------------------
// Denoise ray generation program.
//-------------------------------------------------------------------------------------------------
namespace AIDenoiser {

rtDeclareVariable(AIDenoiserStateGPU, g_AI_denoiser_state, , );

RT_PROGRAM void path_tracing_RPG() {
    float3 albedo = { 0, 0, 0 };

    accumulate([&](MonteCarloPayload payload) -> float3 {
        bool properties_accumulated = false;
        do {
            float3 last_ray_direction = payload.direction;
            path_trace_single_bounce(payload);

            // Accumulate surface properties of the first non-specular surface hit.
            // If no non-specular hits are found or the BSDF PDF is zero due to a bad sample being drawn, 
            // then use the last hit to ensure that some feature data is output.
            float3 shading_normal = payload.shading_normal;
            bool non_zero_normal = shading_normal.x != 0 || shading_normal.y != 0 || shading_normal.z != 0;
            if (!properties_accumulated && non_zero_normal) {
                properties_accumulated = true;

                if (payload.material_index > 0) {
                    using namespace Shading::ShadingModels;
                    const float abs_cos_theta = abs(dot(last_ray_direction, shading_normal));
                    const auto material_parameters = g_materials[payload.material_index];

                    // Use DefaultMaterial for all materials.
                    // This is because default material's rho contains the view dependent specular highlight as well,
                    // which can give an edge in the image between surfaces with different normals,
                    // so the albedo image captures both the surface albedo and surface normal effects.
                    float4 tint_and_roughness_scale = unorm8_to_float(payload.tint_and_roughness_scale);
                    const auto material = DefaultShading(material_parameters, payload.texcoord, tint_and_roughness_scale, abs_cos_theta);
                    albedo = material.rho(abs_cos_theta);
                } else
                    // Must have hit a light source. Accumulate it's contribution, but normalize to [0, 1] range otherwise AI denoising blows up.
                    albedo = payload.radiance / (1 + payload.radiance);
            }

        } while (payload.bounces <= g_camera_state.max_bounce_count && !is_black(payload.throughput));

        return payload.radiance;
    });

    // Accumulate albedo
    auto albedo_buffer = g_AI_denoiser_state.albedo_buffer;
    const float3 prev_albedo = make_float3(albedo_buffer[g_launch_index]);
    const float3 accumulated_albedo = lerp(prev_albedo, albedo, 1.0f / (g_camera_state.accumulations + 1));
    albedo_buffer[g_launch_index] = make_float4(accumulated_albedo, 1.0f);

    // Output radiance.
#ifdef DOUBLE_PRECISION_ACCUMULATION_BUFFER
    double4 p = g_camera_state.accumulation_buffer[g_launch_index];
    float4 noisy_pixel = make_float4(p.x, p.y, p.z, 1.0f);
#else
    float4 noisy_pixel = g_camera_state.accumulation_buffer[g_launch_index];
#endif

    g_AI_denoiser_state.noisy_pixels_buffer[g_launch_index] = noisy_pixel;
}

RT_PROGRAM void copy_to_output() {
    float4 pixel = g_AI_denoiser_state.denoised_pixels_buffer[g_launch_index];

    if (g_AI_denoiser_state.flags & unsigned int(AIDenoiserFlag::VisualizeNoise)) {
#ifdef DOUBLE_PRECISION_ACCUMULATION_BUFFER
        double4 p = g_camera_state.accumulation_buffer[g_launch_index];
        pixel = make_float4(p.x, p.y, p.z, 1.0f);
#else
        pixel = g_camera_state.accumulation_buffer[g_launch_index];
#endif
    } else if (g_AI_denoiser_state.flags & int(AIDenoiserFlag::VisualizeAlbedo))
        pixel = g_AI_denoiser_state.albedo_buffer[g_launch_index];

    g_camera_state.output_buffer[g_launch_index] = float_to_half(pixel);
}

} // NS AIDenoiser

  //-------------------------------------------------------------------------------------------------
  // Ray generation program for visualizing aggregated depth.
  //-------------------------------------------------------------------------------------------------

RT_PROGRAM void depth_RPG() {

    accumulate([=](MonteCarloPayload payload) -> float3 {
        float depth = 0;
        do {
            float3 last_position = payload.position;
            Ray ray(payload.position, payload.direction, RayTypes::MonteCarlo, payload.ray_min_t);
            rtTrace(g_scene_root, ray, payload, RT_VISIBILITY_ALL, RT_RAY_FLAG_DISABLE_ANYHIT);
            depth += length(last_position - payload.position);
        } while (payload.material_index == 0 && !is_black(payload.throughput));

        return make_float3(depth, depth, depth);
    });

#ifdef DOUBLE_PRECISION_ACCUMULATION_BUFFER
    float depth = (float)g_camera_state.accumulation_buffer[g_launch_index].x;
#else
    float depth = g_camera_state.accumulation_buffer[g_launch_index].x;
#endif

    // Compute the max depth as the length from the center of the near plane to the center of the far plane.
    float4 NDC_near_pos = { 0.0f, 0.0f, -1.0f, 1.0f };
    float4 scaled_near_view_pos = g_camera_state.inverse_projection_matrix * NDC_near_pos;
    float near_view_Z = scaled_near_view_pos.z / scaled_near_view_pos.w;

    float4 NDC_far_pos = { 0.0f, 0.0f, 1.0f, 1.0f };
    float4 scaled_far_view_pos = g_camera_state.inverse_projection_matrix * NDC_far_pos;
    float far_view_Z = scaled_far_view_pos.z / scaled_far_view_pos.w;
    float max_depth = far_view_Z - near_view_Z;

    float d = depth / max_depth;
    g_camera_state.output_buffer[g_launch_index] = float_to_half(make_float4(d, d, d, 1.0f));
}

//-------------------------------------------------------------------------------------------------
// Ray generation programs for visualizing aggregated material properties.
//-------------------------------------------------------------------------------------------------

template <typename IntersectionProcessor>
__inline_dev__ void process_material_intersection(IntersectionProcessor process_intersection) {
    accumulate([process_intersection](MonteCarloPayload payload) -> float3 {
        float3 last_ray_direction = payload.direction;
        do {
            last_ray_direction = payload.direction;
            Ray ray(payload.position, payload.direction, RayTypes::MonteCarlo, payload.ray_min_t);
            rtTrace(g_scene_root, ray, payload, RT_VISIBILITY_ALL, RT_RAY_FLAG_DISABLE_ANYHIT);
        } while (payload.material_index == 0 && !is_black(payload.throughput));

        if (payload.material_index == 0)
            return make_float3(0, 0, 0);

        return process_intersection(payload, last_ray_direction);
    });
}

RT_PROGRAM void albedo_RPG() {
    process_material_intersection([](const MonteCarloPayload& payload, float3 last_ray_direction) -> float3 {
        float abs_cos_theta = abs(dot(last_ray_direction, payload.shading_normal));
        const auto& material_parameters = g_materials[payload.material_index];
        float4 tint_and_roughness_scale = unorm8_to_float(payload.tint_and_roughness_scale);
        if (material_parameters.shading_model == Material::ShadingModel::Default) {
            auto material = Shading::ShadingModels::DefaultShading(material_parameters, payload.texcoord, tint_and_roughness_scale, abs_cos_theta);
            return material.rho(abs_cos_theta);
        } else if (material_parameters.shading_model == Material::ShadingModel::Diffuse) {
            float4 tint_roughness = material_parameters.get_tint_roughness(payload.texcoord) * tint_and_roughness_scale;
            return Shading::ShadingModels::DiffuseShading(make_float3(tint_roughness), tint_roughness.w).rho(abs_cos_theta);
        } else if (material_parameters.shading_model == Material::ShadingModel::Transmissive) {
            auto material = Shading::ShadingModels::TransmissiveShading(material_parameters, payload.texcoord, tint_and_roughness_scale, abs_cos_theta);
            return material.rho(abs_cos_theta);
        } else
            return { 0, 0, 0 };
    });
}

RT_PROGRAM void tint_RPG() {
    process_material_intersection([](const MonteCarloPayload& payload, float3 last_ray_direction) -> float3 {
        float3 tint_scale = make_float3(unorm8_to_float(payload.tint_and_roughness_scale));
        const auto& material_parameters = g_materials[payload.material_index];
        return make_float3(material_parameters.get_tint_roughness(payload.texcoord)) * tint_scale;
    });
}

RT_PROGRAM void roughness_RPG() {
    process_material_intersection([](const MonteCarloPayload& payload, float3 last_ray_direction) -> float3 {
        float roughness_scale = unorm8_to_float(payload.tint_and_roughness_scale).w;
        const auto& material_parameters = g_materials[payload.material_index];
        float roughness = material_parameters.get_tint_roughness(payload.texcoord).w * roughness_scale;
        return { roughness, roughness, roughness };
    });
}

RT_PROGRAM void shading_normal_RPG() {
    process_material_intersection([](const MonteCarloPayload& payload, float3 last_ray_direction) -> float3 {
        return payload.shading_normal * 0.5f + 0.5f;
    });
}

// Convert a primitive ID to a color. Primitive ID's that are close to each other should map to very different colors.
// This is achieved by interpreting the primitive ID as a 30bit reversed morton code. Reversing the bits give us high variance
// in the most significant bits for neighbouring ID's and decoding the bitmask as a morton code
// splits the most significant bits into the different color channels.
// To have different colors across different geometry instances, we additionally xor the primitive ID with the instance ID.
__inline_dev__ float3 primitive_id_to_color(PrimitiveID primitive_id) {
    unsigned int instance_encoding = primitive_id.instance_id.id & 0x3FFFFFF;
    unsigned int primitive_encoding = reverse_bits(primitive_id.primitive_id + 1) >> 2;
    uint3 color = morton_decode_3D(instance_encoding ^ primitive_encoding);
    return make_float3(color) / 1023.0f;
}

RT_PROGRAM void primitive_id_RPG() {
    process_material_intersection([](const MonteCarloPayload& payload, float3 last_ray_direction) -> float3 {
        return primitive_id_to_color(payload.primitive_id);
    });
}

//-------------------------------------------------------------------------------------------------
// Miss program for monte carlo rays.
//-------------------------------------------------------------------------------------------------

rtDeclareVariable(Ray, ray, rtCurrentRay, );
rtDeclareVariable(MonteCarloPayload, monte_carlo_payload, rtPayload, );

RT_PROGRAM void miss() {
    float3 environment_radiance = g_scene.environment_light.get_tint();

    unsigned int environment_map_ID = g_scene.environment_light.environment_map_ID;
    if (environment_map_ID)
        environment_radiance = LightSources::evaluate_intersection(g_scene.environment_light, ray.origin, ray.direction, 
                                                                   monte_carlo_payload.bsdf_PDF);

    monte_carlo_payload.radiance += monte_carlo_payload.throughput * environment_radiance;
    monte_carlo_payload.throughput = make_float3(0.0f);
    monte_carlo_payload.position = 1e30f * monte_carlo_payload.direction;
    monte_carlo_payload.shading_normal = -ray.direction;
    monte_carlo_payload.primitive_id = PrimitiveID::make(InstanceID::analytical_light_sources(), 0xFFFFFFFF);
}

//-------------------------------------------------------------------------------------------------
// Exception program.
//-------------------------------------------------------------------------------------------------
RT_PROGRAM void exceptions() {
    rtPrintExceptionDetails();
}
