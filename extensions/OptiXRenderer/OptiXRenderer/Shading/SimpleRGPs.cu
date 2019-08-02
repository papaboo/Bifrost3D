// Simple OptiX ray generation programs, such as path tracing, normal and albedo visualization
// ------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#include <OptiXRenderer/Shading/ShadingModels/DefaultShading.h>
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

const int pmj_period = 9;
__constant__ float2 pmj_offsets[81] = {
    { 0.93f, 0.43f }, { 0.48f, 0.38f }, { 0.96f, 0.77f }, { 0.22f, 0.02f }, { 0.23f, 0.14f }, { 0.95f, 0.65f }, { 0.67f, 0.07f }, { 0.06f, 0.56f }, { 0.99f, 0.99f },
    { 0.51f, 0.60f }, { 0.27f, 0.47f }, { 0.88f, 0.98f }, { 0.72f, 0.52f }, { 0.52f, 0.72f }, { 0.20f, 0.79f }, { 0.53f, 0.83f }, { 0.16f, 0.46f }, { 0.09f, 0.78f },
    { 0.78f, 0.09f }, { 0.33f, 0.04f }, { 0.36f, 0.26f }, { 0.80f, 0.31f }, { 0.14f, 0.23f }, { 0.69f, 0.30f }, { 0.57f, 0.17f }, { 0.12f, 0.12f }, { 0.58f, 0.28f },
    { 0.84f, 0.64f }, { 0.32f, 0.91f }, { 0.31f, 0.80f }, { 0.85f, 0.75f }, { 0.35f, 0.15f }, { 0.73f, 0.63f }, { 0.07f, 0.67f }, { 0.04f, 0.33f }, { 0.28f, 0.58f },
    { 0.64f, 0.84f }, { 0.70f, 0.41f }, { 0.15f, 0.35f }, { 0.68f, 0.19f }, { 0.89f, 0.10f }, { 0.30f, 0.69f }, { 0.77f, 0.96f }, { 0.11f, 0.01f }, { 0.10f, 0.89f },
    { 0.91f, 0.32f }, { 0.75f, 0.85f }, { 0.25f, 0.25f }, { 0.83f, 0.53f }, { 0.56f, 0.06f }, { 0.17f, 0.57f }, { 0.60f, 0.51f }, { 0.42f, 0.81f }, { 0.26f, 0.36f },
    { 0.37f, 0.37f }, { 0.21f, 0.90f }, { 0.19f, 0.68f }, { 0.41f, 0.70f }, { 0.63f, 0.73f }, { 0.90f, 0.21f }, { 0.47f, 0.27f }, { 0.86f, 0.86f }, { 0.05f, 0.44f },
    { 0.59f, 0.40f }, { 0.74f, 0.74f }, { 0.02f, 0.22f }, { 0.40f, 0.59f }, { 0.46f, 0.16f }, { 0.01f, 0.11f }, { 0.94f, 0.54f }, { 0.43f, 0.93f }, { 0.00f, 0.00f },
    { 0.49f, 0.49f }, { 0.44f, 0.05f }, { 0.98f, 0.88f }, { 0.79f, 0.20f }, { 0.62f, 0.62f }, { 0.81f, 0.42f }, { 0.38f, 0.48f }, { 0.65f, 0.95f }, { 0.54f, 0.94f },
};

__inline_dev__ MonteCarloPayload initialize_monte_carlo_payload(int x, int y, int image_width, int image_height,
    int accumulation_count, optix::float3 camera_position, const optix::Matrix4x4& inverted_view_projection_matrix) {
    using namespace optix;

    MonteCarloPayload payload;
    payload.radiance = make_float3(0.0f);

    RNG::LinearCongruential pmj_offset_rng; pmj_offset_rng.seed(__brev(RNG::teschner_hash(x, y)));
    float2 offset = pmj_offsets[(x % pmj_period) + (y % pmj_period) * pmj_period];
    offset += make_float2(-1 / 18.0f) + pmj_offset_rng.sample2f() / pmj_period;
    payload.pmj_rng_state = PMJSamplerState::make(accumulation_count, offset.x, offset.y);

    /*
    const int period = 5;
    const float pixel_span = 1.0f / period;
    int y_stratum = (x % period);
    y_stratum ^= y_stratum >> 1;
    float pmj_y_offset = y_stratum * pixel_span + pixel_span * pmj_offset_rng.sample1f();
    int x_stratum = ((x + y) % period);
    x_stratum ^= x_stratum >> 1;
    float pmj_x_offset = x_stratum * pixel_span + pixel_span * pmj_offset_rng.sample1f();
    payload.pmj_rng_state = PMJSampler::make(accumulation_count, pmj_x_offset, pmj_y_offset);
    */

    payload.throughput = make_float3(1.0f);
    payload.bounces = 0;
    payload.bsdf_MIS_PDF = 0.0f;
    payload.shading_normal = make_float3(0.0f);
    payload.material_index = 0;

    // Generate rays.
    RNG::LinearCongruential rng; rng.seed(__brev(RNG::teschner_hash(x, y, accumulation_count)));
    float2 screen_pos = make_float2(x, y) + (accumulation_count == 0 ? make_float2(0.5f) : rng.sample2f());
    float2 viewport_pos = make_float2(screen_pos.x / float(image_width), screen_pos.y / float(image_height));
    payload.position = camera_position;
    payload.direction = project_ray_direction(viewport_pos, payload.position, inverted_view_projection_matrix);
    return payload;
}

template <typename Evaluator>
__inline_dev__ void accumulate(Evaluator evaluator) {
    const CameraStateGPU& camera_state = g_camera_state;
    const int accumulation_count = camera_state.accumulations;
    size_t2 screen_size = camera_state.accumulation_buffer.size();

    MonteCarloPayload payload = initialize_monte_carlo_payload(g_launch_index.x, g_launch_index.y,
        screen_size.x, screen_size.y, accumulation_count,
        camera_state.camera_position, camera_state.inverted_view_projection_matrix);

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
RT_PROGRAM void path_tracing_RPG() {

    accumulate([](MonteCarloPayload payload) -> float3 {
        do {
            Ray ray(payload.position, payload.direction, RayTypes::MonteCarlo, g_scene.ray_epsilon);
            rtTrace(g_scene_root, ray, payload);
        } while (payload.bounces < g_camera_state.max_bounce_count && !is_black(payload.throughput));

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
    float3 normal = { 0, 0, 0 };

    accumulate([&](MonteCarloPayload payload) -> float3 {
        bool properties_accumulated = false;
        do {
            float3 last_ray_direction = payload.direction;
            Ray ray(payload.position, payload.direction, RayTypes::MonteCarlo, g_scene.ray_epsilon);
            rtTrace(g_scene_root, ray, payload);

            bool terminate_ray = !(payload.bounces < g_camera_state.max_bounce_count && !is_black(payload.throughput));

            // Accumulate surface properties of the first non-specular surface hit.
            // If no non-specular hits are found or the BSDF PDF is zero due to a bad sample being drawn, 
            // then use the last hit to ensure that some feature data is output.
            if (!properties_accumulated && (payload.bsdf_MIS_PDF != 0.0f || terminate_ray)) {
                properties_accumulated = true;

                normal = payload.shading_normal;

                if (payload.material_index > 0) {
                    using namespace Shading::ShadingModels;
                    const float abs_cos_theta = abs(dot(last_ray_direction, normal));
                    const auto material_parameters = g_materials[payload.material_index];
                    const auto material = DefaultShading(material_parameters, abs_cos_theta, payload.texcoord);
                    albedo = material.rho(abs_cos_theta);
                }
            }

        } while (payload.bounces < g_camera_state.max_bounce_count && !is_black(payload.throughput));

        return payload.radiance;
    });

    // Accumulate normals
    // TODO Transform to camera space and potentialy flip y/z.
    if (g_AI_denoiser_state.normals_buffer != 0) {
        auto normals_buffer = g_AI_denoiser_state.normals_buffer;
        float3 prev_normals = make_float3(normals_buffer[g_launch_index]);
        const float magnitude = g_camera_state.accumulations ? normals_buffer[g_launch_index].w : 0.0f;
        prev_normals *= magnitude;

        // OptiX expects a normal in view space with red going from left to right, green as up and blue along the depth, with normals pointing towards the camera as 100% blue.
        float3 denoiser_normal = g_camera_state.world_to_view_rotation * normal;
        denoiser_normal.z = -denoiser_normal.z;

        float3 accumulated_normals = prev_normals + denoiser_normal;
        float new_length = length(accumulated_normals);
        normals_buffer[g_launch_index] = make_float4(accumulated_normals / new_length, new_length);
    }

    // Accumulate albedo
    if (g_AI_denoiser_state.albedo_buffer != 0) {
        auto albedo_buffer = g_AI_denoiser_state.albedo_buffer;
        const float3 prev_albedo = make_float3(albedo_buffer[g_launch_index]);
        const float accumulation_count = g_camera_state.accumulations ? (albedo_buffer[g_launch_index].w + 1.0f) : 1.0f;
        const float3 accumulated_albedo = lerp(prev_albedo, albedo, 1.0f / accumulation_count);
        albedo_buffer[g_launch_index] = make_float4(accumulated_albedo, accumulation_count);
    }

    // Output radiance.
#ifdef DOUBLE_PRECISION_ACCUMULATION_BUFFER
    double4 p = g_camera_state.accumulation_buffer[g_launch_index];
    float4 noisy_pixel = make_float4(p.x, p.y, p.z, 1.0f);
#else
    float4 noisy_pixel = g_camera_state.accumulation_buffer[g_launch_index];
#endif

    g_AI_denoiser_state.noisy_pixels_buffer[g_launch_index] = gamma_correct(noisy_pixel);
}

RT_PROGRAM void copy_to_output() {
    float4 pixel = reverse_gamma_correct(g_AI_denoiser_state.denoised_pixels_buffer[g_launch_index]);

    if (g_AI_denoiser_state.flags & unsigned int(AIDenoiserFlag::VisualizeNoise)) {
#ifdef DOUBLE_PRECISION_ACCUMULATION_BUFFER
        double4 p = g_camera_state.accumulation_buffer[g_launch_index];
        pixel = make_float4(p.x, p.y, p.z, 1.0f);
#else
        pixel = g_camera_state.accumulation_buffer[g_launch_index];
#endif
    } else if (g_AI_denoiser_state.flags & int(AIDenoiserFlag::VisualizeAlbedo))
        pixel = g_AI_denoiser_state.albedo_buffer[g_launch_index];
    else if (g_AI_denoiser_state.flags & int(AIDenoiserFlag::VisualizeNormals))
        pixel = g_AI_denoiser_state.normals_buffer[g_launch_index] * 0.5f + 0.5f;

    g_camera_state.output_buffer[g_launch_index] = float_to_half(pixel);
}

} // NS AIDenoiser

//-------------------------------------------------------------------------------------------------
// Ray generation program for visualizing estimated and sampled albedo.
//-------------------------------------------------------------------------------------------------

RT_PROGRAM void albedo_RPG() {

    accumulate([](MonteCarloPayload payload) -> float3 {
        float3 last_ray_direction = payload.direction;
        do {
            last_ray_direction = payload.direction;
            Ray ray(payload.position, payload.direction, RayTypes::MonteCarlo, g_scene.ray_epsilon);
            rtTrace(g_scene_root, ray, payload);
        } while (payload.material_index == 0 && !is_black(payload.throughput));

        size_t2 screen_size = g_camera_state.accumulation_buffer.size();
        bool valid_material = payload.material_index != 0;
        if (g_launch_index.x < screen_size.x / 2 && valid_material) {
            using namespace Shading::ShadingModels;
            const Material& material_parameter = g_materials[payload.material_index];
            const float abs_cos_theta = abs(dot(last_ray_direction, payload.shading_normal));
            const DefaultShading material = DefaultShading(material_parameter, abs_cos_theta, payload.texcoord);
            return material.rho(abs_cos_theta);
        } else
            return payload.throughput;
    });
}

//-------------------------------------------------------------------------------------------------
// Miss program for monte carlo rays.
//-------------------------------------------------------------------------------------------------

rtDeclareVariable(Ray, ray, rtCurrentRay, );
rtDeclareVariable(MonteCarloPayload, monte_carlo_payload, rtPayload, );

RT_PROGRAM void miss() {
    float3 environment_radiance = g_scene.environment_tint;

    unsigned int environment_map_ID = g_scene.environment_light.environment_map_ID;
    if (environment_map_ID) {
        bool next_event_estimated = monte_carlo_payload.bounces != 0; // Was next event estimated at previous intersection.
        environment_radiance *= LightSources::evaluate_intersection(g_scene.environment_light, ray.origin, ray.direction, 
                                                                    monte_carlo_payload.bsdf_MIS_PDF, next_event_estimated);
    }

    monte_carlo_payload.radiance += monte_carlo_payload.throughput * environment_radiance;
    monte_carlo_payload.throughput = make_float3(0.0f);
    monte_carlo_payload.shading_normal = -ray.direction;
}

//-------------------------------------------------------------------------------------------------
// Exception program.
//-------------------------------------------------------------------------------------------------
RT_PROGRAM void exceptions() {
    rtPrintExceptionDetails();
}
