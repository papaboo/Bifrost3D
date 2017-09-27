// Simple OptiX path ray generation programs, such as path tracing, normal and albedo visualization
// ------------------------------------------------------------------------------------------------
// Copyright (C) 2016, Cogwheel. See AUTHORS.txt for authors
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

rtDeclareVariable(int, g_accumulations, , );
rtBuffer<ushort4, 2>  g_output_buffer;
#ifdef DOUBLE_PRECISION_ACCUMULATION_BUFFER
rtBuffer<double4, 2>  g_accumulation_buffer;
#else
rtBuffer<float4, 2>  g_accumulation_buffer;
#endif

rtDeclareVariable(float4, g_camera_position, , );
rtDeclareVariable(Matrix4x4, g_inverted_view_projection_matrix, , );

// Scene variables
rtDeclareVariable(rtObject, g_scene_root, , );
rtDeclareVariable(float, g_scene_epsilon, , );
rtDeclareVariable(int, g_max_bounce_count, , );

template <typename Evaluator>
__inline_dev__ void accumulate(Evaluator evaluator) {
    if (g_accumulations == 0)
#ifdef DOUBLE_PRECISION_ACCUMULATION_BUFFER
        g_accumulation_buffer[g_launch_index] = make_double4(0.0, 0.0, 0.0, 0.0);
#else
        g_accumulation_buffer[g_launch_index] = make_float4(0.0f);
#endif

    MonteCarloPayload payload = initialize_monte_carlo_payload(g_launch_index.x, g_launch_index.y,
        g_accumulation_buffer.size().x, g_accumulation_buffer.size().y, g_accumulations,
        make_float3(g_camera_position), g_inverted_view_projection_matrix);

    float3 color = evaluator(payload);

#ifdef DOUBLE_PRECISION_ACCUMULATION_BUFFER
    double3 prev_radiance = make_double3(g_accumulation_buffer[g_launch_index].x, g_accumulation_buffer[g_launch_index].y, g_accumulation_buffer[g_launch_index].z);
    double3 accumulated_radiance_d = lerp_double(prev_radiance, make_double3(color.x, color.y, color.z), 1.0 / (g_accumulations + 1.0));
    g_accumulation_buffer[g_launch_index] = make_double4(accumulated_radiance_d.x, accumulated_radiance_d.y, accumulated_radiance_d.z, 1.0f);
    float3 accumulated_radiance = make_float3(accumulated_radiance_d.x, accumulated_radiance_d.y, accumulated_radiance_d.z);
#else
    float3 prev_radiance = make_float3(g_accumulation_buffer[g_launch_index]);
    float3 accumulated_radiance = lerp(prev_radiance, color, 1.0f / (g_accumulations + 1.0f));
    g_accumulation_buffer[g_launch_index] = make_float4(accumulated_radiance, 1.0f);
#endif

    g_output_buffer[g_launch_index] = float_to_half(make_float4(accumulated_radiance, 1.0f));
}

//-------------------------------------------------------------------------------------------------
// Path tracing ray generation program.
//-------------------------------------------------------------------------------------------------
RT_PROGRAM void path_tracing_RPG() {

    accumulate([](MonteCarloPayload payload) -> float3 {
        do {
            Ray ray(payload.position, payload.direction, RayTypes::MonteCarlo, g_scene_epsilon);
            rtTrace(g_scene_root, ray, payload);
        } while (payload.bounces < g_max_bounce_count && !is_black(payload.throughput));

        return payload.radiance;
    });
}

//-------------------------------------------------------------------------------------------------
// Ray generation program for visualizing normals.
//-------------------------------------------------------------------------------------------------
RT_PROGRAM void normals_RPG() {

    accumulate([](MonteCarloPayload payload) -> float3 {
        // Iterate until a material is sampled.
        float3 last_ray_direction = payload.direction;
        do {
            last_ray_direction = payload.direction;
            Ray ray(payload.position, payload.direction, RayTypes::MonteCarlo, g_scene_epsilon);
            rtTrace(g_scene_root, ray, payload);
        } while (payload.bsdf_MIS_PDF == 0.0f && !is_black(payload.throughput));

        float D_dot_N = -dot(last_ray_direction, payload.shading_normal);
        if (D_dot_N < 0.0f)
            return make_float3(0.25f - 0.75f * D_dot_N, 0.0f, 0.0f);
        else
            return make_float3(0.0f, 0.25f + 0.75f * D_dot_N, 0.0f);
    });
}

//-------------------------------------------------------------------------------------------------
// Ray generation program for visualizing estimated and sampled albedo.
//-------------------------------------------------------------------------------------------------
rtBuffer<Material, 1> g_materials;

RT_PROGRAM void albedo_RPG() {

    accumulate([](MonteCarloPayload payload) -> float3 {
        float3 last_ray_direction = payload.direction;
        do {
            last_ray_direction = payload.direction;
            Ray ray(payload.position, payload.direction, RayTypes::MonteCarlo, g_scene_epsilon);
            rtTrace(g_scene_root, ray, payload);
        } while (payload.material_index == 0 && !is_black(payload.throughput));

        bool valid_material = payload.material_index != 0;
        if (g_launch_index.x < g_accumulation_buffer.size().x / 2 && valid_material) {
            using namespace Shading::ShadingModels;
            const Material& material_parameter = g_materials[payload.material_index];
            const DefaultShading material = DefaultShading(material_parameter, payload.texcoord);
            return material.IBL(last_ray_direction, payload.shading_normal, 0); // TODO Use a white IBL.
        } else
            return payload.throughput;
    });
}

//-------------------------------------------------------------------------------------------------
// Miss program for monte carlo rays.
//-------------------------------------------------------------------------------------------------

rtDeclareVariable(Ray, ray, rtCurrentRay, );
rtDeclareVariable(MonteCarloPayload, monte_carlo_payload, rtPayload, );
rtDeclareVariable(float3, g_scene_environment_tint, , );
#if PRESAMPLE_ENVIRONMENT_MAP
rtDeclareVariable(PresampledEnvironmentLight, g_scene_environment_light, , );
#else
rtDeclareVariable(EnvironmentLight, g_scene_environment_light, , );
#endif

RT_PROGRAM void miss() {
    float3 environment_radiance = g_scene_environment_tint;

    unsigned int environment_map_ID = g_scene_environment_light.environment_map_ID;
    if (environment_map_ID) {
        bool next_event_estimated = monte_carlo_payload.bounces != 0; // Was next event estimated at previous intersection.
        environment_radiance *= LightSources::evaluate_intersection(g_scene_environment_light, ray.origin, ray.direction, 
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

#ifdef DOUBLE_PRECISION_ACCUMULATION_BUFFER
    g_accumulation_buffer[g_launch_index] = make_double4(100000, 0, 0, 1.0);
#else
    g_accumulation_buffer[g_launch_index] = make_float4(100000, 0, 0, 1.0f);
#endif
}
