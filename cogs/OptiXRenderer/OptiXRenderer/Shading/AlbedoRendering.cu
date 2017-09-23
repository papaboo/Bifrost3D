// OptiX programs for visualizing a models normals.
// ---------------------------------------------------------------------------
// Copyright (C) 2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#include <OptiXRenderer/Types.h>
#include <OptiXRenderer/Utils.h>

#include <optix.h>
#include <optixu/optixu_matrix_namespace.h>

using namespace OptiXRenderer;
using namespace optix;

struct NormalVisualizationPayload {
    float4 color;
};

rtDeclareVariable(NormalVisualizationPayload, normal_visualization_payload, rtPayload, );

rtDeclareVariable(uint2, g_launch_index, rtLaunchIndex, );
rtDeclareVariable(rtObject, g_scene_root, , );
rtDeclareVariable(float, g_scene_epsilon, , );

rtDeclareVariable(float4, g_camera_position, , );
rtDeclareVariable(Matrix4x4, g_inverted_view_projection_matrix, , );

rtDeclareVariable(int, g_accumulations, , );
rtBuffer<ushort4, 2>  g_output_buffer;
#ifdef DOUBLE_PRECISION_ACCUMULATION_BUFFER
rtBuffer<double4, 2>  g_accumulation_buffer;
#else
rtBuffer<float4, 2>  g_accumulation_buffer;
#endif

//----------------------------------------------------------------------------
// Ray generation program for visualizing normals.
//----------------------------------------------------------------------------
RT_PROGRAM void ray_generation() {
    if (g_accumulations == 0)
#ifdef DOUBLE_PRECISION_ACCUMULATION_BUFFER
        g_accumulation_buffer[g_launch_index] = make_double4(0.0, 0.0, 0.0, 0.0);
#else
        g_accumulation_buffer[g_launch_index] = make_float4(0.0f);
#endif

    MonteCarloPayload payload = initialize_monte_carlo_payload(g_launch_index.x, g_launch_index.y,
        g_accumulation_buffer.size().x, g_accumulation_buffer.size().y, g_accumulations,
        make_float3(g_camera_position), g_inverted_view_projection_matrix);

    // Iterate until a material is sampled.
    float3 last_ray_direction = payload.direction;
    do {
        last_ray_direction = payload.direction;
        Ray ray(payload.position, payload.direction, RayTypes::MonteCarlo, g_scene_epsilon);
        rtTrace(g_scene_root, ray, payload);
    } while (payload.material_index == 0 && !is_black(payload.throughput));

    payload.radiance = payload.throughput;

#ifdef DOUBLE_PRECISION_ACCUMULATION_BUFFER
    double3 prev_radiance = make_double3(g_accumulation_buffer[g_launch_index].x, g_accumulation_buffer[g_launch_index].y, g_accumulation_buffer[g_launch_index].z);
    double3 accumulated_radiance_d = lerp_double(prev_radiance, make_double3(payload.radiance.x, payload.radiance.y, payload.radiance.z), 1.0 / (g_accumulations + 1.0));
    g_accumulation_buffer[g_launch_index] = make_double4(accumulated_radiance_d.x, accumulated_radiance_d.y, accumulated_radiance_d.z, 1.0f);
    float3 accumulated_radiance = make_float3(accumulated_radiance_d.x, accumulated_radiance_d.y, accumulated_radiance_d.z);
#else
    float3 prev_radiance = make_float3(g_accumulation_buffer[g_launch_index]);
    float3 accumulated_radiance = lerp(prev_radiance, payload.radiance, 1.0f / (g_accumulations + 1.0f));
    g_accumulation_buffer[g_launch_index] = make_float4(accumulated_radiance, 1.0f);
#endif

    g_output_buffer[g_launch_index] = float_to_half(make_float4(accumulated_radiance, 1.0f));
}