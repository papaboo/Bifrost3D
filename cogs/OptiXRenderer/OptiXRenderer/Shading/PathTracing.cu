// OptiX path tracing ray generation program and integrator.
// ---------------------------------------------------------------------------
// Copyright (C) 2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#include <OptiXRenderer/Types.h>

#include <optix.h>
#include <optixu/optixu_matrix_namespace.h>

using namespace optix;

rtDeclareVariable(uint2, g_launch_index, rtLaunchIndex, );

rtDeclareVariable(float, g_frame_number, , );
rtBuffer<float4, 2>  g_accumulation_buffer; // TODO Make double4

rtDeclareVariable(float4, g_camera_position, , );
rtDeclareVariable(Matrix4x4, g_inverted_view_projection_matrix, , );

rtDeclareVariable(float, g_scene_epsilon, , );

//----------------------------------------------------------------------------
// Ray generation program
//----------------------------------------------------------------------------
RT_PROGRAM void path_tracing() {
    if (g_frame_number == 0.0f)
        g_accumulation_buffer[g_launch_index] = make_float4(0.0, 0.0, 0.0, 0.0);

    // Generate rays.
    const float2 screen_pos = make_float2(g_launch_index.x / float(g_accumulation_buffer.size().x), g_launch_index.y / float(g_accumulation_buffer.size().y));
    const float4 normalized_screen_pos = make_float4(screen_pos.x * 2.0f - 1.0f, screen_pos.y * 2.0f - 1.0f, 1.0f, 1.0f);

    const float4 screenspace_world_pos = normalized_screen_pos * g_inverted_view_projection_matrix;

    const float3 ray_end = make_float3(screenspace_world_pos) / screenspace_world_pos.w;

    float3 origin = make_float3(g_camera_position);
    float3 direction = normalize(ray_end - origin);
    Ray ray(origin, direction, unsigned int(OptiXRenderer::RayTypes::MonteCarlo), g_scene_epsilon);

    g_accumulation_buffer[g_launch_index] = make_float4(direction * 0.5f + 0.5f, 1.0f);
}