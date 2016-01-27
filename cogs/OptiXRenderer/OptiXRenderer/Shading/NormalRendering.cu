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

using namespace OptiXRenderer;
using namespace optix;

struct NormalVisualizationPRD {
    float4 color;
};

rtDeclareVariable(NormalVisualizationPRD, normal_visualization_PRD, rtPayload, );

rtDeclareVariable(uint2, g_launch_index, rtLaunchIndex, );
rtDeclareVariable(rtObject, g_scene_root, , );
rtDeclareVariable(float, g_scene_epsilon, , );

rtDeclareVariable(float4, g_camera_position, , );
rtDeclareVariable(Matrix4x4, g_inverted_view_projection_matrix, , );

rtBuffer<float4, 2>  g_accumulation_buffer; // TODO Make double4

//----------------------------------------------------------------------------
// Ray generation program for visualizing normals.
//----------------------------------------------------------------------------
RT_PROGRAM void ray_generation() {
    // Generate rays.
    const float2 screen_pos = make_float2(g_launch_index.x / float(g_accumulation_buffer.size().x), g_launch_index.y / float(g_accumulation_buffer.size().y));
    const float4 normalized_screen_pos = make_float4(screen_pos.x * 2.0f - 1.0f,
                                                     1.0f - screen_pos.y * 2.0f, // Inlined negate of the screen position.
                                                     1.0f, 1.0f);

    const float4 screenspace_world_pos = g_inverted_view_projection_matrix * normalized_screen_pos;

    const float3 ray_end = make_float3(screenspace_world_pos) / screenspace_world_pos.w;

    float3 origin = make_float3(g_camera_position);
    float3 direction = normalize(ray_end - origin);
    Ray ray(origin, direction, unsigned int(RayTypes::NormalVisualization), g_scene_epsilon);

    NormalVisualizationPRD prd;
    rtTrace(g_scene_root, ray, prd);

    g_accumulation_buffer[g_launch_index] = prd.color;
}

//----------------------------------------------------------------------------
// Closest hit program for visualizing normals.
//----------------------------------------------------------------------------

rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, );

RT_PROGRAM void closest_hit() {
    float3 remapped_normal = geometric_normal * 0.5f + 0.5f;
    normal_visualization_PRD.color = make_float4(remapped_normal, 1.0);
}

//----------------------------------------------------------------------------
// Miss program for normal visualization.
//----------------------------------------------------------------------------

RT_PROGRAM void miss() {
    normal_visualization_PRD.color = make_float4(0.0f, 0.0f, 0.0f, 1.0);
}