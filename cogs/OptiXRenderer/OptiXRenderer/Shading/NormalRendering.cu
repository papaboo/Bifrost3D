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

rtDeclareVariable(float4, g_camera_position, , );
rtDeclareVariable(Matrix4x4, g_inverted_view_projection_matrix, , );

rtBuffer<ushort4, 2>  g_output_buffer;

//----------------------------------------------------------------------------
// Ray generation program for visualizing normals.
//----------------------------------------------------------------------------
RT_PROGRAM void ray_generation() {
    // Generate rays.
    float2 viewport_pos = make_float2(g_launch_index.x / float(g_output_buffer.size().x), g_launch_index.y / float(g_output_buffer.size().y));
    float3 origin = make_float3(g_camera_position);
    float3 direction = project_ray_direction(viewport_pos, origin, g_inverted_view_projection_matrix);
    Ray ray(origin, direction, RayTypes::NormalVisualization, 0.0f);

    NormalVisualizationPayload payload;
    rtTrace(g_scene_root, ray, payload);

    g_output_buffer[g_launch_index] = float_to_half(payload.color);
}

//----------------------------------------------------------------------------
// Closest hit program for visualizing normals.
//----------------------------------------------------------------------------

rtDeclareVariable(Ray, ray, rtCurrentRay, );

rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, );
rtDeclareVariable(float3, shading_normal, attribute shading_normal, );

RT_PROGRAM void closest_hit() {
    const float3 world_shading_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
    float D_dot_N = -dot(ray.direction, world_shading_normal);
    if (D_dot_N < 0.0f)
        normal_visualization_payload.color = make_float4(0.25f - 0.75f * D_dot_N, 0.0f, 0.0f, 1.0);
    else
        normal_visualization_payload.color = make_float4(0.0f, 0.25f + 0.75f * D_dot_N, 0.0f, 1.0);
}

//----------------------------------------------------------------------------
// Miss program for normal visualization.
//----------------------------------------------------------------------------
RT_PROGRAM void miss() {
    normal_visualization_payload.color = make_float4(ray.direction * 0.2f + 0.2f, 1.0);
}