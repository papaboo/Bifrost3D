// OptiX path tracing ray generation program and integrator.
// ---------------------------------------------------------------------------
// Copyright (C) 2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#include <OptiXRenderer/Types.h>

#include <optix.h>

struct NormalVisualizationPRD {
    float4 color;
};

rtDeclareVariable(NormalVisualizationPRD, normal_visualization_PRD, rtPayload, );

rtDeclareVariable(uint2, g_launch_index, rtLaunchIndex, );
rtDeclareVariable(rtObject, g_scene_root, , );
rtDeclareVariable(float, g_scene_epsilon, , );

rtBuffer<float4, 2>  g_accumulation_buffer; // TODO Make double4

//----------------------------------------------------------------------------
// Ray generation program for visualizing normals.
//----------------------------------------------------------------------------
RT_PROGRAM void ray_generation() {
    NormalVisualizationPRD prd;

    const float3 position = make_float3(0.0f, 0.0f, 0.0f);
    const float3 direction = make_float3(0.0f, 0.0f, 1.0f);

    //optix::Ray ray(position, direction, unsigned int(OptiXRenderer::RayTypes::NormalVisualization), g_scene_epsilon);
    //rtTrace(g_scene_root, ray, prd);

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