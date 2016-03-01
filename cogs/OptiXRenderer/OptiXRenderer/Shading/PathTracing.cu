// OptiX path tracing ray generation program and integrator.
// ---------------------------------------------------------------------------
// Copyright (C) 2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#include <OptiXRenderer/Shading/Utils.h>
#include <OptiXRenderer/Types.h>

#include <optix.h>
#include <optixu/optixu_matrix_namespace.h>

using namespace OptiXRenderer;
using namespace OptiXRenderer::Shading;
using namespace optix;

rtDeclareVariable(uint2, g_launch_index, rtLaunchIndex, );

rtDeclareVariable(int, g_accumulations, , );
rtBuffer<float4, 2>  g_accumulation_buffer;

rtDeclareVariable(float4, g_camera_position, , );
rtDeclareVariable(Matrix4x4, g_inverted_view_projection_matrix, , );

rtDeclareVariable(rtObject, g_scene_root, , );
rtDeclareVariable(float, g_scene_epsilon, , );

//----------------------------------------------------------------------------
// Ray generation program
//----------------------------------------------------------------------------
RT_PROGRAM void path_tracing() {
    if (g_accumulations == 0)
        g_accumulation_buffer[g_launch_index] = make_float4(0.0, 0.0, 0.0, 0.0);

    // Generate rays.
    float2 viewport_pos = make_float2(g_launch_index.x / float(g_accumulation_buffer.size().x), g_launch_index.y / float(g_accumulation_buffer.size().y));
    float3 origin = make_float3(g_camera_position);
    float3 direction = project_ray_direction(viewport_pos, origin, g_inverted_view_projection_matrix);

    Ray ray(origin, direction, unsigned int(RayTypes::MonteCarlo), g_scene_epsilon);

    MonteCarloPRD prd;
    prd.radiance = make_float3(0.0f);
    rtTrace(g_scene_root, ray, prd);

    // Simple gamma correction.
    float inv_screen_gamma = 1.0f / 2.2f;
    prd.radiance = gammacorrect(prd.radiance, inv_screen_gamma);

    g_accumulation_buffer[g_launch_index] = make_float4(prd.radiance, 1.0f);
}

//----------------------------------------------------------------------------
// Miss program for monte carlo rays.
//----------------------------------------------------------------------------

rtDeclareVariable(MonteCarloPRD, monte_carlo_PRD, rtPayload, );
RT_PROGRAM void miss() {
    monte_carlo_PRD.radiance = make_float3(0.68f, 0.92f, 1.0f);
}

//----------------------------------------------------------------------------
// Exception program.
//----------------------------------------------------------------------------
RT_PROGRAM void exceptions() {
    rtPrintExceptionDetails();

    g_accumulation_buffer[g_launch_index] = make_float4(100000, 0, 0, 1.0f);
}
