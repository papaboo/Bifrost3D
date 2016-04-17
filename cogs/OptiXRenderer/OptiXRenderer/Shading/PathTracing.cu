// OptiX path tracing ray generation program and integrator.
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

rtDeclareVariable(uint2, g_launch_index, rtLaunchIndex, );

rtDeclareVariable(int, g_accumulations, , );
rtBuffer<float4, 2>  g_accumulation_buffer;

rtDeclareVariable(float4, g_camera_position, , );
rtDeclareVariable(Matrix4x4, g_inverted_view_projection_matrix, , );

rtDeclareVariable(rtObject, g_scene_root, , );
rtDeclareVariable(float, g_scene_epsilon, , );

__inline_dev__ bool is_black(const optix::float3 color) {
    return color.x <= 0.0f && color.y <= 0.0f && color.z <= 0.0f;
}

//----------------------------------------------------------------------------
// Ray generation program
//----------------------------------------------------------------------------
RT_PROGRAM void path_tracing() {
    if (g_accumulations == 0)
        g_accumulation_buffer[g_launch_index] = make_float4(0.0, 0.0, 0.0, 0.0);

    unsigned int index = g_launch_index.y * g_accumulation_buffer.size().x + g_launch_index.x;

    MonteCarloPRD prd;
    prd.radiance = make_float3(0.0f);
    prd.rng.seed(RNG::hash(index) ^ __brev(g_accumulations));
    prd.throughput = make_float3(1.0f);
    prd.bounces = 0;
    prd.bsdf_MIS_PDF = 0.0f;
    prd.path_PDF = 1.0f;

    // Generate rays.
    float2 screen_pos_offset = prd.rng.sample2f(); // Always advance the rng by two samples, even if we ignore them.
    float2 screen_pos = make_float2(g_launch_index) + (g_accumulations == 0 ? make_float2(0.5f) : screen_pos_offset);
    float2 viewport_pos = make_float2(screen_pos.x / float(g_accumulation_buffer.size().x), screen_pos.y / float(g_accumulation_buffer.size().y));
    prd.position = make_float3(g_camera_position);
    prd.direction = project_ray_direction(viewport_pos, prd.position, g_inverted_view_projection_matrix);

    do {
        Ray ray(prd.position, prd.direction, unsigned int(RayTypes::MonteCarlo), g_scene_epsilon);
        rtTrace(g_scene_root, ray, prd);
    } while (prd.bounces < 4 && !is_black(prd.throughput));

    // Apply simple gamma correction to the output. TODO Use second output buffer, to avoid transforming back and forth between linear and gamma space.
    const float screen_gamma = 2.2f;
    const float inv_screen_gamma = 1.0f / 2.2f;
    float3 prev_radiance = gammacorrect(make_float3(g_accumulation_buffer[g_launch_index]), screen_gamma);
    float3 accumulated_radiance = lerp(prev_radiance, prd.radiance, 1.0f / (g_accumulations + 1.0f));
    g_accumulation_buffer[g_launch_index] = make_float4(gammacorrect(accumulated_radiance, inv_screen_gamma), 1.0f);
}

//----------------------------------------------------------------------------
// Miss program for monte carlo rays.
//----------------------------------------------------------------------------

rtDeclareVariable(MonteCarloPRD, monte_carlo_PRD, rtPayload, );
RT_PROGRAM void miss() {
    // monte_carlo_PRD.radiance += monte_carlo_PRD.throughput * make_float3(0.68f, 0.92f, 1.0f);
    monte_carlo_PRD.throughput = make_float3(0.0f);
}

//----------------------------------------------------------------------------
// Exception program.
//----------------------------------------------------------------------------
RT_PROGRAM void exceptions() {
    rtPrintExceptionDetails();

    g_accumulation_buffer[g_launch_index] = make_float4(100000, 0, 0, 1.0f);
}
