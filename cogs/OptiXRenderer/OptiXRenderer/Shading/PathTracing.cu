// OptiX path tracing ray generation program and integrator.
// ---------------------------------------------------------------------------
// Copyright (C) 2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#include <OptiXRenderer/Shading/LightSources/EnvironmentLightImpl.h>
#include <OptiXRenderer/Types.h>
#include <OptiXRenderer/Utils.h>

#include <optix.h>
#include <optixu/optixu_matrix_namespace.h>

using namespace OptiXRenderer;
using namespace optix;

rtDeclareVariable(uint2, g_launch_index, rtLaunchIndex, );

rtDeclareVariable(int, g_accumulations, , );
rtBuffer<float4, 2>  g_output_buffer;
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

__inline_dev__ bool is_black(const optix::float3 color) {
    return color.x <= 0.0f && color.y <= 0.0f && color.z <= 0.0f;
}

__inline_dev__ inline optix::double3 lerp_double(const optix::double3& a, const optix::double3& b, const double t) {
    return optix::make_double3(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t, a.z + (b.z - a.z) * t);
}

//----------------------------------------------------------------------------
// Ray generation program
//----------------------------------------------------------------------------
RT_PROGRAM void path_tracing() {
    if (g_accumulations == 0)
#ifdef DOUBLE_PRECISION_ACCUMULATION_BUFFER
        g_accumulation_buffer[g_launch_index] = make_double4(0.0, 0.0, 0.0, 0.0);
#else
        g_accumulation_buffer[g_launch_index] = make_float4(0.0f);
#endif

    unsigned int index = g_launch_index.y * g_accumulation_buffer.size().x + g_launch_index.x;

    MonteCarloPRD prd;
    prd.radiance = make_float3(0.0f);
    prd.rng.seed(RNG::hash(index) ^ __brev(g_accumulations));
    // prd.rng.seed(__brev(g_accumulations)); // Uniform seed.
    prd.throughput = make_float3(1.0f);
    prd.bounces = 0;
    prd.bsdf_MIS_PDF = 0.0f;
    prd.clamped_path_PDF = prd.path_PDF = 1.0f;

    // Generate rays.
    float2 screen_pos_offset = prd.rng.sample2f(); // Always advance the rng by two samples, even if we ignore them.
    float2 screen_pos = make_float2(g_launch_index) + (g_accumulations == 0 ? make_float2(0.5f) : screen_pos_offset);
    float2 viewport_pos = make_float2(screen_pos.x / float(g_accumulation_buffer.size().x), screen_pos.y / float(g_accumulation_buffer.size().y));
    prd.position = make_float3(g_camera_position);
    prd.direction = project_ray_direction(viewport_pos, prd.position, g_inverted_view_projection_matrix);

    do {
        Ray ray(prd.position, prd.direction, unsigned int(RayTypes::MonteCarlo), g_scene_epsilon);
        rtTrace(g_scene_root, ray, prd);
    } while (prd.bounces < g_max_bounce_count && !is_black(prd.throughput));

#ifdef DOUBLE_PRECISION_ACCUMULATION_BUFFER
    double3 prev_radiance = make_double3(g_accumulation_buffer[g_launch_index].x, g_accumulation_buffer[g_launch_index].y, g_accumulation_buffer[g_launch_index].z);
    double3 accumulated_radiance_d = lerp_double(prev_radiance, make_double3(prd.radiance.x, prd.radiance.y, prd.radiance.z), 1.0 / (g_accumulations + 1.0));
    g_accumulation_buffer[g_launch_index] = make_double4(accumulated_radiance_d.x, accumulated_radiance_d.y, accumulated_radiance_d.z, 1.0f);
    float3 accumulated_radiance = make_float3(accumulated_radiance_d.x, accumulated_radiance_d.y, accumulated_radiance_d.z);
#else
    float3 prev_radiance = make_float3(g_accumulation_buffer[g_launch_index]);
    float3 accumulated_radiance = lerp(prev_radiance, prd.radiance, 1.0f / (g_accumulations + 1.0f));
    g_accumulation_buffer[g_launch_index] = make_float4(accumulated_radiance, 1.0f);
#endif

    // Apply simple gamma correction to the output.
    const float inv_screen_gamma = 1.0f / 2.2f;
    g_output_buffer[g_launch_index] = make_float4(gammacorrect(accumulated_radiance, inv_screen_gamma), 1.0f);
}

//----------------------------------------------------------------------------
// Miss program for monte carlo rays.
//----------------------------------------------------------------------------

rtDeclareVariable(Ray, ray, rtCurrentRay, );
rtDeclareVariable(MonteCarloPRD, monte_carlo_PRD, rtPayload, );
rtDeclareVariable(float3, g_scene_background_color, , );
rtDeclareVariable(EnvironmentLight, g_scene_environment_light, , ); // TODO Fetch from the end of the light source buffer and check type.

RT_PROGRAM void miss() {
    float3 environment_radiance;
    unsigned int environment_map_ID = g_scene_environment_light.environment_map_ID;
    if (environment_map_ID) {
        environment_radiance = LightSources::evaluate(g_scene_environment_light, ray.origin, ray.direction);
        
        bool next_event_estimatable = g_scene_environment_light.per_pixel_PDF_ID != RT_TEXTURE_ID_NULL;
        if (next_event_estimatable) {
            bool next_event_estimated = monte_carlo_PRD.bounces != 0; // Was next event estimated at previous intersection.
            bool apply_MIS = monte_carlo_PRD.bsdf_MIS_PDF > 0.0f;
            if (apply_MIS) {
                // Calculate MIS weight and scale the radiance by it.
                const float light_PDF = LightSources::PDF(g_scene_environment_light, ray.origin, ray.direction);
                float mis_weight = is_PDF_valid(light_PDF) ? RNG::power_heuristic(monte_carlo_PRD.bsdf_MIS_PDF, light_PDF) : 0.0f;
                environment_radiance *= mis_weight;
            } else if (next_event_estimated)
                // Previous bounce used next even estimation, but did not calculate MIS, so don't apply light contribution.
                // TODO Could this be handled by setting bsdf_MIS_PDF to 0 instead? 
                //      Wait until we have a specular BRDF implementation and
                //      remember to test with next event estimation on and off.
                environment_radiance = make_float3(0.0f);
        }
    } else
        environment_radiance = g_scene_background_color;

    float3 scaled_radiance = monte_carlo_PRD.throughput * environment_radiance;
    monte_carlo_PRD.radiance += clamp_light_contribution_by_pdf(scaled_radiance, monte_carlo_PRD.clamped_path_PDF, g_accumulations);

    monte_carlo_PRD.throughput = make_float3(0.0f);
}

//----------------------------------------------------------------------------
// Exception program.
//----------------------------------------------------------------------------
RT_PROGRAM void exceptions() {
    rtPrintExceptionDetails();

#ifdef DOUBLE_PRECISION_ACCUMULATION_BUFFER
    g_accumulation_buffer[g_launch_index] = make_double4(100000, 0, 0, 1.0);
#else
    g_accumulation_buffer[g_launch_index] = make_float4(100000, 0, 0, 1.0f);
#endif
}
