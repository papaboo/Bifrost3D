// Simple OptiX ray generation programs, such as path tracing, normal and albedo visualization
// ------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#include <optix.h>

#include <OptiXRenderer/Types.h>

using namespace Bifrost::Math;
using namespace OptiXRenderer;

float3 make_float3(Vector3f v) { return make_float3(v.x, v.y, v.z); }

extern "C" {
__constant__ PipelineParams g_params;
}

__inline_dev__ static void split_pointer(void* ptr, unsigned int& i0, unsigned int& i1) {
    const unsigned long long uptr = reinterpret_cast<unsigned long long>(ptr);
    i0 = uptr >> 32;
    i1 = uptr & 0xFFFFFFFF;
}

__inline_dev__ static void* combine_to_pointer(unsigned int i0, unsigned int i1) {
    const unsigned long long uptr = static_cast<unsigned long long>(i0) << 32 | i1;
    return reinterpret_cast<void*>(uptr);
}

__inline_dev__ static RadiancePayload* get_radiance_payload() {
    const unsigned int prd0 = optixGetPayload_0();
    const unsigned int prd1 = optixGetPayload_1();
    return reinterpret_cast<RadiancePayload*>(combine_to_pointer(prd0, prd1));
}

__inline_dev__ void fill_ray_info(Vector2f viewport_pos, const PipelineParams& params,
                                  Vector3f& origin, Vector3f& direction) {

    Vector4f NDC_near_pos = Vector4f(viewport_pos * 2.0f - 1.0f, -1.0f, 1.0f);
    Vector4f scaled_near_world_pos = params.inverse_view_projection_matrix * NDC_near_pos;
    origin = Vector3f(scaled_near_world_pos.x, scaled_near_world_pos.y, scaled_near_world_pos.z) / scaled_near_world_pos.w;

    Vector4f NDC_far_pos = Vector4f(NDC_near_pos.x, NDC_near_pos.y, 1.0f, 1.0f);
    Vector4f scaled_near_view_pos = params.inverse_projection_matrix * NDC_far_pos;
    direction = normalize(params.view_to_world_rotation * Vector3f(scaled_near_view_pos.x, scaled_near_view_pos.y, scaled_near_view_pos.z));
}

extern "C" __global__ void __raygen__radiance() {
    uint3 launch_index = optixGetLaunchIndex();
    unsigned int width = g_params.frame_width;
    unsigned int height = g_params.frame_height;

    // Generate rays.
    Vector2f screen_pos = Vector2f(launch_index.x + 0.5f, launch_index.y + 0.5f);
    Vector2f viewport_pos = Vector2f(screen_pos.x / float(width), screen_pos.y / float(height));
    Vector3f ray_origin, ray_direction;
    fill_ray_info(viewport_pos, g_params, ray_origin, ray_direction);

    float t_min = 0.0f, t_max = 1e30f, time = 0.0f;

    RadiancePayload prd = {};
    unsigned int prd0, prd1;
    split_pointer(&prd, prd0, prd1);

    optixTrace(g_params.scene.traversable, make_float3(ray_origin), make_float3(ray_direction), t_min, t_max, time,
        OptixVisibilityMask(1), OPTIX_RAY_FLAG_NONE,
        RayTypes::Radiance, // SBT offset
        RayTypes::Count, // SBT stride
        RayTypes::Radiance, // SBT miss index
        prd0, prd1); // Per ray data pointer

    int pixel_index = launch_index.y * width + launch_index.x;
    g_params.output_buffer[pixel_index] = make_rgba16f(prd.radiance);
}

extern "C" __global__ void __closesthit__radiance() {
    RadiancePayload* prd = get_radiance_payload();
    prd->radiance = RGB(0.1f, 0.9f, 0.1f);
}

extern "C" __global__ void __miss__radiance() {
    RadiancePayload* prd = get_radiance_payload();
    prd->radiance = g_params.scene.environment_tint;
}
