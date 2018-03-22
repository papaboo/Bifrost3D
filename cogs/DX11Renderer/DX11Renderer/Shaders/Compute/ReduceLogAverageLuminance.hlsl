// Compute the log-average of a texture and compute the linear exposure from that
// ------------------------------------------------------------------------------------------------
// Copyright (C) 2018, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _DX11_RENDERER_SHADERS_COMPUTE_REDUCE_LOG_AVERAGE_LUMINANCE_H_
#define _DX11_RENDERER_SHADERS_COMPUTE_REDUCE_LOG_AVERAGE_LUMINANCE_H_

#include "../Utils.hlsl"

static const uint GROUP_WIDTH = 8u;
static const uint GROUP_HEIGHT = 16u;
static const uint GROUP_THREAD_COUNT = GROUP_WIDTH * GROUP_HEIGHT;
static const uint MAX_GROUPS_DISPATCHED = 128u;

cbuffer constants : register(b0) {
    float min_log_luminance;
    float max_log_luminance;
    float min_percentage;
    float max_percentage;
}

groupshared float shared_log_luminance[GROUP_THREAD_COUNT];

Texture2D pixels : register(t0);
RWStructuredBuffer<float> log_average_write_buffer : register(u0);

[numthreads(GROUP_WIDTH, GROUP_HEIGHT, 1)]
void first_reduction(uint3 local_thread_ID : SV_GroupThreadID, uint3 group_ID : SV_GroupID, uint3 global_thread_ID : SV_DispatchThreadID) {

    uint linear_local_thread_ID = local_thread_ID.x + local_thread_ID.y * GROUP_WIDTH;
    shared_log_luminance[linear_local_thread_ID] = 0;

    uint width, height;
    pixels.GetDimensions(width, height);

    // Sum log luminance in shared memory by letting the groups sweep the image horizontally in steps of size GROUP_HEIGHT.
    uint2 pixel_coord;
    for (pixel_coord.x = global_thread_ID.x; pixel_coord.x < width; pixel_coord.x += GROUP_WIDTH * MAX_GROUPS_DISPATCHED) {
        for (pixel_coord.y = global_thread_ID.y; pixel_coord.y < height; pixel_coord.y += GROUP_HEIGHT) {
            float3 pixel = pixels[pixel_coord].rgb;
            float log_luminance = log2(max(luminance(pixel), 0.0001f));
            shared_log_luminance[linear_local_thread_ID] += log_luminance;
        }
    }
    GroupMemoryBarrierWithGroupSync();

    // Sum in shared memory
    for (uint offset = GROUP_THREAD_COUNT >> 1; offset > 0; offset >>= 1) {
        if (linear_local_thread_ID < offset)
            shared_log_luminance[linear_local_thread_ID] += shared_log_luminance[linear_local_thread_ID + offset];
        GroupMemoryBarrierWithGroupSync();
    }

    if (linear_local_thread_ID == 0)
        log_average_write_buffer[group_ID.x] = shared_log_luminance[0] / (width * height);
}

// ------------------------------------------------------------------------------------------------
// Compute exposure from the histogram.
// ------------------------------------------------------------------------------------------------

// Compute linear exposure from the geometric mean. See MJP's tonemapping sample.
// https://mynameismjp.wordpress.com/2010/04/30/a-closer-look-at-tone-mapping/
float geometric_mean_linear_exposure(float average_luminance) {
    float key_value = 1.03f - (2.0f / (2 + log2(average_luminance + 1)));
    return key_value / average_luminance;
}

// Single element buffer.
StructuredBuffer<float> log_average_read_buffer : register(t0);
RWStructuredBuffer<float> linear_exposure_buffer : register(u0);

groupshared float shared_log_average[MAX_GROUPS_DISPATCHED];

[numthreads(MAX_GROUPS_DISPATCHED, 1, 1)]
void second_reduction(uint3 local_thread_ID : SV_GroupThreadID) {
    uint thread_ID = local_thread_ID.x;

    shared_log_average[thread_ID] = log_average_read_buffer[thread_ID];
    GroupMemoryBarrierWithGroupSync();

    // Sum in shared memory
    for (uint offset = MAX_GROUPS_DISPATCHED >> 1; offset > 0; offset >>= 1) {
        if (thread_ID < offset)
            shared_log_average[thread_ID] += shared_log_average[thread_ID + offset];
        GroupMemoryBarrierWithGroupSync();
    }

    if (thread_ID == 0)
        linear_exposure_buffer[0] = shared_log_average[0];
}

[numthreads(MAX_GROUPS_DISPATCHED, 1, 1)]
void compute_linear_exposure(uint3 local_thread_ID : SV_GroupThreadID) {
    uint thread_ID = local_thread_ID.x;

    shared_log_average[thread_ID] = log_average_read_buffer[thread_ID];
    GroupMemoryBarrierWithGroupSync();

    // Sum in shared memory
    for (uint offset = MAX_GROUPS_DISPATCHED >> 1; offset > 0; offset >>= 1) {
        if (thread_ID < offset)
            shared_log_average[thread_ID] += shared_log_average[thread_ID + offset];
        GroupMemoryBarrierWithGroupSync();
    }

    if (thread_ID == 0) {
        float average_log_luminance = shared_log_average[0];
        average_log_luminance = clamp(average_log_luminance, min_log_luminance, max_log_luminance);
        linear_exposure_buffer[0] = geometric_mean_linear_exposure(exp2(average_log_luminance));
    }
}

#endif // _DX11_RENDERER_SHADERS_COMPUTE_REDUCE_LOG_AVERAGE_LUMINANCE_H_