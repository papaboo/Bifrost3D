// Compute the exposure histogram from a list of pixels.
// ------------------------------------------------------------------------------------------------
// Copyright (C) 2018, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _DX11_RENDERER_SHADERS_COMPUTE_REDUCE_EXPOSURE_HISTOGRAM_H_
#define _DX11_RENDERER_SHADERS_COMPUTE_REDUCE_EXPOSURE_HISTOGRAM_H_

#include "../Utils.hlsl"

static const uint HISTOGRAM_SIZE = 64u;
static const uint LOG2_HISTOGRAM_SIZE = log2(HISTOGRAM_SIZE);
static const uint GROUP_WIDTH = 16u;
static const uint GROUP_HEIGHT = 8u;
static const uint GROUP_THREAD_COUNT = GROUP_WIDTH * GROUP_HEIGHT;
static const uint SHARED_ELEMENT_COUNT = GROUP_THREAD_COUNT * HISTOGRAM_SIZE;

cbuffer constants : register(b0) {
    float min_log_luminance;
    float max_log_luminance;
    float2 __padding2;
}

groupshared uint shared_histograms[SHARED_ELEMENT_COUNT];

Texture2D pixels : register(t0);
RWStructuredBuffer<uint> histogram_buffer : register(u0);

[numthreads(GROUP_WIDTH, GROUP_HEIGHT, 1)]
void reduce(uint3 local_thread_ID : SV_GroupThreadID, uint3 group_ID : SV_GroupID, uint3 global_thread_ID : SV_DispatchThreadID) {

    uint linear_local_thread_ID = local_thread_ID.x + local_thread_ID.y * GROUP_WIDTH;
    uint shared_histogram_offset = linear_local_thread_ID * HISTOGRAM_SIZE;

    for (uint bin = 0; bin < HISTOGRAM_SIZE; ++bin)
        shared_histograms[shared_histogram_offset + bin] = 0u;

    uint width, height;
    pixels.GetDimensions(width, height);

    // Reduce the histogram in local memory by letting the groups sweep the image horizontally in steps of size GROUP_HEIGHT.
    uint2 pixel_coord = { local_thread_ID.x + group_ID.x * GROUP_WIDTH, local_thread_ID.y };
    if (pixel_coord.x < width) {
        for (; pixel_coord.y < height; pixel_coord.y += GROUP_HEIGHT) {
            float3 pixel = pixels[pixel_coord].rgb;
            float log_luminance = log2(max(luminance(pixel), 0.0001f));
            float normalized_index = inverse_lerp(min_log_luminance, max_log_luminance, log_luminance);
            int bin_index = clamp(int(normalized_index * HISTOGRAM_SIZE), 0, int(HISTOGRAM_SIZE) - 1);
            ++shared_histograms[shared_histogram_offset + bin_index];
        }
    }
    GroupMemoryBarrierWithGroupSync();

    /*
    // General purpose shared memory histogram reduction.
    uint shared_histogram_bin = linear_local_thread_ID;
    for (uint histogram_ID = 1; histogram_ID < GROUP_THREAD_COUNT; ++histogram_ID) {
        if (shared_histogram_bin < HISTOGRAM_SIZE)
            shared_histograms[shared_histogram_bin] += shared_histograms[histogram_ID * HISTOGRAM_SIZE + shared_histogram_bin];
    }
    GroupMemoryBarrierWithGroupSync();
    */

    // Reduce the histogram in shared memory by first having all threads reduce a single bin and then having the first HISTOGRAM_SIZE threads do the final reduction.
    // Only works if the thread count modulo histogram size is zero.
    for (uint histogram_bin_ID = linear_local_thread_ID + GROUP_THREAD_COUNT; 
        histogram_bin_ID < SHARED_ELEMENT_COUNT; 
        histogram_bin_ID += GROUP_THREAD_COUNT)
        shared_histograms[linear_local_thread_ID] += shared_histograms[histogram_bin_ID];
    GroupMemoryBarrierWithGroupSync();

    uint shared_reduced_histograms = GROUP_THREAD_COUNT / HISTOGRAM_SIZE;
    uint shared_histogram_bin = linear_local_thread_ID;
    for (uint histogram_ID = 1; histogram_ID < shared_reduced_histograms; ++histogram_ID) {
        if (shared_histogram_bin < HISTOGRAM_SIZE)
            shared_histograms[shared_histogram_bin] += shared_histograms[histogram_ID * HISTOGRAM_SIZE + shared_histogram_bin];
    }
    GroupMemoryBarrierWithGroupSync();

    if (linear_local_thread_ID < HISTOGRAM_SIZE) {
        uint dummy;
        InterlockedAdd(histogram_buffer[linear_local_thread_ID], shared_histograms[linear_local_thread_ID], dummy);
    }
}

#endif // _DX11_RENDERER_SHADERS_COMPUTE_REDUCE_EXPOSURE_HISTOGRAM_H_