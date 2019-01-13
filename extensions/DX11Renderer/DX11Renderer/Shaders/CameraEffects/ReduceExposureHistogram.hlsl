// Compute the exposure histogram from a list of pixels.
// ------------------------------------------------------------------------------------------------
// Copyright (C) 2018, Bifrost. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#include "Utils.hlsl"

#pragma warning (disable: 3557) // Disable warning about single iteration loop being force unrolled.

namespace CameraEffects {

static const uint HISTOGRAM_SIZE = 64u;
static const uint LOG2_HISTOGRAM_SIZE = log2(HISTOGRAM_SIZE);
static const uint GROUP_WIDTH = 16u;
static const uint GROUP_HEIGHT = 8u;
static const uint GROUP_THREAD_COUNT = GROUP_WIDTH * GROUP_HEIGHT;
static const uint SHARED_ELEMENT_COUNT = GROUP_THREAD_COUNT * HISTOGRAM_SIZE;

// NOTE To reduce memory usage or increase thread count this could be stored as ushort2.
groupshared uint shared_histograms[SHARED_ELEMENT_COUNT];

Texture2D pixels : register(t0);
RWStructuredBuffer<uint> histogram_buffer : register(u0);

[numthreads(GROUP_WIDTH, GROUP_HEIGHT, 1)]
void reduce(uint3 local_thread_ID : SV_GroupThreadID, uint3 group_ID : SV_GroupID, uint3 global_thread_ID : SV_DispatchThreadID) {

    uint linear_local_thread_ID = local_thread_ID.x + local_thread_ID.y * GROUP_WIDTH;
    uint shared_histogram_offset = linear_local_thread_ID * HISTOGRAM_SIZE;

    for (uint bin = 0; bin < HISTOGRAM_SIZE; ++bin)
        shared_histograms[shared_histogram_offset + bin] = 0u;

    uint width = input_viewport.z;
    uint height = input_viewport.w;

    // Reduce the histogram in local memory by letting the groups sweep the image horizontally in steps of size GROUP_HEIGHT.
    uint2 pixel_coord = global_thread_ID.xy + input_viewport.xy;
    if (pixel_coord.x < width) {
        for (; pixel_coord.y < height; pixel_coord.y += GROUP_HEIGHT) {
            float3 pixel = pixels[pixel_coord].rgb;
            float log_luminance = log2(max(luminance(pixel), 0.0001));
            float normalized_index = inverse_lerp(min_log_luminance, max_log_luminance, log_luminance);
            int bin_index = clamp(int(normalized_index * HISTOGRAM_SIZE + 0.5), 0, int(HISTOGRAM_SIZE) - 1);
            ++shared_histograms[shared_histogram_offset + bin_index];
        }
    }
    GroupMemoryBarrierWithGroupSync();

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

// ------------------------------------------------------------------------------------------------
// Compute exposure from the histogram.
// ------------------------------------------------------------------------------------------------

// Single element buffer.

StructuredBuffer<uint> histogram_read_buffer : register(t0);
RWStructuredBuffer<float> linear_exposure_buffer : register(u0);

groupshared float shared_histogram[HISTOGRAM_SIZE + 1];
groupshared float shared_weighted_luminance[HISTOGRAM_SIZE];

[numthreads(HISTOGRAM_SIZE, 1, 1)]
void compute_linear_exposure(uint3 local_thread_ID : SV_GroupThreadID) {
    int thread_ID = local_thread_ID.x;

    shared_histogram[thread_ID] = histogram_read_buffer[thread_ID];
    GroupMemoryBarrierWithGroupSync();

    { // Compute prefix sum of the histogram in shared memory.

        // Reduce
        for (int step_size = 1; step_size < int(HISTOGRAM_SIZE); step_size <<= 1) {
            int src_index = 2 * thread_ID * step_size + step_size - 1;
            int dst_index = src_index + step_size;
            if (dst_index < int(HISTOGRAM_SIZE))
                shared_histogram[dst_index] += shared_histogram[src_index];
            GroupMemoryBarrierWithGroupSync();
        }

        // Copy element count to last bin
        if (thread_ID == 0)
            shared_histogram[HISTOGRAM_SIZE] = shared_histogram[HISTOGRAM_SIZE - 1];
        GroupMemoryBarrierWithGroupSync();

        // Downsweep
        for (int downsweep_step_size = int(pow(2, LOG2_HISTOGRAM_SIZE - 1)); downsweep_step_size > 0; downsweep_step_size >>= 1) {
            int low_index = 2 * thread_ID * downsweep_step_size + downsweep_step_size - 1;
            int high_index = low_index + downsweep_step_size;
            if (high_index < int(HISTOGRAM_SIZE)) {
                int t = shared_histogram[low_index];
                shared_histogram[low_index] = shared_histogram[high_index];
                shared_histogram[high_index] += t;
            }
            GroupMemoryBarrierWithGroupSync();
        }

        // Subtract max element
        shared_histogram[thread_ID] -= shared_histogram[HISTOGRAM_SIZE];
    }

    // Adjust prefix sum to min and max boundary values.
    // Clamp values above the max boundary and zero values outside of the min boundary.
    // NOTE Bin prefix and next bin prefix can be computed pr per thread and thereby avoid the sync below, but this seems to trigger a bug in the compiler/driver.
    float max_pixel_count = shared_histogram[HISTOGRAM_SIZE] * max_percentage;
    float min_pixel_count = shared_histogram[HISTOGRAM_SIZE] * min_percentage;
    shared_histogram[thread_ID] = max(0.0, min(shared_histogram[thread_ID], max_pixel_count) - min_pixel_count);
    GroupMemoryBarrierWithGroupSync();

    if (thread_ID == int(HISTOGRAM_SIZE - 1))
        shared_histogram[HISTOGRAM_SIZE] = max_pixel_count - min_pixel_count;

    float bin_count = shared_histogram[thread_ID + 1] - shared_histogram[thread_ID];
    float normalized_index = (thread_ID + 0.5) / HISTOGRAM_SIZE;
    float bin_log_luminance = lerp(min_log_luminance, max_log_luminance, normalized_index);
    shared_weighted_luminance[thread_ID] = exp2(bin_log_luminance) * bin_count;
    GroupMemoryBarrierWithGroupSync();

    // Sum in shared memory
    for (int offset = HISTOGRAM_SIZE >> 1; offset > 0; offset >>= 1) {
        if (thread_ID < offset)
            shared_weighted_luminance[thread_ID] += shared_weighted_luminance[thread_ID + offset];
        GroupMemoryBarrierWithGroupSync();
    }

    if (thread_ID == 0) {
        float average_luminance = shared_weighted_luminance[0] / (max_pixel_count - min_pixel_count);
        float linear_exposure = exp2(log_lumiance_bias) / average_luminance;
        linear_exposure_buffer[0] = eye_adaptation(linear_exposure_buffer[0], linear_exposure);
    }
}

} // NS CameraEffects