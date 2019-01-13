// Prefix sum implementation as described in Scan Primitives for GPU Computing, Sengupta et al.
// ------------------------------------------------------------------------------------------------
// Copyright (C) 2017, Bifrost. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _DX11_RENDERER_SHADERS_COMPUTE_PREFIX_SUM_H_
#define _DX11_RENDERER_SHADERS_COMPUTE_PREFIX_SUM_H_

#define GROUP_SIZE 256
#define LOG2_GROUP_SIZE 8

cbuffer constants : register(b0) {
    int element_interval; // The interval between the elements loaded.
    int zero_last_entry; // Used by the downsweep to determine if it should zero the last element.
    int is_inner_loop;
    int __padding;
}

RWStructuredBuffer<uint> input_buffer : register(u0);

groupshared uint shared_memory[GROUP_SIZE];

// ------------------------------------------------------------------------------------------------
// Utility functions.
// ------------------------------------------------------------------------------------------------

unsigned int ceil_divide(uint a, uint b) {
    return (a / b) + ((a % b) > 0);
}

// Ensure that the last of the threads point to the last of the elements in the buffer.
// This will result in the first threads having negative indices and therefore loading zeroed values.
uint compute_global_index(uint global_thread_ID, uint element_count) {
    uint scaled_element_count = ceil_divide(element_count, element_interval);
    uint group_count = ceil_divide(scaled_element_count, GROUP_SIZE);
    uint threads_launched = group_count * GROUP_SIZE;
    uint element_padding = element_interval * (threads_launched - 1u) - (element_count - 1u);
    return element_interval * global_thread_ID - element_padding;
}

// ------------------------------------------------------------------------------------------------
// Reduce, algorithm 1.
// ------------------------------------------------------------------------------------------------

[numthreads(GROUP_SIZE, 1, 1)]
void reduce(uint3 local_thread_ID : SV_GroupThreadID, uint3 global_thread_ID : SV_DispatchThreadID) {
    uint element_count, element_size;
    input_buffer.GetDimensions(element_count, element_size);
    uint global_index = compute_global_index(global_thread_ID.x, element_count);

    // TODO do first reduction in place
    shared_memory[local_thread_ID.x] = input_buffer[global_index];
    GroupMemoryBarrierWithGroupSync();

    for (int step_size = 1; step_size < GROUP_SIZE; step_size <<= 1) {
        int src_index = 2 * local_thread_ID.x * step_size + step_size - 1;
        int dst_index = src_index + step_size;
        if (dst_index < GROUP_SIZE)
            shared_memory[dst_index] += shared_memory[src_index];
        GroupMemoryBarrierWithGroupSync();
    }

    input_buffer[global_index] = shared_memory[local_thread_ID.x];
}

// ------------------------------------------------------------------------------------------------
// Down-sweep, algorithm 2.
// ------------------------------------------------------------------------------------------------
[numthreads(GROUP_SIZE, 1, 1)]
void downsweep(uint3 local_thread_ID : SV_GroupThreadID, uint3 global_thread_ID : SV_DispatchThreadID) {
    uint element_count, element_size;
    input_buffer.GetDimensions(element_count, element_size);
    uint global_index = compute_global_index(global_thread_ID.x, element_count);

    // TODO do first reduction in place
    uint value = input_buffer[global_index];
    bool zero_entry = zero_last_entry && global_index == (element_count - 1u);
    shared_memory[local_thread_ID.x] = zero_entry ? 0u : value;
    GroupMemoryBarrierWithGroupSync();

    for (int step_size = (int)pow(2, LOG2_GROUP_SIZE- 1); step_size > 0; step_size >>= 1) {
        int low_index = 2 * local_thread_ID.x * step_size + step_size - 1;
        int high_index = low_index + step_size;
        if (high_index < GROUP_SIZE) {
            int t = shared_memory[low_index];
            shared_memory[low_index] = shared_memory[high_index];
            shared_memory[high_index] += t;
        }
        GroupMemoryBarrierWithGroupSync();
    }

    input_buffer[global_index] = shared_memory[local_thread_ID.x];
}

#endif // _DX11_RENDERER_SHADERS_COMPUTE_PREFIX_SUM_H_