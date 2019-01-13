// OptiX renderer manager.
// ---------------------------------------------------------------------------
// Copyright (C) 2016, Bifrost. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#include <OptiXRenderer/Kernel.h>

#include <cuda_runtime.h>

#include <cstdio>

namespace OptiXRenderer {

__global__ void lala(unsigned int* input, unsigned int element_count) {
    const unsigned int thread_index = threadIdx.x + blockDim.x * blockIdx.x;
    if (thread_index >= element_count)
        return;

    input[thread_index] = thread_index;
}

void cuda_check_error(cudaError_t error_code) {
    if (error_code == cudaSuccess)
        return;
        
    printf("Detected cuda error: %u\n", (int)error_code);
    exit(0);
}   

void launch() {

    cuda_check_error(cudaSetDevice(0));

    unsigned int* device_data = nullptr;
    cuda_check_error(cudaMalloc(&device_data, 128 * sizeof(unsigned int)));

    lala<<<2, 64 >>>(device_data, 128);

    unsigned int* host_data = new unsigned int[128];
    cuda_check_error(cudaMemcpy(host_data, device_data, 128 * sizeof(unsigned int), cudaMemcpyDeviceToHost));

    cuda_check_error(cudaFree(device_data));

    for (unsigned int i = 0; i < 128; ++i)
        printf("%u ", host_data[i]);
    printf("\n");
}

} // NS OptiXRenderer