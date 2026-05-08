// OptiX renderer utilities for working with CUDA host code.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#include <OptiXRenderer/CUDAUtils.h>

#include <string>
#include <stdexcept>

namespace OptiXRenderer {

void throw_cuda_error(cudaError_t error, const char* const file, int line) {
    if (error != cudaSuccess) {
        std::string message = "[file:" + std::string(file) + " line:" + std::to_string(line) + "] CUDA error: " + std::string(cudaGetErrorString(error));
        printf("%s.\n", message.c_str());
        throw std::runtime_error(message);
    }
}

} // NS OptiXRenderer