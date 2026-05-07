// OptiX renderer utilies for working with CUDA.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _OPTIXRENDERER_CUDA_UTILS_H_
#define _OPTIXRENDERER_CUDA_UTILS_H_

#include <OptiXRenderer/Defines.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <string>

namespace OptiXRenderer {

inline void throw_cuda_error(cudaError_t error, const std::string& file, int line) {
    if (error != cudaSuccess) {
        std::string message = "[file:" + file + " line:" + std::to_string(line) + "] CUDA errror: " + std::string(cudaGetErrorString(error));
        printf("%s.\n", message.c_str());
        throw std::exception(message.c_str(), error);
    }
}
#define THROW_CUDA_ERROR(error) throw_cuda_error(error, __FILE__,__LINE__)

// Typed variant of CUdeviceptr
template <typename T>
struct DevicePtr {
    CUdeviceptr ptr;

    DevicePtr() : ptr(0u) { }

    DevicePtr(DevicePtr&& other)
        : ptr(other.detach()) {}

    DevicePtr& operator=(DevicePtr&& rhs) {
        if (ptr)
            cudaFree((void**)&ptr);
        ptr = rhs.detach();
        return *this;
    }

    ~DevicePtr() {
        if (ptr)
            cudaFree((void**)&ptr);
        ptr = {};
    }

    static DevicePtr<T> create() {
        DevicePtr<T> ptr = {};
        THROW_CUDA_ERROR(cudaMalloc((void**)&ptr.ptr, sizeof(T)));
        return ptr;
    }

    void upload(const T& data) {
        THROW_CUDA_ERROR(cudaMemcpy((void*)ptr, &data, sizeof(T), cudaMemcpyHostToDevice));
    }

    static DevicePtr<T> create(const T& data) {
        DevicePtr<T> ptr = create();
        ptr.upload(data);
        return ptr;
    }

private:
    inline CUdeviceptr detach() { CUdeviceptr tmp = ptr; ptr = {}; return tmp; }

    // Disallow multiple ownership of the same data to avoid pointing to deleted data.
    DevicePtr(DevicePtr& other) = delete;
    DevicePtr& operator=(DevicePtr& rhs) = delete;
};


} // NS OptiXRenderer

#endif // _OPTIXRENDERER_CUDA_UTILS_H_