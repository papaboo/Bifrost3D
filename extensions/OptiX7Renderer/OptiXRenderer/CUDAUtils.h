// OptiX renderer utilities for working with CUDA host code.
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

#include <initializer_list>

namespace OptiXRenderer {

void throw_cuda_error(cudaError_t error, const char* const file, int line);
#define CUDA_CHECK(error) throw_cuda_error(error, __FILE__,__LINE__)

// Typed variant of CUdeviceptr
template <typename T>
struct DevicePtr {
    DevicePtr() : m_ptr(nullptr) { }

    DevicePtr(DevicePtr&& other)
        : m_ptr(other.detach()) {}

    DevicePtr& operator=(DevicePtr&& rhs) {
        if (m_ptr)
            cudaFree(m_ptr);
        m_ptr = rhs.detach();
        return *this;
    }

    ~DevicePtr() {
        if (m_ptr)
            cudaFree(m_ptr);
        m_ptr = nullptr;
    }

    static inline DevicePtr<T> create() {
        DevicePtr<T> ptr = {};
        CUDA_CHECK(cudaMalloc(&ptr.m_ptr, sizeof(T)));
        return ptr;
    }

    static inline DevicePtr<T> create(const T& data) {
        DevicePtr<T> ptr = create();
        ptr.upload(data);
        return ptr;
    }

    inline void upload(const T& data) {
        CUDA_CHECK(cudaMemcpy(m_ptr, &data, sizeof(T), cudaMemcpyHostToDevice));
    }

    inline T* data() { return m_ptr; }
    inline CUdeviceptr device_ptr() { return (CUdeviceptr)m_ptr; }

private:
    inline T* detach() { T* tmp = m_ptr; m_ptr = {}; return tmp; }

    // Disallow multiple ownership of the same data to avoid pointing to deleted data.
    DevicePtr(DevicePtr& other) = delete;
    DevicePtr& operator=(DevicePtr& rhs) = delete;

    T* m_ptr;
};

// Wrapper around a CUDA device array for data management. Could be replaced with thrust device_vector
template <typename T>
struct DeviceArray {
    DeviceArray() : m_size(0u), m_ptr(nullptr) {}

    DeviceArray(size_t size) : m_size(size) {
        CUDA_CHECK(cudaMalloc(&m_ptr, sizeof(T) * m_size));
    }

    DeviceArray(const T* const host_data, size_t size) : m_size(size) {
        CUDA_CHECK(cudaMalloc(&m_ptr, sizeof(T) * m_size));
        upload(host_data, m_size);
    }

    DeviceArray(const std::initializer_list<T>& list) : m_size(list.size()) {
        CUDA_CHECK(cudaMalloc(&m_ptr, sizeof(T) * m_size));
        upload(list.begin(), m_size);
    }

    DeviceArray(DeviceArray&& other)
        : m_size(other.m_size), m_ptr(other.detach()) {
        other.m_size = 0u;
    }

    DeviceArray& operator=(DeviceArray&& rhs) {
        if (m_ptr)
            cudaFree(m_ptr);
        m_size = rhs.m_size;
        m_ptr = rhs.detach();
        return *this;
    }

    ~DeviceArray() {
        if (m_ptr)
            cudaFree(m_ptr);
        m_size = 0u;
        m_ptr = nullptr;
    }

    inline size_t size() const { return m_size; }
    inline T* data() { return m_ptr; }
    inline CUdeviceptr device_ptr() { return (CUdeviceptr)m_ptr; }

    inline void resize(size_t new_size) {
        if (new_size == 0) {
            if (m_ptr)
                cudaFree(m_ptr);
            m_size = 0u;
            m_ptr = nullptr;
            return;
        }

        T* new_ptr;
        CUDA_CHECK(cudaMalloc(&new_ptr, sizeof(T) * new_size));

        size_t min_size = std::min(m_size, new_size);
        CUDA_CHECK(cudaMemcpy(new_ptr, m_ptr, sizeof(T) * min_size, cudaMemcpyDeviceToDevice));

        if (m_ptr)
            cudaFree(m_ptr);

        m_size = new_size;
        m_ptr = new_ptr;
    }

    inline void upload(const T* const host_data, size_t element_count) {
        CUDA_CHECK(cudaMemcpy(m_ptr, host_data, sizeof(T) * element_count, cudaMemcpyHostToDevice));
    }
    inline void upload(const T* const host_data) { upload(host_data, m_size); }

    inline void upload(size_t device_offset, const T* const host_data, size_t element_count) {
        CUDA_CHECK(cudaMemcpy(m_ptr + device_offset, host_data, sizeof(T) * element_count, cudaMemcpyHostToDevice));
    }

    inline void readback(T* host_output, size_t element_count) {
        CUDA_CHECK(cudaMemcpy(host_output, m_ptr, sizeof(T) * element_count, cudaMemcpyDeviceToHost));
    }

private:
    inline T* detach() { T* tmp = m_ptr; m_ptr = nullptr; return tmp; }

    // Disallow multiple ownership of the same data to avoid pointing to deleted data.
    DeviceArray(DeviceArray& other) = delete;
    DeviceArray& operator=(DeviceArray& rhs) = delete;

    size_t m_size;
    T* m_ptr;
};

} // NS OptiXRenderer

#endif // _OPTIXRENDERER_CUDA_UTILS_H_