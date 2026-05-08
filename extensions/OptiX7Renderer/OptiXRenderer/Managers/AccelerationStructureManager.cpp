// OptiX acceleration structure manager.
// ------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#include <OptiXRenderer/Managers/AccelerationStructureManager.h>

#include <OptiXRenderer/Utils.h>

#include <optix.h>
#include <optix_stubs.h>

namespace OptiXRenderer::Managers {

AccelerationStructureManager::AccelerationStructureManager(OptixDeviceContext context)
    : m_context(context), m_handle(0), m_backing_buffer(0) {}

void AccelerationStructureManager::handle_updates() {

    if (m_handle)
        return;

    // Use default options for simplicity. In a real use case we would want to enable compaction, etc
    OptixAccelBuildOptions options = {};
    options.buildFlags = OPTIX_BUILD_FLAG_NONE;
    options.operation = OPTIX_BUILD_OPERATION_BUILD;

    // Triangle mesh
    const unsigned int vertex_count = 3;
    const float3 vertices[vertex_count] =
    {
        { -0.5f, -0.5f, 0.0f },
        {  0.5f, -0.5f, 0.0f },
        {  0.0f,  0.5f, 0.0f }
    };

    DeviceArray<float3> device_vertices = DeviceArray<float3>(vertices, vertex_count);

    // Our build input is a simple list of non-indexed triangle vertices
    CUdeviceptr vertex_buffers[1] = { device_vertices.device_ptr() };
    const unsigned int triangle_input_flags[1] = { OPTIX_GEOMETRY_FLAG_NONE };
    OptixBuildInput triangle_input = {};
    triangle_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    triangle_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    triangle_input.triangleArray.numVertices = vertex_count;
    triangle_input.triangleArray.vertexBuffers = vertex_buffers;
    triangle_input.triangleArray.flags = triangle_input_flags;
    triangle_input.triangleArray.numSbtRecords = 1;

    OptixAccelBufferSizes gas_buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(m_context, &options, &triangle_input, 1, &gas_buffer_sizes));

    DeviceArray<Bifrost::byte> temp_gas_buffer = DeviceArray<Bifrost::byte>(gas_buffer_sizes.tempSizeInBytes);

    m_backing_buffer = DeviceArray<Bifrost::byte>(gas_buffer_sizes.outputSizeInBytes);

    OPTIX_CHECK(optixAccelBuild(
        m_context,
        0, // CUDA stream
        &options,
        &triangle_input,
        1, // num build inputs
        temp_gas_buffer.device_ptr(),
        gas_buffer_sizes.tempSizeInBytes,
        m_backing_buffer.device_ptr(),
        gas_buffer_sizes.outputSizeInBytes,
        &m_handle,
        nullptr, // emitted property list
        0 // num emitted properties
    ));

    // Clear the backing buffer
    m_backing_buffer = DeviceArray<Bifrost::byte>();
}

} // NS OptiXRenderer::Managers