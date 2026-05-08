// OptiX acceleration structure manager.
// ------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _OPTIXRENDERER_MANAGERS_ACCELERATION_STRUCTURE_MANAGER_H_
#define _OPTIXRENDERER_MANAGERS_ACCELERATION_STRUCTURE_MANAGER_H_

#include <OptiXRenderer/CUDAUtils.h>

typedef struct OptixDeviceContext_t* OptixDeviceContext;
typedef unsigned long long OptixTraversableHandle;
typedef unsigned long long CUdeviceptr;

namespace OptiXRenderer::Managers {

// ------------------------------------------------------------------------------------------------
// Wrapper around the top level and bottom level acceleration structures.
// ------------------------------------------------------------------------------------------------
class AccelerationStructureManager {
public:

    AccelerationStructureManager() = default;
    AccelerationStructureManager(OptixDeviceContext context);
    AccelerationStructureManager(AccelerationStructureManager&& other) = default;
    AccelerationStructureManager& operator=(AccelerationStructureManager&& rhs) = default;

    inline OptixTraversableHandle get_acceleration_structure() { return m_handle; }

    void handle_updates();

private:
    AccelerationStructureManager(AccelerationStructureManager& other) = delete;
    AccelerationStructureManager& operator=(AccelerationStructureManager& rhs) = delete;

    OptixDeviceContext m_context;

    OptixTraversableHandle m_handle;
    DeviceArray<Bifrost::byte> m_backing_buffer;
};

} // NS OptiXRenderer::Managers

#endif //  _OPTIXRENDERER_MANAGERS_ACCELERATION_STRUCTURE_MANAGER_H_