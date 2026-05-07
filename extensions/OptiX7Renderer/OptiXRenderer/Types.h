// OptiX renderer POD types.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _OPTIXRENDERER_TYPES_H_
#define _OPTIXRENDERER_TYPES_H_

#include <OptiXRenderer/Defines.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <optix_types.h>

namespace OptiXRenderer {

struct half4 { __half x, y, z, w; };
half4 create_half4(float x, float y, float z, float w) { return { x, y, z, w }; }

// Record for the shader binding table
template <typename T>
struct SbtRecord {
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

struct CameraState {
    half4* output_buffer;
    unsigned int output_buffer_width;
    unsigned int output_buffer_height;
};

struct RayGenData {
    float r, g, b;
};

struct MissShaderData { };

} // NS OptiXRenderer

#endif // _OPTIXRENDERER_TYPES_H_