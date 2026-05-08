// Simple OptiX ray generation programs, such as path tracing, normal and albedo visualization
// ------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#include <optix.h>

#include <OptiXRenderer/Types.h>

using namespace OptiXRenderer;

extern "C" {
__constant__ PipelineParams params;
}

extern "C"
__global__ void __raygen__solid_color()
{
    uint3 launch_index = optixGetLaunchIndex();
    int pixel_index = launch_index.y * params.frame_width + launch_index.x;
    params.output_buffer[pixel_index] = create_half4(params.scene.environment_tint, 1.0f);
}
