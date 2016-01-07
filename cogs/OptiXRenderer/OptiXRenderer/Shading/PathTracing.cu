// OptiX path tracing ray generation program and integrator.
// ---------------------------------------------------------------------------
// Copyright (C) 2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#include <OptiXRenderer/Types.h>

#include <optix.h>
#include <optixu/optixu_math.h>

rtDeclareVariable(uint2, g_launch_index, rtLaunchIndex, );

rtDeclareVariable(float, g_frame_number, , );
rtBuffer<float4, 2>  g_accumulation_buffer; // TODO Make double4

//----------------------------------------------------------------------------
// Ray generation program
//----------------------------------------------------------------------------
RT_PROGRAM void PathTracing() {
    if (g_frame_number == 0.0f)
        g_accumulation_buffer[g_launch_index] = make_float4(0.0, 0.0, 0.0, 0.0);

    g_accumulation_buffer[g_launch_index] = make_float4(1.0f / (g_frame_number + 1.0f), 1.0f - 1.0f / (g_frame_number + 1.0f), 0.0, 1.0);
}