// OptiX path tracing ray generation program and integrator.
// ---------------------------------------------------------------------------
// Copyright (C) 2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#include <OptiXRenderer/Types.h>

#include <optix.h>

using namespace OptiXRenderer;
using namespace optix;

rtDeclareVariable(float3, g_color, , );


rtDeclareVariable(MonteCarloPRD, monte_carlo_PRD, rtPayload, );

//----------------------------------------------------------------------------
// Closest hit program for monte carlo sampling rays.
//----------------------------------------------------------------------------

RT_PROGRAM void closest_hit() {
    monte_carlo_PRD.color = g_color;
}

//----------------------------------------------------------------------------
// Miss program for monte carlo rays.
//----------------------------------------------------------------------------

RT_PROGRAM void miss() {
    monte_carlo_PRD.color = make_float3(0.68f, 0.92f, 1.0f);
}