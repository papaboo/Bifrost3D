// OptiX renderer POD types.
// ---------------------------------------------------------------------------
// Copyright (C) 2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _OPTIXRENDERER_TYPES_H_
#define _OPTIXRENDERER_TYPES_H_

#include <optixu/optixu_math.h>

namespace OptiXRenderer {

enum class RayTypes {
    MonteCarlo = 0,
    NormalVisualization,
    Count
};

enum class EntryPoints {
    PathTracing = 0,
    NormalVisualization,
    Count
};

struct MonteCarloPRD {
    float3 weight;
    float3 color;
};

} // NS OptiXRenderer

#endif // _OPTIXRENDERER_TYPES_H_