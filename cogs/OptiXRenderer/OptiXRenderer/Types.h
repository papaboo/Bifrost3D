// OptiX renderer POD types.
// ---------------------------------------------------------------------------
// Copyright (C) 2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _OPTIXRENDERER_TYPES_H_
#define _OPTIXRENDERER_TYPES_H_

#include <optixu/optixu_math_namespace.h>

#ifndef __inline_all__
#    if (defined(__CUDACC__) || defined(__CUDABE__))
#        define __inline_all__ __forceinline__ __host__ __device__
#    else
#        define __inline_all__ inline
#    endif
#endif

#ifndef __inline_dev__
#    if (defined(__CUDACC__) || defined(__CUDABE__))
#        define __inline_dev__ __forceinline__ __device__
#    else
#        define __inline_dev__ inline
#    endif
#endif

#define PIf 3.14159265358979323846f

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
    optix::float3 weight;
    optix::float3 color;
};

} // NS OptiXRenderer

#endif // _OPTIXRENDERER_TYPES_H_