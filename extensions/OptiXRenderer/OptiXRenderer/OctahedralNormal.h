// OptiX octahedral encoded normal.
// ---------------------------------------------------------------------------
// Copyright (C) 2016, Bifrost. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _OPTIXRENDERER_OCTAHEDRAL_NORMAL_H_
#define _OPTIXRENDERER_OCTAHEDRAL_NORMAL_H_

#include <OptiXRenderer/Defines.h>

#include <optixu/optixu_math_namespace.h>

namespace OptiXRenderer {

//-------------------------------------------------------------------------------------------------
// OptiX decoder for Bifrost::Math::OctahedralNormal.
//-------------------------------------------------------------------------------------------------
struct __align__(4) OctahedralNormal {

    optix::short2 encoding;

    __inline_all__ static float sign(float v) { return v >= 0.0f ? +1.0f : -1.0f; }

    __inline_all__ optix::float3 decode_unnormalized() const {
        optix::float2 p2 = optix::make_float2(encoding.x, encoding.y);
        optix::float3 n = optix::make_float3(p2, SHRT_MAX - fabsf(p2.x) - fabsf(p2.y));
        if (n.z < 0.0f) {
            float tmp_x = (SHRT_MAX - fabsf(n.y)) * sign(n.x);
            n.y = (SHRT_MAX - fabsf(n.x)) * sign(n.y);
            n.x = tmp_x;
        }
        return n;
    }
};

} // NS OptiXRenderer

#endif // _OPTIXRENDERER_OCTAHEDRAL_NORMAL_H_