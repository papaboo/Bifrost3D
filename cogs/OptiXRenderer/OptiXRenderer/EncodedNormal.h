// OptiX encoded normal.
// ---------------------------------------------------------------------------
// Copyright (C) 2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _OPTIXRENDERER_ENCODED_NORMAL_H_
#define _OPTIXRENDERER_ENCODED_NORMAL_H_

#include <OptiXRenderer/Defines.h>

#include <optixu/optixu_math_namespace.h>

#include <string>

namespace OptiXRenderer {

//----------------------------------------------------------------------------
// Encodes a normal by storing the sign of the z component in 
// the second most significant bit of the y component.
// The reason why this works is because the second most significant bit,
// or the most significant bit of the exponent, is only ever set when 
// a floating point value is larger than abs(2), which is never the case for 
// normals.
//----------------------------------------------------------------------------
class __align__(8) EncodedNormal {
private:
    float x;
    int y_bitmask_z_sign_y; // Contains the bitmask of the y component and the sign of z encoded in second most significant bit.

public:
    EncodedNormal(optix::float3 normal) {
        static_assert(sizeof(float) == sizeof(int), "Implementation needed for when float and int have different sizes.");
        x = normal.x;
        memcpy(&y_bitmask_z_sign_y, &normal.y, sizeof(float));
        y_bitmask_z_sign_y |= normal.z < 0.0f ? 0 : (1 << 30);
    }

    optix::float3 decode() const {
        optix::float3 res;
        res.x = x;
        int sign = y_bitmask_z_sign_y & (1 << 30);
        int y_bitmask = y_bitmask_z_sign_y & ~(1 << 30);
        memcpy(&res.y, &y_bitmask, sizeof(float));
        res.z = sqrt(optix::fmaxf(1.0f - res.x * res.x - res.y * res.y, 0.0f));
        res.z *= sign == 0 ? -1.0f : 1.0f;
        return res;
    }
};

} // NS OptiXRenderer

#endif // _OPTIXRENDERER_ENCODED_NORMAL_H_