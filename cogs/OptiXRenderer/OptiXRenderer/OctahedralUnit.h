// OptiX octahedral encoded normal.
// ---------------------------------------------------------------------------
// Copyright (C) 2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _OPTIXRENDERER_OCTAHEDRAL_UNIT_H_
#define _OPTIXRENDERER_OCTAHEDRAL_UNIT_H_

#include <OptiXRenderer/Defines.h>

#include <optixu/optixu_math_namespace.h>

namespace OptiXRenderer {

//-------------------------------------------------------------------------------------------------
// Encodes a unit vector using an octahedral representation and directx 11's fixed point encoding.
// See A Survey of Efficient Representations for Independent Unit Vectors, McGuire et al., 2014 for how to encode.
// See https://msdn.microsoft.com/en-us/library/windows/desktop/dd607323(v=vs.85).aspx#fixed_point_integer_conversion for how to convert a float to a fixed point snorm.
//-------------------------------------------------------------------------------------------------
struct __align__(4) OctahedralUnit32 {

    const static int INT16_MAX_VAL = 32767;

    optix::short2 encoding;

    __inline_all__ static float sign(float v) { return v >= 0.0f ? +1.0f : -1.0f; }
    __inline_all__ static optix::float2 sign(optix::float2 v) { return optix::make_float2(sign(v.x), sign(v.y)); }


    __inline_all__ static OctahedralUnit32 encode(optix::float3 n) {

        // Project the sphere onto the octahedron, and then onto the xy plane.
        optix::float2 p = optix::make_float2(n.x, n.y) / (abs(n.x) + abs(n.y) + abs(n.z));

        // Reflect the folds of the lower hemisphere over the diagonals.
        optix::float2 p2 = n.z < 0 ? (optix::make_float2(1.0f) - optix::make_float2(abs(p.y), abs(p.x))) * sign(p) : p;

        // Fixed point encoding.
        OctahedralUnit32 res = { optix::make_short2(short(optix::clamp(p2.x, -1.0f, 1.0f) * INT16_MAX_VAL + (p2.x < 0 ? -0.5f : 0.5f)),
                                                    short(optix::clamp(p2.y, -1.0f, 1.0f) * INT16_MAX_VAL + (p2.y < 0 ? -0.5f : 0.5f))) };
        return res;
    }

    __inline_all__ static OctahedralUnit32 encode_precise(optix::float3 n) {
        using namespace optix;

        // Project the sphere onto the octahedron, and then onto the xy plane.
        float2 p = make_float2(n.x, n.y) / (abs(n.x) + abs(n.y) + abs(n.z));

        // Reflect the folds of the lower hemisphere over the diagonals.
        float2 p2 = n.z < 0 ? (make_float2(1.0f) - make_float2(abs(p.y), abs(p.x))) * sign(p) : p;

        // Fixed point encoding.
        OctahedralUnit32 floored_oct = { make_short2(short(floor(clamp(p2.x, -1.0f, 1.0f) * INT16_MAX_VAL)),
                                                     short(floor(clamp(p2.y, -1.0f, 1.0f) * INT16_MAX_VAL))) };

        OctahedralUnit32 best_representation = floored_oct;
        float lowest_sqrd_mag = length(best_representation.decode() - n);

        auto best_oct_tester = [&](OctahedralUnit32 new_oct) {
            float m = length(new_oct.decode() - n);
            if (m < lowest_sqrd_mag) {
                best_representation = new_oct;
                lowest_sqrd_mag = m;
            }
        };

        OctahedralUnit32 upper_left = { make_short2(floored_oct.encoding.x, floored_oct.encoding.x + 1) };
        best_oct_tester(upper_left);
        OctahedralUnit32 lower_right = { make_short2(floored_oct.encoding.x + 1, floored_oct.encoding.x) };
        best_oct_tester(lower_right);
        OctahedralUnit32 upper_right = { make_short2(floored_oct.encoding.x + 1, floored_oct.encoding.x + 1) };
        best_oct_tester(upper_right);

        return best_representation;
    }

    __inline_all__ static OctahedralUnit32 encode_precise(float x, float y, float z) {
        return encode_precise(::optix::make_float3(x, y, z));
    }

    __inline_all__ optix::float3 decode() const {
        optix::float2 p2 = optix::make_float2(encoding.x, encoding.y);
        optix::float3 n = optix::make_float3(encoding.x, encoding.y, INT16_MAX_VAL - abs(p2.x) - abs(p2.y));
        if (n.z < 0.0f) {
            float tmp_x = (INT16_MAX_VAL - abs(n.y)) * sign(n.x);
            n.y = (INT16_MAX_VAL - abs(n.x)) * sign(n.y);
            n.x = tmp_x;
        }
        return optix::normalize(n);
    }

    __inline_all__ optix::float3 decode_unnormalized() const {
        optix::float2 p2 = optix::make_float2(encoding.x, encoding.y);
        optix::float3 n = optix::make_float3(encoding.x, encoding.y, INT16_MAX_VAL - abs(p2.x) - abs(p2.y));
        if (n.z < 0.0f) {
            float tmp_x = (INT16_MAX_VAL - abs(n.y)) * sign(n.x);
            n.y = (INT16_MAX_VAL - abs(n.x)) * sign(n.y);
            n.x = tmp_x;
        }
        return n;
    }
};

} // NS OptiXRenderer

#endif // _OPTIXRENDERER_OCTAHEDRAL_UNIT_H_