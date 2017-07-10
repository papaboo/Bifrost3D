// Octahedral encoded unit vector.
// ------------------------------------------------------------------------------------------------
// Copyright (C) 2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _COGWHEEL_MATH_OCTAHEDRAL_NORMAL_H_
#define _COGWHEEL_MATH_OCTAHEDRAL_NORMAL_H_

#include <Cogwheel/Math/Vector.h>
#include <Cogwheel/Math/Utils.h>

#include <limits.h>

namespace Cogwheel {
namespace Math {

//-------------------------------------------------------------------------------------------------
// Encodes a unit vector using an octahedral representation and directx 11's fixed point encoding.
// See A Survey of Efficient Representations for Independent Unit Vectors, McGuire et al., 2014 for how to encode.
// See https://msdn.microsoft.com/en-us/library/windows/desktop/dd607323(v=vs.85).aspx#fixed_point_integer_conversion for how to convert a float to a fixed point snorm.
//-------------------------------------------------------------------------------------------------
struct OctahedralNormal {
private:
    static float sign(float v) { return v >= 0.0f ? +1.0f : -1.0f; }
    static Vector2f sign(Vector2f v) { return Vector2f(sign(v.x), sign(v.y)); }

public:

    Vector2s encoding;

    static OctahedralNormal encode(Vector3f n) {

        // Project the sphere onto the octahedron, and then onto the xy plane.
        Vector2f p = Vector2f(n.x, n.y) / (abs(n.x) + abs(n.y) + abs(n.z));

        // Reflect the folds of the lower hemisphere over the diagonals.
        Vector2f p2 = n.z < 0 ? (Vector2f(1.0f) - Vector2f(abs(p.y), abs(p.x))) * sign(p) : p;

        // Fixed point encoding.
        OctahedralNormal res = { Vector2s(short(clamp(p2.x, -1.0f, 1.0f) * SHRT_MAX + (p2.x < 0 ? -0.5f : 0.5f)),
                                          short(clamp(p2.y, -1.0f, 1.0f) * SHRT_MAX + (p2.y < 0 ? -0.5f : 0.5f))) };
        return res;
    }

    static OctahedralNormal encode(float x, float y, float z) {
        return encode(Vector3f(x, y, z));
    }

    static OctahedralNormal encode_precise(Vector3f n) {
        // Project the sphere onto the octahedron, and then onto the xy plane.
        Vector2f p = Vector2f(n.x, n.y) / (abs(n.x) + abs(n.y) + abs(n.z));

        // Reflect the folds of the lower hemisphere over the diagonals.
        Vector2f p2 = n.z < 0 ? (Vector2f(1.0f) - Vector2f(abs(p.y), abs(p.x))) * sign(p) : p;

        // Fixed point encoding.
        OctahedralNormal floored_oct = { Vector2s(short(floor(clamp(p2.x, -1.0f, 1.0f) * SHRT_MAX)),
                                                  short(floor(clamp(p2.y, -1.0f, 1.0f) * SHRT_MAX))) };

        OctahedralNormal best_representation = floored_oct;
        float lowest_sqrd_mag = squared_magnitude(best_representation.decode() - n);

        auto best_oct_tester = [&](OctahedralNormal new_oct) {
            float m = squared_magnitude(new_oct.decode() - n);
            if (m < lowest_sqrd_mag) {
                best_representation = new_oct;
                lowest_sqrd_mag = m;
            }
        };

        OctahedralNormal upper_left = { floored_oct.encoding + Vector2s(0, 1) };
        best_oct_tester(upper_left);
        OctahedralNormal lower_right = { floored_oct.encoding + Vector2s(1, 0) };
        best_oct_tester(lower_right);
        OctahedralNormal upper_right = { floored_oct.encoding + Vector2s(1, 1) };
        best_oct_tester(upper_right);

        return best_representation;
    }

    static OctahedralNormal encode_precise(float x, float y, float z) {
        return encode_precise(Vector3f(x, y, z));
    }

    Vector3f decode() const {
        Vector2f p2 = Vector2f(encoding);
        Vector3f n = Vector3f(p2, SHRT_MAX - abs(p2.x) - abs(p2.y));
        if (n.z < 0.0f) {
            float tmp_x = (SHRT_MAX - abs(n.y)) * sign(n.x);
            n.y = (SHRT_MAX - abs(n.x)) * sign(n.y);
            n.x = tmp_x;
        }
        return normalize(n);
    }
};

} // NS Math
} // NS Cogwheel

#endif // _COGWHEEL_MATH_OCTAHEDRAL_NORMAL_H_