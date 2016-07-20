// Cogwheel functions for converting between math primitives.
// ----------------------------------------------------------------------------
// Copyright (C) 2015, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ----------------------------------------------------------------------------

#ifndef _COGWHEEL_MATH_CONVERSIONS_H_
#define _COGWHEEL_MATH_CONVERSIONS_H_

#include <Cogwheel/Math/Matrix.h>
#include <Cogwheel/Math/Quaternion.h>
#include <Cogwheel/Math/Transform.h>

namespace Cogwheel {
namespace Math {

// Converts a quaternion to it's 3x3 matrix representation.
// http://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToMatrix/index.htm
template <typename T>
inline Matrix3x3<T> to_matrix3x3(Quaternion<T> q) {
    const T one = T(1);
    const T two = T(2);
    const T x = q.x; const T y = q.y; const T z = q.z; const T w = q.w;
    Matrix3x3<T> res =
        { one - two*(y*y + z*z), two*(x*y - w*z), two*(x*z + w*y),
          two*(x*y + w*z), one - two*(x*x + z*z), two*(z*y - w*x),
          two*(x*z - w*y), two*(z*y + w*x), one - two*(x*x + y*y) };
    return res;
}

// Constructs a Quaternion from a rotation matrix.
// See the quaternion constructor in pbrt v3: Quaternion::ToTransform().
template <typename T>
inline Quaternion<T> to_quaternion(const Matrix3x3<T> m) {
    // Compute the trace of the matrix
    T trace = m[0][0] + m[1][1] + m[2][2];
    
    if (trace > T(0)) {
        // Compute w from matrix trace, then xyz
        // 4w^2 = m[0][0] + m[1][1] + m[2][2] + m[3][3] (but m[3][3] == 1)
        T s = std::sqrt(trace + T(1.0));
        T real = s * T(0.5);
        s = 0.5f / s;
        Vector3<T> imaginary = Vector3<T>(m[2][1] - m[1][2],
                                          m[0][2] - m[2][0],
                                          m[1][0] - m[0][1]) * s;
        return Quaternion<T>(imaginary, real);
    } else {
        // Compute largest of x, y, or z, then remaining components
        const int nxt[3] = { 1, 2, 0 };
        int i = 0;
        if (m[1][1] > m[0][0]) i = 1;
        if (m[2][2] > m[i][i]) i = 2;
        int j = nxt[i];
        int k = nxt[j];
        T s = std::sqrt((m[i][i] - (m[j][j] + m[k][k])) + 1.0f);
        Vector3<T> imaginary;
        imaginary[i] = s * 0.5f;
        if (s != 0.f) s = 0.5f / s;
        T real = (m[k][j] - m[j][k]) * s;
        imaginary[j] = (m[j][i] + m[i][j]) * s;
        imaginary[k] = (m[k][i] + m[i][k]) * s;
        return Quaternion<T>(imaginary, real);
    }
}

inline Matrix4x3f to_matrix4x3(Transform t) {
    const Matrix3x3f r = to_matrix3x3(t.rotation);
    const float s = t.scale;
    Matrix4x3f res = { s * r[0][0], s * r[0][1], s * r[0][2], t.translation.x,
                       s * r[1][0], s * r[1][1], s * r[1][2], t.translation.y,
                       s * r[2][0], s * r[2][1], s * r[2][2], t.translation.z };
    return res;
}

inline Matrix4x4f to_matrix4x4(Transform t) {
    const Matrix3x3f r = to_matrix3x3(t.rotation);
    const float s = t.scale;
    Matrix4x4f res = { s * r[0][0], s * r[0][1], s * r[0][2], t.translation.x,
                       s * r[1][0], s * r[1][1], s * r[1][2], t.translation.y,
                       s * r[2][0], s * r[2][1], s * r[2][2], t.translation.z,
                       0.0f,        0.0f,        0.0f,        1.0f };
    return res;
}
} // NS Math
} // NS Cogwheel

#endif // _COGWHEEL_MATH_CONVERSIONS_H_