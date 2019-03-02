// Bifrost Vector abstraction.
// ----------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ----------------------------------------------------------------------------

#ifndef _BIFROST_MATH_QUATERNION_H_
#define _BIFROST_MATH_QUATERNION_H_

#include <Bifrost/Core/Defines.h>
#include <Bifrost/Math/Constants.h>
#include <Bifrost/Math/Utils.h>
#include <Bifrost/Math/Vector.h>

#include <algorithm>
#include <cstring>
#include <cmath>
#include <sstream>

namespace Bifrost {
namespace Math {

//----------------------------------------------------------------------------
// Implementation of a quaternion for rotations in 3D.
// http://en.wikipedia.org/wiki/Quaternion
// w holds the real part, (x,y,z) holds the imaginary part.
//
// For Bifrost the only real usage we have for quaternions are as rotations and as
// such they should always be normalized. We therefore do not expose any
// functions that can return a non-normalized quaternion, within the limits of
// floating point precision.
//----------------------------------------------------------------------------
template <typename T>
struct Quaternion final {
public:
    typedef T value_type;
    static const int N = 4;

    //*****************************************************************************
    // Public members
    //*****************************************************************************
    T x;
    T y;
    T z;
    T w;

    // Default constructor.
    Quaternion() = default;

    // Constructor. Assumes that the parameters will create a normalized quaternion. Otherwise the user should normalize after construction.
    Quaternion(T x, T y, T z, T w)
        : x(x), y(y), z(z), w(w) { }

    // Constructs a quaternion from real and imaginary components.
    Quaternion(Vector3<T> v, T w)
        : x(v.x), y(v.y), z(v.z), w(w){}

    // Cast constructor.
    template <typename U>
    Quaternion(const Quaternion<U>& v) : x(T(v.x)), y(T(v.y)), z(T(v.z)), w(T(v.w)) { }

    // The identity quaternion.
    static __always_inline__ Quaternion<T> identity() {
        return Quaternion(0, 0, 0, 1);
    }

    // Create a quaternion describing a rotation in angles around a normalized axis in R^3.
    // http://www.euclideanspace.com/maths/geometry/rotations/conversions/angleToQuaternion/
    static __always_inline__ Quaternion<T> from_angle_axis(T angle_in_radians, Vector3<T> axis) {
        T radian_halved = angle_in_radians * T(0.5);
        T sin_angle = sin(radian_halved);
        Vector3<T> imaginary = axis * sin_angle;
        T real = cos(radian_halved);
        return Quaternion<T>(imaginary, real);
    }

    // Create a quaternion with forward pointing along direction and that has the upvector up.
    // See the quaternion constructor in pbrt v3: Quaternion::ToTransform().
    static inline Quaternion<T> look_in(Vector3<T> direction, Vector3<T> up = Vector3<T>::up()) {
        Vector3<T> right = normalize(cross(up, direction));
        up = cross(direction, right);

        // Compute the trace of the [right, up, dir] matrix.
        T trace = right.x + up.y + direction.z;

        if (trace > T(0)) {
            // Compute w from matrix trace, then xyz.
            // 4w^2 = m[0][0] + m[1][1] + m[2][2] + m[3][3] (but m[3][3] == 1)
            T s = std::sqrt(trace + T(1.0));
            T real = s * T(0.5);
            s = 0.5f / s;
            Vector3<T> imaginary = Vector3<T>(up.z - direction.y,
                                              direction.x - right.z,
                                              right.y - up.x) * s;
            return Quaternion<T>(imaginary, real);
        } else {
            Vector3<T> m[] = { right, up, direction };

            // Compute largest of x, y or z, then remaining components.
            const int next[3] = { 1, 2, 0 };
            int i = 0;
            if (m[1][1] > m[0][0]) i = 1;
            if (m[2][2] > m[i][i]) i = 2;
            int j = next[i];
            int k = next[j];
            T s = std::sqrt((m[i][i] - (m[j][j] + m[k][k])) + T(1.0));
            Vector3<T> imaginary;
            imaginary[i] = s * T(0.5);
            if (s != T(0)) s = T(0.5) / s;
            T real = (m[j][k] - m[k][j]) * s;
            imaginary[j] = (m[i][j] + m[j][i]) * s;
            imaginary[k] = (m[i][k] + m[k][i]) * s;
            return Quaternion<T>(imaginary, real);
        }
    }

    //*****************************************************************************
    // Comparison operators.
    //*****************************************************************************
    __always_inline__ bool operator==(Quaternion<T> rhs) const {
        return memcmp(this, &rhs, sizeof(rhs)) == 0;
    }
    __always_inline__ bool operator!=(Quaternion<T> rhs) const {
        return memcmp(this, &rhs, sizeof(rhs)) != 0;
    }

    // Pointer to the first element of the quaternion.
    __always_inline__ T* begin() { return &x; }
    __always_inline__ const T* begin() const { return &x; }

    // The imaginary part of the quaternion.
    __always_inline__ Vector3<T> imaginary() const { return Vector3<T>(x, y, z); }

    // The real part of the quaternion.
    __always_inline__ T real() const { return w; }

    // Quaternion multiplication.
    __always_inline__ Quaternion<T> operator*(Quaternion<T> rhs) const {
        T real_part = w * rhs.w - dot(imaginary(), rhs.imaginary());
        Vector3<T> imaginary_part = cross(imaginary(), rhs.imaginary()) + rhs.imaginary() * w + imaginary() * rhs.w;
        return Quaternion(imaginary_part, real_part);
    }
    __always_inline__ Quaternion<T>& operator*=(Quaternion<T> rhs) {
        Quaternion<T> r = *this * rhs;
        x = r.x; y = r.y; z = r.z;
        w = r.w;
        return *this;
    }

    // Multiplying a vector by a quaternion, e.g. rotating it.
    __always_inline__ Vector3<T> operator*(Vector3<T> rhs) const {
        Vector3<T> img = imaginary();
        Vector3<T> uv = cross(img, rhs);
        Vector3<T> uuv = cross(img, uv);

        Vector3<T> half_res = (uv * w) + uuv;
        return rhs + half_res * T(2);
    }

    // The forward vector of the rotation.
    // The same as rotating Vector3::forward() by the quaternion.
    __always_inline__ Vector3<T> forward() const {
        return *this * Vector3<T>::forward();
    }

    // The up vector of the rotation.
    // The same as rotating Vector3::up() by the quaternion.
    __always_inline__ Vector3<T> up() const {
        return *this * Vector3<T>::up();
    }

    // The right vector of the rotation.
    // The same as rotating Vector3::right() by the quaternion.
    __always_inline__ Vector3<T> right() const {
        return *this * Vector3<T>::right();
    }

    // To string.
    inline std::string to_string() const {
        std::ostringstream out;
        out << "[v: [x: " << x << ", y: " << y << ", z: " << z << "], w: " << w << "]";
        return out.str();
    }
};

// Returns the conjugate of the quaternion.
template <typename T>
__always_inline__ Quaternion<T> conjugate(Quaternion<T> v) {
    return Quaternion<T>(-v.x, -v.y, -v.z, v.w);
}

// The inverse of a unit quaternion.
template <typename T>
__always_inline__ Quaternion<T> inverse_unit(Quaternion<T> v) {
    return conjugate(v);
}

// Dot product between two quaternions.
template<typename T>
__always_inline__ T dot(Quaternion<T> lhs, Quaternion<T> rhs) {
    return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z + lhs.w * rhs.w;
}

// Computes the magnitude of a quaternion.
template<typename T>
__always_inline__ T magnitude(Quaternion<T> v) {
    return sqrt(dot(v, v));
}

// Returns the input quaternion as normalized quaternion.
template<typename T>
__always_inline__ Quaternion<T> normalize(Quaternion<T> v) {
    T m = magnitude(v);
    return Quaternion<T>(v.x / m, v.y / m, v.z / m, v.w / m);
}

// Lerps between two quaternions and normalizes the result.
template<typename T>
__always_inline__ Quaternion<T> nlerp(Quaternion<T> from, Quaternion<T> to, T by) {
    return Quaternion<T>(lerp(from.x, to.x, by),
                         lerp(from.y, to.y, by),
                         lerp(from.z, to.z, by),
                         lerp(from.w, to.w, by)).normalize();
}

// Comparison that checks if two quaternions are almost equal.
template<typename T>
__always_inline__ bool almost_equal(Quaternion<T> lhs, Quaternion<T> rhs, unsigned short max_ulps = 4) {
    return almost_equal(lhs.x, rhs.x, max_ulps)
        && almost_equal(lhs.y, rhs.y, max_ulps)
        && almost_equal(lhs.z, rhs.z, max_ulps)
        && almost_equal(lhs.w, rhs.w, max_ulps);
}

// Comparison that checks if two quaternions interpreted as rotations in R3 are almost equal.
template<typename T>
__always_inline__ bool almost_equal_rotation(Quaternion<T> lhs, Quaternion<T> rhs, unsigned short max_ulps = 4) {
    // A quaternion as rotation is conceptually equal to the same quaternion with all elements negated.
    // Therefore if the left and right quaternion's real component do not have the same sign, we negate rhs.
    if (lhs.w * rhs.w < T(0))
        rhs = Quaternion<T>(-rhs.x, -rhs.y, -rhs.z, -rhs.w);

    return almost_equal(lhs.x, rhs.x, max_ulps)
        && almost_equal(lhs.y, rhs.y, max_ulps)
        && almost_equal(lhs.z, rhs.z, max_ulps)
        && almost_equal(lhs.w, rhs.w, max_ulps);
}


//*****************************************************************************
// Typedefs.
//*****************************************************************************

typedef Quaternion<float> Quaternionf;
typedef Quaternion<double> Quaterniond;

} // NS Math
} // NS Bifrost

// Convenience function that appends a quaternion's string representation to an ostream.
template<class T>
__always_inline__ std::ostream& operator<<(std::ostream& s, Bifrost::Math::Quaternion<T> v){
    return s << v.to_string();
}

#endif // _BIFROST_MATH_QUATERNION_H_
