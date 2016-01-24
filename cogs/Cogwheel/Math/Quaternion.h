// Cogwheel Vector abstraction.
// ----------------------------------------------------------------------------
// Copyright (C) 2015, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ----------------------------------------------------------------------------

#ifndef _COGWHEEL_MATH_QUATERNION_H_
#define _COGWHEEL_MATH_QUATERNION_H_

#include <Math/Constants.h>
#include <Math/Utils.h>
#include <Math/Vector.h>

#include <algorithm>
#include <cstring>
#include <cmath>
#include <sstream>

namespace Cogwheel {
namespace Math {

//----------------------------------------------------------------------------
// Implementation of a quaternion for rotations in 3D.
// http://en.wikipedia.org/wiki/Quaternion
// w holds the real part, (x,y,z) holds the imaginary part.
//
// For Cogwheel the only real usage we have for quaternions are as rotations and as
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

    // Constructor. Assumes that the parameters will create a normalized quaternion. Otherwise the user should normalized after construction.
    Quaternion(T x, T y, T z, T w)
        : x(x), y(y), z(z), w(w) { }

    // Constructs a quaternion from real and imaginary components.
    Quaternion(Vector3<T> v, T w)
        : x(v.x), y(v.y), z(v.z), w(w){}

    // The identity quaternion.
    static inline Quaternion<T> identity() {
        return Quaternion(0, 0, 0, 1);
    }

    // Create a quaternion describing a rotation in angles around a normalized axis in R^3.
    // http://www.euclideanspace.com/maths/geometry/rotations/conversions/angleToQuaternion/
    static inline Quaternion<T> from_angle_axis(T angle_in_radians, Vector3<T> axis) {
        T radian_halved = angle_in_radians * T(0.5); // TODO really? Halve it?
        T sin_angle = sin(radian_halved);
        Vector3<T> imaginary = axis * sin_angle;
        T real = cos(radian_halved);
        return Quaternion<T>(imaginary, real);
    }

    // Create a quaternion that looks in direction and has the upvector up.
    // http://www.gamedev.net/topic/613595-quaternion-lookrotationlookat-up/
    static inline Quaternion<T> look_in(Vector3<T> direction, Vector3<T> up = Vector3<T>::up()) {
        Vector3<T> right = normalize(cross(up, direction));
        up = cross(direction, right);

        float real = sqrt(T(1) + right.x + up.y + direction.z) * T(0.5);
        float d = T(1) / (T(4) * real);
        Vector3<T> imaginary = Vector3<T>(up.z - direction.y,
                                          direction.x - right.z,
                                          right.y - up.x) * d;

        return Quaternion<T>(imaginary, real);
    }

    //*****************************************************************************
    // Comparison operators.
    //*****************************************************************************
    inline bool operator==(Quaternion<T> rhs) const {
        return memcmp(this, &rhs, sizeof(rhs)) == 0;
    }
    inline bool operator!=(Quaternion<T> rhs) const {
        return memcmp(this, &rhs, sizeof(rhs)) != 0;
    }

    // Pointer to the first element of the quaternion.
    inline T* begin() { return &x; }
    inline const T* begin() const { return &x; }

    // The imaginary part of the quaternion.
    inline Vector3<T>& imaginary() {
        return *static_cast<Vector3<T>*>(static_cast<void*>(&x));
    }
    // The imaginary part of the quaternion.
    inline Vector3<T> imaginary() const {
        return Vector3<T>(x, y, z);
    }

    // The real part of the quaternion.
    inline T& real() { return w; }
    inline T real() const { return w; }

    // Quaternion multiplication.
    inline Quaternion<T> operator*(const Quaternion<T>& rhs) const {
        // TODO Test new and old implementation for speed. Commented version is written in Rust.
        // Quaternion::new(self.w * rhs.w - self.i * rhs.i - self.j * rhs.j - self.k * rhs.k,
        //                 self.w * rhs.i + self.i * rhs.w + self.j * rhs.k - self.k * rhs.j,
        //                 self.w * rhs.j + self.j * rhs.w + self.k * rhs.i - self.i * rhs.k,
        //                 self.w * rhs.k + self.k * rhs.w + self.i * rhs.j - self.j * rhs.i)
        float real_part = w * rhs.w - dot(imaginary(), rhs.imaginary());
        Vector3<T> imaginary_part = cross(imaginary(), rhs.imaginary()) + rhs.imaginary() * w + imaginary() * rhs.w;
        return Quaternion(imaginary_part, real_part);
    }

    // Multiplying a vector by a quaternion, e.g. rotating it.
    inline Vector3<T> operator*(Vector3<T> rhs) const {
        Vector3<T> uv = cross(imaginary(), rhs);
        Vector3<T> uuv = cross(imaginary(), uv);

        Vector3<T> half_res = (uv * w) + uuv;
        return rhs + half_res * T(2);
    }
        
    // The forward vector of the rotation.
    // The same as rotating Vector3::forward() by the quaternion.
    inline Vector3<T> forward() const {
        return *this * Vector3f::forward();
    }

    // The up vector of the rotation.
    // The same as rotating Vector3::up() by the quaternion.
    inline Vector3<T> up() const {
        return *this * Vector3f::up();
    }

    // The right vector of the rotation.
    // The same as rotating Vector3::right() by the quaternion.
    inline Vector3<T> right() const {
        return *this * Vector3f::right();
    }

    // To string.
    const std::string to_string() const {
        std::ostringstream out;
        out << "[v: [x: " << x << ", y: " << y << ", z: " << z << "], w: " << w << "]";
        return out.str();
    }
};

// Returns the conjugate of the quaternion.
template <typename T>
inline Quaternion<T> conjugate(Quaternion<T> v) {
    return Quaternion<T>(-v.x, -v.y, -v.z, v.w);
}

// The inverse of a unit quaternion.
template <typename T>
inline Quaternion<T> inverse_unit(Quaternion<T> v) {
    return conjugate(v);
}

// Dot product between two quaternions.
template<typename T>
inline T dot(Quaternion<T> lhs, Quaternion<T> rhs) {
    return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z + lhs.w * rhs.w;
}

// Computes the magnitude of a quaternion.
template<typename T>
inline T magnitude(Quaternion<T> v) {
    return sqrt(dot(v, v));
}

// Returns the input quaternion as normalized quaternion.
template<typename T>
inline Quaternion<T> normalize(Quaternion<T> v) {
    T m = magnitude(v);
    return Quaternion<T>(v.x / m, v.y / m, v.z / m, v.w / m);
}

// Lerps between two quaternions and normalizes the result.
template<typename T>
inline Quaternion<T> nlerp(Quaternion<T> from, Quaternion<T> to, T by) {
    return Quaternion<T>(lerp(from.x, to.x, by),
                         lerp(from.y, to.y, by),
                         lerp(from.z, to.z, by),
                         lerp(from.w, to.w, by)).normalize();
}

// Comparison that checks if two quaternions are almost equal.
template<typename T>
inline bool almost_equal(Quaternion<T> lhs, Quaternion<T> rhs, unsigned short max_ulps = 4) {
    return almost_equal(lhs.x, rhs.x, max_ulps)
        && almost_equal(lhs.y, rhs.y, max_ulps)
        && almost_equal(lhs.z, rhs.z, max_ulps)
        && almost_equal(lhs.w, rhs.w, max_ulps);
}

// Comparison that checks if two quaternions interpreted as rotations in R3 are almost equal.
template<typename T>
inline bool almost_equal_rotation(Quaternion<T> lhs, Quaternion<T> rhs, unsigned short max_ulps = 4) {
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
} // NS Cogwheel

// Convenience function that appends a quaternion's string representation to an ostream.
template<class T>
inline std::ostream& operator<<(std::ostream& s, Cogwheel::Math::Quaternion<T> v){
    return s << v.to_string();
}

#endif // _COGWHEEL_MATH_QUATERNION_H_