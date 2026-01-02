// Bifrost Vector abstraction.
// ----------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ----------------------------------------------------------------------------

#ifndef _BIFROST_MATH_VECTOR_H_
#define _BIFROST_MATH_VECTOR_H_

#include <Bifrost/Core/Defines.h>

#include <cstring>
#include <sstream>

namespace Bifrost {
namespace Math {

template <typename T>
struct Vector2 final {
public:
    template <typename T> using Vector = Vector2;
    typedef T value_type;
    static const int N = 2;

    //*****************************************************************************
    // Public members
    //*****************************************************************************
    T x;
    T y;

    Vector2() = default;
    explicit Vector2(T s) : x(s), y(s) { }
    Vector2(T x, T y) : x(x), y(y) { }
    template <typename U>
    explicit Vector2(const Vector2<U>& v) : x(T(v.x)), y(T(v.y)) { }

    static __always_inline__ Vector2<T> zero() { return Vector2<T>(0, 0); }
    static __always_inline__ Vector2<T> one() { return Vector2<T>(1, 1); }

    __always_inline__ T* begin() { return &x; }
    __always_inline__ const T* const begin() const { return &x; }
    __always_inline__ T* end() { return begin() + N; }
    __always_inline__ const T* const end() const { return begin() + N; }

    inline std::string to_string() const {
        std::ostringstream out;
        out << "[x: " << x << ", y: " << y << "]";
        return out.str();
    }

#include "VectorOperators.h"
};

template <typename T>
struct Vector3 final {
public:
    template <typename T> using Vector = Vector3;
    typedef T value_type;
    static const int N = 3;

    //*****************************************************************************
    // Public members
    //*****************************************************************************
    T x;
    T y;
    T z;

    Vector3() = default;
    explicit Vector3(T s) : x(s), y(s), z(s) { }
    Vector3(T x, T y, T z) : x(x), y(y), z(z) { }
    template <typename U>
    explicit Vector3(const Vector3<U>& v) : x(T(v.x)), y(T(v.y)), z(T(v.z)) { }
    Vector3(const Vector2<T> v, T z) : x(v.x), y(v.y), z(z) { }

    static __always_inline__ Vector3<T> zero() { return Vector3(0, 0, 0); }
    static __always_inline__ Vector3<T> one() { return Vector3(1, 1, 1); }

    static __always_inline__ Vector3<T> forward() { return Vector3(0, 0, 1); }
    static __always_inline__ Vector3<T> up() { return Vector3(0, 1, 0); }
    static __always_inline__ Vector3<T> right() { return Vector3(1, 0, 0); }

    __always_inline__ T* begin() { return &x; }
    __always_inline__ const T* const begin() const { return &x; }
    __always_inline__ T* end() { return begin() + N; }
    __always_inline__ const T* const end() const { return begin() + N; }

    inline std::string to_string() const {
        std::ostringstream out;
        out << "[x: " << x << ", y: " << y << ", z: " << z << "]";
        return out.str();
    }

#include "VectorOperators.h"
};

template <typename T>
struct Vector4 final {
public:
    template <typename T> using Vector = Vector4;
    typedef T value_type;
    static const int N = 4;

    //*****************************************************************************
    // Public members
    //*****************************************************************************
    T x;
    T y;
    T z;
    T w;

    Vector4() = default;
    explicit Vector4(T s) : x(s), y(s), z(s), w(s) { }
    Vector4(T x, T y, T z, T w) : x(x), y(y), z(z), w(w) { }
    template <typename U>
    Vector4(const Vector4<U> v) : x(T(v.x)), y(T(v.y)), z(T(v.z)), w(T(v.w)) { }
    Vector4(const Vector2<T> v, T z, T w) : x(v.x), y(v.y), z(z), w(w) { }
    Vector4(const Vector3<T> v, T w) : x(v.x), y(v.y), z(v.z), w(w) { }

    static __always_inline__ Vector4<T> zero() { return Vector4<T>(0, 0, 0, 0); }
    static __always_inline__ Vector4<T> one() { return Vector4<T>(1, 1, 1, 1); }

    __always_inline__ T* begin() { return &x; }
    __always_inline__ const T* const begin() const { return &x; }
    __always_inline__ T* end() { return begin() + N; }
    __always_inline__ const T* const end() const { return begin() + N; }

    inline std::string to_string() const {
        std::ostringstream out;
        out << "[x: " << x << ", y: " << y << ", z: " << z << ", w: " << w << "]";
        return out.str();
    }

#include "VectorOperators.h"
};

//*************************************************************************
// Utility methods for vectors.
//*************************************************************************

// Compute the dot product between two vectors.
template<template<typename> class Vector, typename T>
__always_inline__ T dot(Vector<T> lhs, Vector<T> rhs) {
    T res = lhs[0] * rhs[0];
    for (int i = 1; i < Vector<T>::N; ++i)
        res += lhs[i] * rhs[i];
    return res;
}

// Compute the squared magnitude of the input vector.
// Useful when comparing the relative size between vectors, where the exact magnitude isn't needed.
template<template<typename> class Vector, typename T>
__always_inline__ T magnitude_squared(Vector<T> v) {
    return dot(v, v);
}

// Compute the magnitude of the input vector.
template<template<typename> class Vector, typename T>
__always_inline__ T magnitude(Vector<T> v) {
    return (T)sqrt(dot(v, v));
}

// Create a normalized version of the input vector.
template<template<typename> class Vector, typename T>
__always_inline__ Vector<T> normalize(Vector<T> v){
    T m = magnitude(v);
    return v / m;
}

// Cross product between two 3-dimensional vectors.
template<typename T>
__always_inline__ Vector3<T> cross(Vector3<T> lhs, Vector3<T> rhs) {
    return Vector3<T>((lhs.y * rhs.z) - (lhs.z * rhs.y),
                      (lhs.z * rhs.x) - (lhs.x * rhs.z),
                      (lhs.x * rhs.y) - (lhs.y * rhs.x));
}

// Computes a tangent and bitangent that together with the normal creates an orthonormal basis.
// Building an Orthonormal Basis, Revisited, Duff et al.
// http://jcgt.org/published/0006/01/01/paper.pdf
template<typename T>
__always_inline__ void compute_tangents(Vector3<T> normal, Vector3<T>& tangent, Vector3<T>& bitangent) {
    T sign = T(copysignf(1.0f, (float)normal.z));
    T a = T(-1) / (sign + normal.z);
    T b = normal.x * normal.y * a;
    tangent = { T(1) + sign * normal.x * normal.x * a, sign * b, -sign * normal.x };
    bitangent = { b, sign + normal.y * normal.y * a, -normal.y };
}

template<typename T>
__always_inline__ Vector2<T> min(Vector2<T> lhs, Vector2<T> rhs) {
    return Vector2<T>(lhs.x > rhs.x ? rhs.x : lhs.x, lhs.y > rhs.y ? rhs.y : lhs.y);
}

template<typename T>
__always_inline__ Vector3<T> min(Vector3<T> lhs, Vector3<T> rhs) {
    return Vector3<T>(lhs.x > rhs.x ? rhs.x : lhs.x,
                      lhs.y > rhs.y ? rhs.y : lhs.y,
                      lhs.z > rhs.z ? rhs.z : lhs.z);
}

template<typename T>
__always_inline__ Vector4<T> min(Vector4<T> lhs, Vector4<T> rhs) {
    return Vector4<T>(lhs.x > rhs.x ? rhs.x : lhs.x,
                      lhs.y > rhs.y ? rhs.y : lhs.y,
                      lhs.z > rhs.z ? rhs.z : lhs.z,
                      lhs.w > rhs.w ? rhs.w : lhs.w);
}

template<typename T>
__always_inline__ Vector2<T> max(Vector2<T> lhs, Vector2<T> rhs) {
    return Vector2<T>(lhs.x < rhs.x ? rhs.x : lhs.x, lhs.y < rhs.y ? rhs.y : lhs.y);
}

template<typename T>
__always_inline__ Vector3<T> max(Vector3<T> lhs, Vector3<T> rhs) {
    return Vector3<T>(lhs.x < rhs.x ? rhs.x : lhs.x,
                      lhs.y < rhs.y ? rhs.y : lhs.y,
                      lhs.z < rhs.z ? rhs.z : lhs.z);
}

template<typename T>
__always_inline__ Vector4<T> max(Vector4<T> lhs, Vector4<T> rhs) {
    return Vector4<T>(lhs.x < rhs.x ? rhs.x : lhs.x,
                      lhs.y < rhs.y ? rhs.y : lhs.y,
                      lhs.z < rhs.z ? rhs.z : lhs.z,
                      lhs.w < rhs.w ? rhs.w : lhs.w);
}

// Comparison that checks if two vectors are almost equal.
template<typename T>
__always_inline__ bool almost_equal(Vector2<T> lhs, Vector2<T> rhs, unsigned short max_ulps = 4) {
    return almost_equal(lhs.x, rhs.x, max_ulps)
        && almost_equal(lhs.y, rhs.y, max_ulps);
}
template<typename T>
__always_inline__ bool almost_equal(Vector3<T> lhs, Vector3<T> rhs, unsigned short max_ulps = 4) {
    return almost_equal(lhs.x, rhs.x, max_ulps)
        && almost_equal(lhs.y, rhs.y, max_ulps)
        && almost_equal(lhs.z, rhs.z, max_ulps);
}
template<typename T>
__always_inline__ bool almost_equal(Vector4<T> lhs, Vector4<T> rhs, unsigned short max_ulps = 4) {
    return almost_equal(lhs.x, rhs.x, max_ulps)
        && almost_equal(lhs.y, rhs.y, max_ulps)
        && almost_equal(lhs.z, rhs.z, max_ulps)
        && almost_equal(lhs.w, rhs.w, max_ulps);
}

//*************************************************************************
// Typedefs.
//*************************************************************************

typedef Vector2<double> Vector2d;
typedef Vector2<float> Vector2f;
typedef Vector2<int> Vector2i;
typedef Vector2<short> Vector2s;
typedef Vector2<unsigned int> Vector2ui;
typedef Vector3<double> Vector3d;
typedef Vector3<float> Vector3f;
typedef Vector3<int> Vector3i;
typedef Vector3<unsigned int> Vector3ui;
typedef Vector4<double> Vector4d;
typedef Vector4<float> Vector4f;
typedef Vector4<int> Vector4i;

} // NS Math
} // NS Bifrost

// ------------------------------------------------------------------------------------------------
// Convenience functions that appends a vector's string representation to an ostream.
// ------------------------------------------------------------------------------------------------
template<class T>
__always_inline__ std::ostream& operator<<(std::ostream& s, Bifrost::Math::Vector2<T> v){
    return s << v.to_string();
}

template<class T>
__always_inline__ std::ostream& operator<<(std::ostream& s, Bifrost::Math::Vector3<T> v){
    return s << v.to_string();
}

template<class T>
__always_inline__ std::ostream& operator<<(std::ostream& s, Bifrost::Math::Vector4<T> v){
    return s << v.to_string();
}

// ------------------------------------------------------------------------------------------------
// Math operator overloading.
// ------------------------------------------------------------------------------------------------

template<class T>
__always_inline__ Bifrost::Math::Vector2<T> operator+(T lhs, Bifrost::Math::Vector2<T> rhs) {
    return rhs + lhs;
}

template<class T>
__always_inline__ Bifrost::Math::Vector3<T> operator+(T lhs, Bifrost::Math::Vector3<T> rhs) {
    return rhs + lhs;
}

template<class T>
__always_inline__ Bifrost::Math::Vector4<T> operator+(T lhs, Bifrost::Math::Vector4<T> rhs) {
    return rhs + lhs;
}

template<class T>
__always_inline__ Bifrost::Math::Vector2<T> operator-(T lhs, Bifrost::Math::Vector2<T> rhs) {
    return Bifrost::Math::Vector2<T>(lhs - rhs.x, lhs - rhs.y);
}

template<class T>
__always_inline__ Bifrost::Math::Vector3<T> operator-(T lhs, Bifrost::Math::Vector3<T> rhs) {
    return Bifrost::Math::Vector3<T>(lhs - rhs.x, lhs - rhs.y, lhs - rhs.z);
}

template<class T>
__always_inline__ Bifrost::Math::Vector4<T> operator-(T lhs, Bifrost::Math::Vector4<T> rhs) {
    return Bifrost::Math::Vector4<T>(lhs - rhs.x, lhs - rhs.y, lhs - rhs.z, lhs - rhs.w);
}

template<class T>
__always_inline__ Bifrost::Math::Vector2<T> operator*(T lhs, Bifrost::Math::Vector2<T> rhs) {
    return rhs * lhs;
}

template<class T>
__always_inline__ Bifrost::Math::Vector3<T> operator*(T lhs, Bifrost::Math::Vector3<T> rhs) {
    return rhs * lhs;
}

template<class T>
__always_inline__ Bifrost::Math::Vector4<T> operator*(T lhs, Bifrost::Math::Vector4<T> rhs) {
    return rhs * lhs;
}

template<class T>
__always_inline__ Bifrost::Math::Vector2<T> operator/(T lhs, Bifrost::Math::Vector2<T> rhs) {
    return Bifrost::Math::Vector2<T>(lhs / rhs.x, lhs / rhs.y);
}

template<class T>
__always_inline__ Bifrost::Math::Vector3<T> operator/(T lhs, Bifrost::Math::Vector3<T> rhs) {
    return Bifrost::Math::Vector3<T>(lhs / rhs.x, lhs / rhs.y, lhs / rhs.z);
}

template<class T>
__always_inline__ Bifrost::Math::Vector4<T> operator/(T lhs, Bifrost::Math::Vector4<T> rhs) {
    return Bifrost::Math::Vector4<T>(lhs / rhs.x, lhs / rhs.y, lhs / rhs.z, lhs / rhs.w);
}

#endif // _BIFROST_MATH_VECTOR_H_
