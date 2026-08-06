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

#ifndef GPU_COMPILATION
#include <sstream>
#endif

namespace Bifrost::Math {

template <typename T>
struct alignas(2 * sizeof(T)) Vector2 final {
public:
    template <typename TT> using Vector = Vector2;
    typedef T value_type;
    static const int N = 2;

    //*****************************************************************************
    // Public members
    //*****************************************************************************
    T x;
    T y;

    Vector2() = default;
    GPU_ENABLED explicit Vector2(T s) : x(s), y(s) { }
    GPU_ENABLED Vector2(T x, T y) : x(x), y(y) { }
    template <typename U>
    GPU_ENABLED explicit Vector2(const Vector2<U>& v) : x(T(v.x)), y(T(v.y)) { }

    static __always_inline__ GPU_ENABLED Vector2<T> zero() { return Vector2<T>(0, 0); }
    static __always_inline__ GPU_ENABLED Vector2<T> one() { return Vector2<T>(1, 1); }

    __always_inline__ GPU_ENABLED T* begin() { return &x; }
    __always_inline__ GPU_ENABLED const T* begin() const { return &x; }
    __always_inline__ GPU_ENABLED T* end() { return begin() + N; }
    __always_inline__ GPU_ENABLED const T* end() const { return begin() + N; }

#ifndef GPU_COMPILATION
    inline std::string to_string() const {
        std::ostringstream out;
        out << "[x: " << x << ", y: " << y << "]";
        return out.str();
    }
#endif

#include "VectorOperators.h"
};

template <typename T>
struct Vector3 final {
public:
    template <typename TT> using Vector = Vector3;
    typedef T value_type;
    static const int N = 3;

    //*****************************************************************************
    // Public members
    //*****************************************************************************
    T x;
    T y;
    T z;

    Vector3() = default;
    GPU_ENABLED explicit Vector3(T s) : x(s), y(s), z(s) { }
    GPU_ENABLED Vector3(T x, T y, T z) : x(x), y(y), z(z) { }
    template <typename U>
    GPU_ENABLED explicit Vector3(const Vector3<U>& v) : x(T(v.x)), y(T(v.y)), z(T(v.z)) { }
    GPU_ENABLED Vector3(const Vector2<T> v, T z) : x(v.x), y(v.y), z(z) { }

    static __always_inline__ GPU_ENABLED Vector3<T> zero() { return Vector3(0, 0, 0); }
    static __always_inline__ GPU_ENABLED Vector3<T> one() { return Vector3(1, 1, 1); }

    static __always_inline__ GPU_ENABLED Vector3<T> forward() { return Vector3(0, 0, 1); }
    static __always_inline__ GPU_ENABLED Vector3<T> up() { return Vector3(0, 1, 0); }
    static __always_inline__ GPU_ENABLED Vector3<T> right() { return Vector3(1, 0, 0); }

    __always_inline__ GPU_ENABLED T* begin() { return &x; }
    __always_inline__ GPU_ENABLED const T* begin() const { return &x; }
    __always_inline__ GPU_ENABLED T* end() { return begin() + N; }
    __always_inline__ GPU_ENABLED const T* end() const { return begin() + N; }

#ifndef GPU_COMPILATION
    inline std::string to_string() const {
        std::ostringstream out;
        out << "[x: " << x << ", y: " << y << ", z: " << z << "]";
        return out.str();
    }
#endif

#include "VectorOperators.h"
};

template <typename T>
struct alignas(4 * sizeof(T)) Vector4 final {
public:
    template <typename TT> using Vector = Vector4;
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
    GPU_ENABLED explicit Vector4(T s) : x(s), y(s), z(s), w(s) { }
    GPU_ENABLED Vector4(T x, T y, T z, T w) : x(x), y(y), z(z), w(w) { }
    template <typename U>
    GPU_ENABLED Vector4(const Vector4<U> v) : x(T(v.x)), y(T(v.y)), z(T(v.z)), w(T(v.w)) { }
    GPU_ENABLED Vector4(const Vector2<T> v, T z, T w) : x(v.x), y(v.y), z(z), w(w) { }
    GPU_ENABLED Vector4(const Vector3<T> v, T w) : x(v.x), y(v.y), z(v.z), w(w) { }

    static __always_inline__ GPU_ENABLED Vector4<T> zero() { return Vector4<T>(0, 0, 0, 0); }
    static __always_inline__ GPU_ENABLED Vector4<T> one() { return Vector4<T>(1, 1, 1, 1); }

    __always_inline__ GPU_ENABLED T* begin() { return &x; }
    __always_inline__ GPU_ENABLED const T* begin() const { return &x; }
    __always_inline__ GPU_ENABLED T* end() { return begin() + N; }
    __always_inline__ GPU_ENABLED const T* end() const { return begin() + N; }

#ifndef GPU_COMPILATION
    inline std::string to_string() const {
        std::ostringstream out;
        out << "[x: " << x << ", y: " << y << ", z: " << z << ", w: " << w << "]";
        return out.str();
    }
#endif

#include "VectorOperators.h"
};

//*************************************************************************
// Utility methods for vectors.
//*************************************************************************

// Compute the dot product between two vectors.
template<template<typename> class Vector, typename T>
__always_inline__ GPU_ENABLED T dot(Vector<T> lhs, Vector<T> rhs) {
    T res = lhs[0] * rhs[0];
    for (int i = 1; i < Vector<T>::N; ++i)
        res += lhs[i] * rhs[i];
    return res;
}

// Compute the squared magnitude of the input vector.
// Useful when comparing the relative size between vectors, where the exact magnitude isn't needed.
template<template<typename> class Vector, typename T>
__always_inline__ GPU_ENABLED T magnitude_squared(Vector<T> v) { return dot(v, v); }

// Compute the magnitude of the input vector.
template<template<typename> class Vector, typename T>
__always_inline__ GPU_ENABLED T magnitude(Vector<T> v) { return (T)sqrt(dot(v, v)); }

// Compute the distance squared between the two points.
template<template<typename> class Vector, typename T>
_inline_all_archs_ T distance_squared(Vector<T> p0, Vector<T> p1) { return magnitude_squared(p0 - p1); }

// Compute the distance between the two points.
template<template<typename> class Vector, typename T>
_inline_all_archs_ T distance(Vector<T> p0, Vector<T> p1) { return magnitude(p0 - p1); }

// Create a normalized version of the input vector.
template<template<typename> class Vector, typename T>
__always_inline__ GPU_ENABLED Vector<T> normalize(Vector<T> v) { return v / magnitude(v); }

// Cross product between two 3-dimensional vectors.
template<typename T>
__always_inline__ GPU_ENABLED Vector3<T> cross(Vector3<T> lhs, Vector3<T> rhs) {
    return Vector3<T>((lhs.y * rhs.z) - (lhs.z * rhs.y),
                      (lhs.z * rhs.x) - (lhs.x * rhs.z),
                      (lhs.x * rhs.y) - (lhs.y * rhs.x));
}

// Computes a tangent and bitangent that together with the normal creates an orthonormal basis.
// Building an Orthonormal Basis, Revisited, Duff et al.
// http://jcgt.org/published/0006/01/01/paper.pdf
template<typename T>
__always_inline__ GPU_ENABLED void compute_tangents(Vector3<T> normal, Vector3<T>& tangent, Vector3<T>& bitangent) {
    T sign = T(copysignf(1.0f, (float)normal.z));
    T a = T(-1) / (sign + normal.z);
    T b = normal.x * normal.y * a;
    tangent = { T(1) + sign * normal.x * normal.x * a, sign * b, -sign * normal.x };
    bitangent = { b, sign + normal.y * normal.y * a, -normal.y };
}

template<typename T>
__always_inline__ GPU_ENABLED Vector2<T> min(Vector2<T> lhs, Vector2<T> rhs) {
    return Vector2<T>(lhs.x > rhs.x ? rhs.x : lhs.x, lhs.y > rhs.y ? rhs.y : lhs.y);
}

template<typename T>
__always_inline__ GPU_ENABLED Vector3<T> min(Vector3<T> lhs, Vector3<T> rhs) {
    return Vector3<T>(lhs.x > rhs.x ? rhs.x : lhs.x,
                      lhs.y > rhs.y ? rhs.y : lhs.y,
                      lhs.z > rhs.z ? rhs.z : lhs.z);
}

template<typename T>
__always_inline__ GPU_ENABLED Vector4<T> min(Vector4<T> lhs, Vector4<T> rhs) {
    return Vector4<T>(lhs.x > rhs.x ? rhs.x : lhs.x,
                      lhs.y > rhs.y ? rhs.y : lhs.y,
                      lhs.z > rhs.z ? rhs.z : lhs.z,
                      lhs.w > rhs.w ? rhs.w : lhs.w);
}

template<typename T>
__always_inline__ GPU_ENABLED Vector2<T> max(Vector2<T> lhs, Vector2<T> rhs) {
    return Vector2<T>(lhs.x < rhs.x ? rhs.x : lhs.x, lhs.y < rhs.y ? rhs.y : lhs.y);
}

template<typename T>
__always_inline__ GPU_ENABLED Vector3<T> max(Vector3<T> lhs, Vector3<T> rhs) {
    return Vector3<T>(lhs.x < rhs.x ? rhs.x : lhs.x,
                      lhs.y < rhs.y ? rhs.y : lhs.y,
                      lhs.z < rhs.z ? rhs.z : lhs.z);
}

template<typename T>
__always_inline__ GPU_ENABLED Vector4<T> max(Vector4<T> lhs, Vector4<T> rhs) {
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
typedef Vector4<unsigned int> Vector4ui;

} // NS Bifrost::Math

// ------------------------------------------------------------------------------------------------
// Convenience functions that appends a vector's string representation to an ostream.
// ------------------------------------------------------------------------------------------------
#ifndef GPU_COMPILATION
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
#endif

// ------------------------------------------------------------------------------------------------
// Math operator overloading.
// ------------------------------------------------------------------------------------------------

template<class T>
__always_inline__ GPU_ENABLED Bifrost::Math::Vector2<T> operator+(T lhs, Bifrost::Math::Vector2<T> rhs) {
    return rhs + lhs;
}

template<class T>
__always_inline__ GPU_ENABLED Bifrost::Math::Vector3<T> operator+(T lhs, Bifrost::Math::Vector3<T> rhs) {
    return rhs + lhs;
}

template<class T>
__always_inline__ GPU_ENABLED Bifrost::Math::Vector4<T> operator+(T lhs, Bifrost::Math::Vector4<T> rhs) {
    return rhs + lhs;
}

template<class T>
__always_inline__ GPU_ENABLED Bifrost::Math::Vector2<T> operator-(T lhs, Bifrost::Math::Vector2<T> rhs) {
    return Bifrost::Math::Vector2<T>(lhs - rhs.x, lhs - rhs.y);
}

template<class T>
__always_inline__ GPU_ENABLED Bifrost::Math::Vector3<T> operator-(T lhs, Bifrost::Math::Vector3<T> rhs) {
    return Bifrost::Math::Vector3<T>(lhs - rhs.x, lhs - rhs.y, lhs - rhs.z);
}

template<class T>
__always_inline__ GPU_ENABLED Bifrost::Math::Vector4<T> operator-(T lhs, Bifrost::Math::Vector4<T> rhs) {
    return Bifrost::Math::Vector4<T>(lhs - rhs.x, lhs - rhs.y, lhs - rhs.z, lhs - rhs.w);
}

template<class T>
__always_inline__ GPU_ENABLED Bifrost::Math::Vector2<T> operator*(T lhs, Bifrost::Math::Vector2<T> rhs) {
    return rhs * lhs;
}

template<class T>
__always_inline__ GPU_ENABLED Bifrost::Math::Vector3<T> operator*(T lhs, Bifrost::Math::Vector3<T> rhs) {
    return rhs * lhs;
}

template<class T>
__always_inline__ GPU_ENABLED Bifrost::Math::Vector4<T> operator*(T lhs, Bifrost::Math::Vector4<T> rhs) {
    return rhs * lhs;
}

template<class T>
__always_inline__ GPU_ENABLED Bifrost::Math::Vector2<T> operator/(T lhs, Bifrost::Math::Vector2<T> rhs) {
    return Bifrost::Math::Vector2<T>(lhs / rhs.x, lhs / rhs.y);
}

template<class T>
__always_inline__ GPU_ENABLED Bifrost::Math::Vector3<T> operator/(T lhs, Bifrost::Math::Vector3<T> rhs) {
    return Bifrost::Math::Vector3<T>(lhs / rhs.x, lhs / rhs.y, lhs / rhs.z);
}

template<class T>
__always_inline__ GPU_ENABLED Bifrost::Math::Vector4<T> operator/(T lhs, Bifrost::Math::Vector4<T> rhs) {
    return Bifrost::Math::Vector4<T>(lhs / rhs.x, lhs / rhs.y, lhs / rhs.z, lhs / rhs.w);
}

#endif // _BIFROST_MATH_VECTOR_H_
