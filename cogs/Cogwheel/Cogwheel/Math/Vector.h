// Cogwheel Vector abstraction.
// ----------------------------------------------------------------------------
// Copyright (C) 2015, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ----------------------------------------------------------------------------

#ifndef _COGWHEEL_MATH_VECTOR_H_
#define _COGWHEEL_MATH_VECTOR_H_

#include <algorithm>
#include <cstring>
#include <sstream>

namespace Cogwheel {
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

    Vector2() : x(0), y(0) { }
    explicit Vector2(T s) : x(s), y(s) { }
    Vector2(T x, T y) : x(x), y(y) { }
    template <typename U>
    Vector2(const Vector2<U>& v) : x(T(v.x)), y(T(v.y)) { }

    static inline Vector2<T> zero() { return Vector2<T>(0, 0); }
    static inline Vector2<T> one() { return Vector2<T>(1, 1); }

    inline T* begin() { return &x; }
    inline const T* begin() const { return &x; }

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

    Vector3() : x(0), y(0), z(0) { }
    explicit Vector3(T s) : x(s), y(s), z(s) { }
    Vector3(T x, T y, T z) : x(x), y(y), z(z) { }
    template <typename U>
    explicit Vector3(const Vector3<U>& v) : x(T(v.x)), y(T(v.y)), z(T(v.z)) { }
    Vector3(const Vector2<T> v, T z) : x(v.x), y(v.y), z(z) { }

    static inline Vector3<T> zero() { return Vector3(0, 0, 0); }
    static inline Vector3<T> one() { return Vector3(1, 1, 1); }

    static inline Vector3<T> forward() { return Vector3(0, 0, 1); }
    static inline Vector3<T> up() { return Vector3(0, 1, 0); }
    static inline Vector3<T> right() { return Vector3(1, 0, 0); }

    inline T* begin() { return &x; }
    inline const T* begin() const { return &x; }

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

    Vector4() : x(0), y(0), z(0), w(0) { }
    explicit Vector4(T s) : x(s), y(s), z(s), w(s) { }
    Vector4(T x, T y, T z, T w) : x(x), y(y), z(z), w(w) { }
    template <typename U>
    Vector4(const Vector4<U> v) : x(T(v.x)), y(T(v.y)), z(T(v.z)), w(T(v.w)) { }
    Vector4(const Vector2<T> v, T z, T w) : x(v.x), y(v.y), z(z), w(w) { }
    Vector4(const Vector3<T> v, T w) : x(v.x), y(v.y), z(v.z), w(w) { }

    static inline Vector4<T> zero() { return Vector4<T>(0, 0, 0, 0); }
    static inline Vector4<T> one() { return Vector4<T>(1, 1, 1, 1); }

    inline T* begin() { return &x; }
    inline const T* begin() const { return &x; }

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
inline T dot(Vector<T> lhs, Vector<T> rhs) {
    T res = lhs[0] * rhs[0];
    for (int i = 1; i < Vector<T>::N; ++i)
        res += lhs[i] * rhs[i];
    return res;
}

// Compute the squared magnitude of the input vector.
// Useful when comparing the relative size between vectors, where the exact magnitude isn't needed.
template<template<typename> class Vector, typename T>
inline T magnitude_squared(Vector<T> v) {
    return dot(v, v);
}

// Compute the magnitude of the input vector.
template<template<typename> class Vector, typename T>
inline T magnitude(Vector<T> v) {
    return sqrt(dot(v, v));
}

// Create a normalized version of the input vector.
template<template<typename> class Vector, typename T>
inline Vector<T> normalize(Vector<T> v){
    T m = magnitude(v);
    return v / m;
}

// Cross product between two 3-dimensional vectors.
template<typename T>
inline Vector3<T> cross(Vector3<T> lhs, Vector3<T> rhs) {
    return Vector3<T>((lhs.y * rhs.z) - (lhs.z * rhs.y),
        (lhs.z * rhs.x) - (lhs.x * rhs.z),
        (lhs.x * rhs.y) - (lhs.y * rhs.x));
}

template<typename T>
inline Vector2<T> min(Vector2<T> lhs, Vector2<T> rhs) {
    return Vector2<T>(std::min(lhs.x, rhs.x), std::min(lhs.y, rhs.y));
}

template<typename T>
inline Vector3<T> min(Vector3<T> lhs, Vector3<T> rhs) {
    return Vector3<T>(std::min(lhs.x, rhs.x), std::min(lhs.y, rhs.y), std::min(lhs.z, rhs.z));
}

template<typename T>
inline Vector4<T> min(Vector4<T> lhs, Vector4<T> rhs) {
    return Vector4<T>(std::min(lhs.x, rhs.x), std::min(lhs.y, rhs.y), std::min(lhs.z, rhs.z), std::min(lhs.w, rhs.w));
}

template<typename T>
inline Vector2<T> max(Vector2<T> lhs, Vector2<T> rhs) {
    return Vector2<T>(std::max(lhs.x, rhs.x), std::max(lhs.y, rhs.y));
}

template<typename T>
inline Vector3<T> max(Vector3<T> lhs, Vector3<T> rhs) {
    return Vector3<T>(std::max(lhs.x, rhs.x), std::max(lhs.y, rhs.y), std::max(lhs.z, rhs.z));
}

template<typename T>
inline Vector4<T> max(Vector4<T> lhs, Vector4<T> rhs) {
    return Vector4<T>(std::max(lhs.x, rhs.x), std::max(lhs.y, rhs.y), std::max(lhs.z, rhs.z), std::max(lhs.w, rhs.w));
}

// Comparison that checks if two vectors are almost equal.
template<typename T>
inline bool almost_equal(Vector2<T> lhs, Vector2<T> rhs, unsigned short max_ulps = 4) {
    return almost_equal(lhs.x, rhs.x, max_ulps)
        && almost_equal(lhs.y, rhs.y, max_ulps);
}
template<typename T>
inline bool almost_equal(Vector3<T> lhs, Vector3<T> rhs, unsigned short max_ulps = 4) {
    return almost_equal(lhs.x, rhs.x, max_ulps)
        && almost_equal(lhs.y, rhs.y, max_ulps)
        && almost_equal(lhs.z, rhs.z, max_ulps);
}
template<typename T>
inline bool almost_equal(Vector4<T> lhs, Vector4<T> rhs, unsigned short max_ulps = 4) {
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
} // NS Cogwheel

// ------------------------------------------------------------------------------------------------
// Convenience functions that appends a vector's string representation to an ostream.
// ------------------------------------------------------------------------------------------------
template<class T>
inline std::ostream& operator<<(std::ostream& s, Cogwheel::Math::Vector2<T> v){
    return s << v.to_string();
}

template<class T>
inline std::ostream& operator<<(std::ostream& s, Cogwheel::Math::Vector3<T> v){
    return s << v.to_string();
}

template<class T>
inline std::ostream& operator<<(std::ostream& s, Cogwheel::Math::Vector4<T> v){
    return s << v.to_string();
}

// ------------------------------------------------------------------------------------------------
// Math operator overloading.
// ------------------------------------------------------------------------------------------------

template<class T>
inline Cogwheel::Math::Vector2<T> operator+(T lhs, Cogwheel::Math::Vector2<T> rhs) {
    return rhs + lhs;
}

template<class T>
inline Cogwheel::Math::Vector3<T> operator+(T lhs, Cogwheel::Math::Vector3<T> rhs) {
    return rhs + lhs;
}

template<class T>
inline Cogwheel::Math::Vector4<T> operator+(T lhs, Cogwheel::Math::Vector4<T> rhs) {
    return rhs + lhs;
}

template<class T>
inline Cogwheel::Math::Vector2<T> operator-(T lhs, Cogwheel::Math::Vector2<T> rhs) {
    return Cogwheel::Math::Vector2<T>(lhs - rhs.x, lhs - rhs.y);
}

template<class T>
inline Cogwheel::Math::Vector3<T> operator-(T lhs, Cogwheel::Math::Vector3<T> rhs) {
    return Cogwheel::Math::Vector3<T>(lhs - rhs.x, lhs - rhs.y, lhs - rhs.z);
}

template<class T>
inline Cogwheel::Math::Vector4<T> operator-(T lhs, Cogwheel::Math::Vector4<T> rhs) {
    return Cogwheel::Math::Vector4<T>(lhs - rhs.x, lhs - rhs.y, lhs - rhs.z, lhs - rhs.w);
}

template<class T>
inline Cogwheel::Math::Vector2<T> operator*(T lhs, Cogwheel::Math::Vector2<T> rhs) {
    return rhs * lhs;
}

template<class T>
inline Cogwheel::Math::Vector3<T> operator*(T lhs, Cogwheel::Math::Vector3<T> rhs) {
    return rhs * lhs;
}

template<class T>
inline Cogwheel::Math::Vector4<T> operator*(T lhs, Cogwheel::Math::Vector4<T> rhs) {
    return rhs * lhs;
}

template<class T>
inline Cogwheel::Math::Vector2<T> operator/(T lhs, Cogwheel::Math::Vector2<T> rhs) {
    return Cogwheel::Math::Vector2<T>(lhs / rhs.x, lhs / rhs.y);
}

template<class T>
inline Cogwheel::Math::Vector3<T> operator/(T lhs, Cogwheel::Math::Vector3<T> rhs) {
    return Cogwheel::Math::Vector3<T>(lhs / rhs.x, lhs / rhs.y, lhs / rhs.z);
}

template<class T>
inline Cogwheel::Math::Vector4<T> operator/(T lhs, Cogwheel::Math::Vector4<T> rhs) {
    return Cogwheel::Math::Vector4<T>(lhs / rhs.x, lhs / rhs.y, lhs / rhs.z, lhs / rhs.w);
}

#endif // _COGWHEEL_MATH_VECTOR_H_