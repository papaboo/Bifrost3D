// Bifrost color abstractions.
// ----------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ----------------------------------------------------------------------------

#ifndef _BIFROST_MATH_COLOR_H_
#define _BIFROST_MATH_COLOR_H_

#include <Bifrost/Core/Defines.h>

#include <algorithm>
#include <sstream>
#include <string>

namespace Bifrost {
namespace Math {

struct RGB final {
    typedef float value_type;
    static const int N = 3;

    //*****************************************************************************
    // Public members
    //*****************************************************************************
    float r;
    float g;
    float b;

    RGB() = default;

    RGB(float intensity)
        : r(intensity), g(intensity), b(intensity) {
    }

    RGB(float r, float g, float b)
        : r(r), g(g), b(b) { }

    static __always_inline__ RGB black()  { return RGB(0.0f, 0.0f, 0.0f); }
    static __always_inline__ RGB blue()   { return RGB(0.0f, 0.0f, 1.0f); }
    static __always_inline__ RGB green()  { return RGB(0.0f, 1.0f, 0.0f); }
    static __always_inline__ RGB cyan()   { return RGB(0.0f, 1.0f, 1.0f); }
    static __always_inline__ RGB red()    { return RGB(1.0f, 0.0f, 0.0f); }
    static __always_inline__ RGB purple() { return RGB(1.0f, 0.0f, 1.0f); }
    static __always_inline__ RGB yellow() { return RGB(1.0f, 1.0f, 0.0f); }
    static __always_inline__ RGB grey()   { return RGB(0.5f, 0.5f, 0.5f); }
    static __always_inline__ RGB white()  { return RGB(1.0f, 1.0f, 1.0f); }

    __always_inline__ float* begin() { return &r; }
    __always_inline__ const float* const begin() const { return &r; }
    __always_inline__ float* end() { return begin() + N; }
    __always_inline__ const float* const end() const { return begin() + N; }

    __always_inline__ float& operator[](const int i) { return begin()[i]; }
    __always_inline__ float operator[](const int i) const { return begin()[i]; }

    //*****************************************************************************
    // Addition operators.
    //*****************************************************************************
    __always_inline__ RGB& operator+=(float v) {
        for (int i = 0; i < N; ++i)
            begin()[i] += v;
        return *this;
    }
    __always_inline__ RGB& operator+=(RGB v) {
        for (int i = 0; i < N; ++i)
            begin()[i] += v[i];
        return *this;
    }
    __always_inline__ RGB operator+(float rhs) const {
        RGB ret(*this);
        for (int i = 0; i < N; ++i)
            ret[i] += rhs;
        return ret;
    }
    __always_inline__ RGB operator+(RGB rhs) const {
        RGB ret(*this);
        for (int i = 0; i < N; ++i)
            ret[i] += rhs[i];
        return ret;
    }

    //*****************************************************************************
    // Subtraction operators.
    //*****************************************************************************
    __always_inline__ RGB& operator-=(const float v) {
        for (int i = 0; i < N; ++i)
            begin()[i] -= v;
        return *this;
    }
    __always_inline__ RGB& operator-=(RGB v) {
        for (int i = 0; i < N; ++i)
            begin()[i] -= v[i];
        return *this;
    }
    __always_inline__ RGB operator-(float rhs) const {
        RGB ret(*this);
        for (int i = 0; i < N; ++i)
            ret[i] -= rhs;
        return ret;
    }
    __always_inline__ RGB operator-(RGB rhs) const {
        RGB ret(*this);
        for (int i = 0; i < N; ++i)
            ret[i] -= rhs[i];
        return ret;
    }

    //*****************************************************************************
    // Multiplication operators.
    //*****************************************************************************
    __always_inline__ RGB& operator*=(float v) {
        for (int i = 0; i < N; ++i)
            begin()[i] *= v;
        return *this;
    }
    __always_inline__ RGB& operator*=(RGB v) {
        for (int i = 0; i < N; ++i)
            begin()[i] *= v[i];
        return *this;
    }
    __always_inline__ RGB operator*(float rhs) const {
        RGB ret(*this);
        for (int i = 0; i < N; ++i)
            ret[i] *= rhs;
        return ret;
    }
    __always_inline__ RGB operator*(RGB rhs) const {
        RGB ret(*this);
        for (int i = 0; i < N; ++i)
            ret[i] *= rhs[i];
        return ret;
    }

    //*****************************************************************************
    // Division operators.
    //*****************************************************************************
    __always_inline__ RGB& operator/=(float v) {
        for (int i = 0; i < N; ++i)
            begin()[i] /= v;
        return *this;
    }
    __always_inline__ RGB& operator/=(RGB v) {
        for (int i = 0; i < N; ++i)
            begin()[i] /= v[i];
        return *this;
    }
    __always_inline__ RGB operator/(float rhs) const {
        RGB ret(*this);
        for (int i = 0; i < N; ++i)
            ret[i] /= rhs;
        return ret;
    }
    __always_inline__ RGB operator/(RGB rhs) const {
        RGB ret(*this);
        for (int i = 0; i < N; ++i)
            ret[i] /= rhs[i];
        return ret;
    }

    //*****************************************************************************
    // Comparison operators.
    //*****************************************************************************
    __always_inline__ bool operator==(RGB rhs) const {
        return memcmp(this, &rhs, sizeof(rhs)) == 0;
    }
    __always_inline__ bool operator!=(RGB rhs) const {
        return memcmp(this, &rhs, sizeof(rhs)) != 0;
    }

    inline std::string to_string() const {
        std::ostringstream out;
        out << "[r: " << r << ", g: " << g << ", b: " << b << "]";
        return out.str();
    }
};

struct RGBA final {
    typedef float value_type;
    static const int N = 4;

    //*****************************************************************************
    // Public members
    //*****************************************************************************
    float r;
    float g;
    float b;
    float a;

    RGBA() = default;

    RGBA(float r, float g, float b, float a)
        : r(r), g(g), b(b), a(a) { }

    RGBA(RGB rgb, float a = 1.0f)
        : r(rgb.r), g(rgb.g), b(rgb.b), a(a) { }

    static __always_inline__ RGBA black()  { return RGBA(RGB::black()); }
    static __always_inline__ RGBA blue()   { return RGBA(RGB::blue()); }
    static __always_inline__ RGBA green()  { return RGBA(RGB::green()); }
    static __always_inline__ RGBA cyan()   { return RGBA(RGB::cyan()); }
    static __always_inline__ RGBA red()    { return RGBA(RGB::red()); }
    static __always_inline__ RGBA purple() { return RGBA(RGB::purple()); }
    static __always_inline__ RGBA yellow() { return RGBA(RGB::yellow()); }
    static __always_inline__ RGBA grey()   { return RGBA(RGB::grey()); }
    static __always_inline__ RGBA white()  { return RGBA(RGB::white()); }

    __always_inline__ float* begin() { return &r; }
    __always_inline__ const float* const begin() const { return &r; }
    __always_inline__ float* end() { return begin() + N; }
    __always_inline__ const float* const end() const { return begin() + N; }

    __always_inline__ float& operator[](const int i) { return begin()[i]; }
    __always_inline__ float operator[](const int i) const { return begin()[i]; }

    __always_inline__ RGB& rgb() {
        return *static_cast<RGB*>(static_cast<void*>(begin()));
    }

    //*****************************************************************************
    // Comparison operators.
    //*****************************************************************************
    __always_inline__ bool operator==(RGBA rhs) const {
        return memcmp(this, &rhs, sizeof(rhs)) == 0;
    }
    __always_inline__ bool operator!=(RGBA rhs) const {
        return memcmp(this, &rhs, sizeof(rhs)) != 0;
    }

    inline std::string to_string() const {
        std::ostringstream out;
        out << "[r: " << r << ", g: " << g << ", b: " << b << ", a: " << a << "]";
        return out.str();
    }
};

//*****************************************************************************
// Free functions.
//*****************************************************************************

__always_inline__ RGBA lerp(RGBA a, RGBA b, float t) {
    return RGBA(a.rgb() + (b.rgb() - a.rgb()) * t, a.a + (b.a - a.a) * t);
}

__always_inline__ RGB gammacorrect(RGB color, float gamma) {
    return RGB(pow(color.r, gamma), pow(color.g, gamma), pow(color.b, gamma));
}

__always_inline__ RGBA gammacorrect(RGBA color, float gamma) {
    return RGBA(gammacorrect(color.rgb(), gamma), color.a);
}

__always_inline__ float luminance(RGB color) {
    return 0.2126f * color.r + 0.7152f * color.g + 0.0722f * color.b;
}

__always_inline__ RGB saturate(RGB color) {
    auto saturate = [](float v) { return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v); };
    return RGB(saturate(color.r), saturate(color.g), saturate(color.b));
}

} // NS Math
} // NS Bifrost

// ------------------------------------------------------------------------------------------------
// Convenience functions that appends a color's string representation to an ostream.
// ------------------------------------------------------------------------------------------------
__always_inline__ std::ostream& operator<<(std::ostream& s, Bifrost::Math::RGB v){
    return s << v.to_string();
}

__always_inline__ std::ostream& operator<<(std::ostream& s, Bifrost::Math::RGBA v){
    return s << v.to_string();
}

// ------------------------------------------------------------------------------------------------
// Math operator overloading.
// ------------------------------------------------------------------------------------------------

__always_inline__ Bifrost::Math::RGB operator+(float lhs, Bifrost::Math::RGB rhs) {
    return rhs + lhs;
}

__always_inline__ Bifrost::Math::RGB operator*(float lhs, Bifrost::Math::RGB rhs) {
    return rhs * lhs;
}

#endif // _BIFROST_MATH_COLOR_H_
