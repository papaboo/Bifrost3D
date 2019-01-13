// Bifrost color abstractions.
// ----------------------------------------------------------------------------
// Copyright (C) 2015, Bifrost. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ----------------------------------------------------------------------------

#ifndef _BIFROST_MATH_COLOR_H_
#define _BIFROST_MATH_COLOR_H_

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

    static inline RGB black()  { return RGB(0.0f, 0.0f, 0.0f); }
    static inline RGB blue()   { return RGB(0.0f, 0.0f, 1.0f); }
    static inline RGB green()  { return RGB(0.0f, 1.0f, 0.0f); }
    static inline RGB cyan()   { return RGB(0.0f, 1.0f, 1.0f); }
    static inline RGB red()    { return RGB(1.0f, 0.0f, 0.0f); }
    static inline RGB purple() { return RGB(1.0f, 0.0f, 1.0f); }
    static inline RGB yellow() { return RGB(1.0f, 1.0f, 0.0f); }
    static inline RGB grey()   { return RGB(0.5f, 0.5f, 0.5f); }
    static inline RGB white()  { return RGB(1.0f, 1.0f, 1.0f); }

    inline float* begin() { return &r; }
    inline const float* begin() const { return &r; }

    inline float& operator[](const int i) { return begin()[i]; }
    inline float operator[](const int i) const { return begin()[i]; }

    //*****************************************************************************
    // Addition operators.
    //*****************************************************************************
    inline RGB& operator+=(float v) {
        for (int i = 0; i < N; ++i)
            begin()[i] += v;
        return *this;
    }
    inline RGB& operator+=(RGB v) {
        for (int i = 0; i < N; ++i)
            begin()[i] += v[i];
        return *this;
    }
    inline RGB operator+(float rhs) const {
        RGB ret(*this);
        for (int i = 0; i < N; ++i)
            ret[i] += rhs;
        return ret;
    }
    inline RGB operator+(RGB rhs) const {
        RGB ret(*this);
        for (int i = 0; i < N; ++i)
            ret[i] += rhs[i];
        return ret;
    }

    //*****************************************************************************
    // Subtraction operators.
    //*****************************************************************************
    inline RGB& operator-=(const float v) {
        for (int i = 0; i < N; ++i)
            begin()[i] -= v;
        return *this;
    }
    inline RGB& operator-=(RGB v) {
        for (int i = 0; i < N; ++i)
            begin()[i] -= v[i];
        return *this;
    }
    inline RGB operator-(float rhs) const {
        RGB ret(*this);
        for (int i = 0; i < N; ++i)
            ret[i] -= rhs;
        return ret;
    }
    inline RGB operator-(RGB rhs) const {
        RGB ret(*this);
        for (int i = 0; i < N; ++i)
            ret[i] -= rhs[i];
        return ret;
    }

    //*****************************************************************************
    // Multiplication operators.
    //*****************************************************************************
    inline RGB& operator*=(float v) {
        for (int i = 0; i < N; ++i)
            begin()[i] *= v;
        return *this;
    }
    inline RGB& operator*=(RGB v) {
        for (int i = 0; i < N; ++i)
            begin()[i] *= v[i];
        return *this;
    }
    inline RGB operator*(float rhs) const {
        RGB ret(*this);
        for (int i = 0; i < N; ++i)
            ret[i] *= rhs;
        return ret;
    }
    inline RGB operator*(RGB rhs) const {
        RGB ret(*this);
        for (int i = 0; i < N; ++i)
            ret[i] *= rhs[i];
        return ret;
    }

    //*****************************************************************************
    // Division operators.
    //*****************************************************************************
    inline RGB& operator/=(float v) {
        for (int i = 0; i < N; ++i)
            begin()[i] /= v;
        return *this;
    }
    inline RGB& operator/=(RGB v) {
        for (int i = 0; i < N; ++i)
            begin()[i] /= v[i];
        return *this;
    }
    inline RGB operator/(float rhs) const {
        RGB ret(*this);
        for (int i = 0; i < N; ++i)
            ret[i] /= rhs;
        return ret;
    }
    inline RGB operator/(RGB rhs) const {
        RGB ret(*this);
        for (int i = 0; i < N; ++i)
            ret[i] /= rhs[i];
        return ret;
    }

    //*****************************************************************************
    // Comparison operators.
    //*****************************************************************************
    inline bool operator==(RGB rhs) const {
        return memcmp(this, &rhs, sizeof(rhs)) == 0;
    }
    inline bool operator!=(RGB rhs) const {
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

    static inline RGBA black()  { return RGBA(RGB::black()); }
    static inline RGBA blue()   { return RGBA(RGB::blue()); }
    static inline RGBA green()  { return RGBA(RGB::green()); }
    static inline RGBA cyan()   { return RGBA(RGB::cyan()); }
    static inline RGBA red()    { return RGBA(RGB::red()); }
    static inline RGBA purple() { return RGBA(RGB::purple()); }
    static inline RGBA yellow() { return RGBA(RGB::yellow()); }
    static inline RGBA grey()   { return RGBA(RGB::grey()); }
    static inline RGBA white()  { return RGBA(RGB::white()); }

    inline float* begin() { return &r; }
    inline const float* begin() const { return &r; }

    inline float& operator[](const int i) { return begin()[i]; }
    inline float operator[](const int i) const { return begin()[i]; }

    inline RGB& rgb() {
        return *static_cast<RGB*>(static_cast<void*>(begin()));
    }

    //*****************************************************************************
    // Comparison operators.
    //*****************************************************************************
    inline bool operator==(RGBA rhs) const {
        return memcmp(this, &rhs, sizeof(rhs)) == 0;
    }
    inline bool operator!=(RGBA rhs) const {
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

inline RGBA lerp(RGBA a, RGBA b, float t) {
    return RGBA(a.rgb() + (b.rgb() - a.rgb()) * t, a.a + (b.a - a.a) * t);
}

inline RGB gammacorrect(RGB color, float gamma) {
    return RGB(pow(color.r, gamma), pow(color.g, gamma), pow(color.b, gamma));
}

inline RGBA gammacorrect(RGBA color, float gamma) {
    return RGBA(gammacorrect(color.rgb(), gamma), color.a);
}

inline float luminance(RGB color) {
    return 0.2126f * color.r + 0.7152f * color.g + 0.0722f * color.b;
}

inline RGB saturate(RGB color) {
    auto saturate = [](float v) { return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v); };
    return RGB(saturate(color.r), saturate(color.g), saturate(color.b));
}

} // NS Math
} // NS Bifrost

// ------------------------------------------------------------------------------------------------
// Convenience functions that appends a color's string representation to an ostream.
// ------------------------------------------------------------------------------------------------
inline std::ostream& operator<<(std::ostream& s, Bifrost::Math::RGB v){
    return s << v.to_string();
}

inline std::ostream& operator<<(std::ostream& s, Bifrost::Math::RGBA v){
    return s << v.to_string();
}

// ------------------------------------------------------------------------------------------------
// Math operator overloading.
// ------------------------------------------------------------------------------------------------

inline Bifrost::Math::RGB operator+(float lhs, Bifrost::Math::RGB rhs) {
    return rhs + lhs;
}

inline Bifrost::Math::RGB operator*(float lhs, Bifrost::Math::RGB rhs) {
    return rhs * lhs;
}

#endif // _BIFROST_MATH_COLOR_H_
