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
#include <Bifrost/Math/FixedPointTypes.h>

#include <sstream>
#include <string>

namespace Bifrost::Math {

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
        : r(intensity), g(intensity), b(intensity) { }

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
// Simple 8bit pr channel color representations.
//*****************************************************************************

struct RGB24 {
    UNorm8 r, g, b;
};

struct RGBA32 {
    UNorm8 r, g, b, a;
};

//*****************************************************************************
// HSV.
//*****************************************************************************

struct HSV {
    float h, s, v;

    HSV() = default;

    HSV(float h, float s, float v) : h(h), s(s), v(v) { }

    // Conversion from https://www.rapidtables.com/convert/color/rgb-to-hsv.html
    explicit HSV(RGB rgb)
    : h(0), s(0), v(0) {
        float c_max = fmaxf(rgb.r, fmaxf(rgb.g, rgb.b));
        float c_min = fminf(rgb.r, fminf(rgb.g, rgb.b));
        float c_delta = c_max - c_min;

        // Degenerate case
        if (c_max == 0)
            return;

        v = c_max;
        s = c_delta / c_max;

        if (c_delta) {
            if (c_max == rgb.r) {
                h = (rgb.g - rgb.b) / c_delta;
                if (h < 0.0f) // simple replacement for 'mod 6'
                    h += 6;
            } else if (c_max == rgb.g)
                h = (rgb.b - rgb.r) / c_delta + 2;
            else // c_max == rgb.b
                h = (rgb.r - rgb.g) / c_delta + 4;
            h *= 60;
        }
    }

    // Conversion from https://www.rapidtables.com/convert/color/hsv-to-rgb.html
    explicit operator RGB() const {
        float c = v * s;
        float x = c * (1 - abs(fmodf(h / 60.0f, 2) - 1));
        float m = v - c;
        float x_m = x + m;

        if (h < 60.0f)
            return RGB(v, x_m, m);
        else if (h < 120.0f)
            return RGB(x_m, v, m);
        else if (h < 180.0f)
            return RGB(m, v, x_m);
        else if (h < 240.0f)
            return RGB(m, x_m, v);
        else if (h < 300.0f)
            return RGB(x_m, m, v);
        else // h < 360.0f
            return RGB(v, m, x_m);
    }

    inline std::string to_string() const {
        std::ostringstream out;
        out << "[h: " << h << ", s: " << s << ", v: " << v << "]";
        return out.str();
    }
};

//*****************************************************************************
// Free functions.
//*****************************************************************************

__always_inline__ RGBA lerp(RGBA a, RGBA b, float t) {
    return RGBA(a.rgb() + (b.rgb() - a.rgb()) * t, a.a + (b.a - a.a) * t);
}

__always_inline__ HSV lerp(HSV a, HSV b, float t) {
    auto lerp = [](float a, float b, float t) -> float { return a + (b - a) * t; };

    // Hue is expressed as angles, so we need to handle the case where a low angle should lerp towards a large angle, e.g. 10 degrees should lerp towards 350 degrees
    // This is done by subtracting 360 from the high number, such that it becomes an interpolation between 10 and -10.
    if (abs(a.h - b.h) > 180)
        if (a.h > b.h)
            a.h -= 360;
        else
            b.h -= 360;
    float h = lerp(a.h, b.h, t);
    if (h < 0.0f)
        h += 360;

    return HSV(h, lerp(a.s, b.s, t), lerp(a.v, b.v, t));
}

__always_inline__ RGB gammacorrect(RGB color, float gamma) {
    return RGB(pow(color.r, gamma), pow(color.g, gamma), pow(color.b, gamma));
}

__always_inline__ RGBA gammacorrect(RGBA color, float gamma) {
    return RGBA(gammacorrect(color.rgb(), gamma), color.a);
}

// Source: https://physicallybased.info/tools/
__always_inline__ float sRGB_to_linear(float sRGB_intensity) {
    if (sRGB_intensity < 0.04045f)
        return sRGB_intensity * 0.0773993808f;
    else
        return pow(sRGB_intensity * 0.9478672986f + 0.0521327014f, 2.4f);
}

__always_inline__ RGB sRGB_to_linear(RGB color) {
    return RGB(sRGB_to_linear(color.r), sRGB_to_linear(color.g), sRGB_to_linear(color.b));
}

__always_inline__ RGBA sRGB_to_linear(RGBA color) {
    return RGBA(sRGB_to_linear(color.r), sRGB_to_linear(color.g), sRGB_to_linear(color.b), color.a);
}

// Source: https://physicallybased.info/tools/
__always_inline__ float linear_to_sRGB(float linear_intensity) {
    if (linear_intensity < 0.0031308f)
        return linear_intensity * 12.92f;
    else
        return 1.055f * pow(linear_intensity, 1.0f / 2.4f) - 0.055f;
}

__always_inline__ RGB linear_to_sRGB(RGB color) {
    return RGB(linear_to_sRGB(color.r), linear_to_sRGB(color.g), linear_to_sRGB(color.b));
}

__always_inline__ RGBA linear_to_sRGB(RGBA color) {
    return RGBA(linear_to_sRGB(color.r), linear_to_sRGB(color.g), linear_to_sRGB(color.b), color.a);
}

__always_inline__ float average(RGB color) {
    return (color.r + color.g + color.b) / 3.0f;
}

__always_inline__ float luminance(RGB color) {
    return 0.2126f * color.r + 0.7152f * color.g + 0.0722f * color.b;
}

__always_inline__ RGB saturate(RGB color) {
    auto saturate = [](float v) { return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v); };
    return RGB(saturate(color.r), saturate(color.g), saturate(color.b));
}

__always_inline__ float hue(RGB color) {
    const float c = 0.8660254f; // 0.5f * sqrtf(3);
    float alpha = color.r - 0.5f * (color.g + color.b);
    float beta = c * (color.g - color.b);
    return atan2f(beta, alpha);
}

__always_inline__ float chroma(RGB color) {
    const float c = 0.8660254f; // 0.5f * sqrtf(3);
    float alpha = color.r - 0.5f * (color.g + color.b);
    float beta = c * (color.g - color.b);
    return sqrtf(alpha * alpha + beta * beta);
}

} // NS Bifrost::Math

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
