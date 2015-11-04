// Cogwheel color abstractions.
// ----------------------------------------------------------------------------
// Copyright (C) 2015, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ----------------------------------------------------------------------------

#ifndef _COGWHEEL_MATH_COLOR_H_
#define _COGWHEEL_MATH_COLOR_H_

#include <algorithm>
#include <sstream>
#include <string>

namespace Cogwheel {
namespace Math {

struct RGB final {
public:
    typedef float value_type;
    static const int N = 3;

    //*****************************************************************************
    // Public members
    //*****************************************************************************
    float r;
    float g;
    float b;

    RGB()
        : r(0.0f), g(0.0f), b(0.0f) { }

    RGB(unsigned char r, unsigned char g, unsigned char b)
        : r(r / 255.0f), g(g / 255.0f), b(b / 255.0f) { }

    RGB(float r, float g, float b)
        : r(r), g(g), b(b) { }

    static inline RGB black()  { return RGB(0.0f, 0.0f, 0.0f); }
    static inline RGB blue()   { return RGB(0.0f, 0.0f, 1.0f); }
    static inline RGB green()  { return RGB(0.0f, 1.0f, 0.0f); }
    static inline RGB cyan()   { return RGB(0.0f, 1.0f, 1.0f); }
    static inline RGB red()    { return RGB(1.0f, 0.0f, 0.0f); }
    static inline RGB purple() { return RGB(1.0f, 0.0f, 1.0f); }
    static inline RGB yellow() { return RGB(1.0f, 1.0f, 0.0f); }
    static inline RGB white()  { return RGB(1.0f, 1.0f, 1.0f); }

    inline float* begin() { return &r; }
    inline const float* const begin() const { return &r; }

    inline float& operator[](const int i) { return begin()[i]; }
    inline float operator[](const int i) const { return begin()[i]; }

    // TODO Remove once we have 8bit colors.
    inline void copyData(unsigned char* vs, const int entries = 3) const {
        for (int e = 0; e < entries; ++e) {
            const float v = 255 * begin()[e] + 0.5f;
            vs[e] = static_cast<unsigned char>(std::max(std::min(v, 0.0f), 1.0f));
        }
    }

    //*****************************************************************************
    // Addition operators.
    //*****************************************************************************
    inline void operator+=(float v) {
        for (int i = 0; i < N; ++i)
            this[i] += v;
    }
    inline void operator+=(RGB v) {
        for (int i = 0; i < N; ++i)
            this[i] += v[i];
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
    inline void operator-=(const float v) {
        for (int i = 0; i < N; ++i)
            this[i] -= v;
    }
    inline void operator-=(RGB v) {
        for (int i = 0; i < N; ++i)
            this[i] -= v[i];
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
    inline void operator*=(float v) {
        for (int i = 0; i < N; ++i)
            this[i] *= v;
    }
    inline void operator*=(RGB v) {
        for (int i = 0; i < N; ++i)
            this[i] *= v[i];
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
    inline void operator/=(float v) {
        for (int i = 0; i < N; ++i)
            this[i] /= v;
    }
    inline void operator/=(RGB v) {
        for (int i = 0; i < N; ++i)
            this[i] /= v[i];
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

    const std::string toString() const {
        std::ostringstream out;
        out << "[r: " << r << ", g: " << g << ", b: " << b << "]";
        return out.str();
    }
};

struct RGBA final {
public:
    typedef float value_type;
    static const int N = 4;

    //*****************************************************************************
    // Public members
    //*****************************************************************************
    float r;
    float g;
    float b;
    float a;

    RGBA()
        : r(0.0f), g(0.0f), b(0.0f), a(1.0f) { }

    RGBA(unsigned char r, unsigned char g, unsigned char b, unsigned char a)
        : r(r / 255.0f), g(g / 255.0f), b(b / 255.0f), a(a / 255.0f) { }

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
    static inline RGBA white()  { return RGBA(RGB::white()); }

    inline float* begin() { return &r; }
    inline const float* const begin() const { return &r; }

    inline float& operator[](const int i) { return begin()[i]; }
    inline float operator[](const int i) const { return begin()[i]; }

    inline RGB& rgb() {
        return *static_cast<RGB*>(static_cast<void*>(begin()));
    }

    // TODO Remove once we have 8bit colors.
    inline void copyData(unsigned char* vs, const int entries = 4) const {
        for (int e = 0; e < entries; ++e) {
            const float v = 255 * begin()[e] + 0.5f;
            vs[e] = static_cast<unsigned char>(std::max(std::min(v, 0.0f), 1.0f));
        }
    }

    const std::string toString() const {
        std::ostringstream out;
        out << "[r: " << r << ", g: " << g << ", b: " << b << ", a: " << a << "]";
        return out.str();
    }
};

} // NS Math
} // NS Cogwheel

// Convinience function that appends an RGBA's string representation to an ostream.
inline std::ostream& operator<<(std::ostream& s, Cogwheel::Math::RGB v){
    return s << v.toString();
}

// Convinience function that appends an RGBA's string representation to an ostream.
inline std::ostream& operator<<(std::ostream& s, Cogwheel::Math::RGBA v){
    return s << v.toString();
}

#endif // _COGWHEEL_MATH_COLOR_H_