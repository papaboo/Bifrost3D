// Bifrost fixed point math types.
// DX11 fixed point encoding: https://msdn.microsoft.com/en-us/library/windows/desktop/dd607323(v=vs.85).aspx#integer_conversion
// ----------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ----------------------------------------------------------------------------

#ifndef _BIFROST_MATH_FIXED_POINT_TYPES_H_
#define _BIFROST_MATH_FIXED_POINT_TYPES_H_

#include <Bifrost/Core/Defines.h>

namespace Bifrost::Math {

struct UNorm8 final {
    byte raw;

    UNorm8() = default;
    UNorm8(float v) : raw(to_byte(v)) {}
    UNorm8(byte v) : raw(v) {}

    static UNorm8 create_unchecked(float v) { return byte(v * 255.0f + 0.5f); }

    static UNorm8 zero() { return {}; }
    static UNorm8 one() { return { byte(255) }; }

    __always_inline__ static float max_precision() { return 1.0f / 510.0f; }
    __always_inline__ static float to_float(byte v) { return v / 255.0f; }
    __always_inline__ static byte to_byte(float v) {
        float clamped_v = v < 0 ? 0.0f : (v > 1.0f ? 1.0f : v);
        return byte(clamped_v * 255.0f + 0.5f);
    }

    __always_inline__ bool operator==(UNorm8 rhs) const { return raw == rhs.raw; }
    __always_inline__ bool operator!=(UNorm8 rhs) const { return raw != rhs.raw; }

    __always_inline__ float value() const { return to_float(raw); }
    __always_inline__ operator float() const { return to_float(raw); }
};

} // NS Bifrost::Math

#endif // _BIFROST_MATH_FIXED_POINT_TYPES_H_