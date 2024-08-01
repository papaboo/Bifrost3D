// Bifrost morton encoding functions.
// ------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _BIFROST_MATH_MORTON_ENCODE_H_
#define _BIFROST_MATH_MORTON_ENCODE_H_

#include <Bifrost/Math/Vector.h>

namespace Bifrost {
namespace Math {

// Insert a 0 bit in between each of the 16 low bits of v.
__always_inline__ unsigned int part_by_1(unsigned int v) {
    v &= 0x0000ffff;                 // v = ---- ---- ---- ---- fedc ba98 7654 3210
    v = (v ^ (v << 8)) & 0x00ff00ff; // v = ---- ---- fedc ba98 ---- ---- 7654 3210
    v = (v ^ (v << 4)) & 0x0f0f0f0f; // v = ---- fedc ---- ba98 ---- 7654 ---- 3210
    v = (v ^ (v << 2)) & 0x33333333; // v = --fe --dc --ba --98 --76 --54 --32 --10
    v = (v ^ (v << 1)) & 0x55555555; // v = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
    return v;
}

__always_inline__ unsigned int morton_encode(unsigned int x, unsigned int y) {
    return part_by_1(y) | (part_by_1(x) << 1);
}

// Insert two 0 bits after each of the 10 low bits of v.
__always_inline__ unsigned int part_by_2(unsigned int v) {
    v &= 0x000003ff;                  // v = ---- ---- ---- ---- ---- --98 7654 3210
    v = (v ^ (v << 16)) & 0xff0000ff; // v = ---- --98 ---- ---- ---- ---- 7654 3210
    v = (v ^ (v << 8)) & 0x0300f00f;  // v = ---- --98 ---- ---- 7654 ---- ---- 3210
    v = (v ^ (v << 4)) & 0x030c30c3;  // v = ---- --98 ---- 76-- --54 ---- 32-- --10
    v = (v ^ (v << 2)) & 0x09249249;  // v = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
    return v;
}

__always_inline__ unsigned int morton_encode(unsigned int x, unsigned int y, unsigned int z) {
    return part_by_2(z) | (part_by_2(y) << 1) | (part_by_2(x) << 2);
}

// Inverse of part_by_1, i.e delete all odd-indexed bits and compacts the rest.
__always_inline__ unsigned int compact_by_1(unsigned int v) {
    v &= 0x55555555;                 // v = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
    v = (v ^ (v >> 1)) & 0x33333333; // v = --fe --dc --ba --98 --76 --54 --32 --10
    v = (v ^ (v >> 2)) & 0x0f0f0f0f; // v = ---- fedc ---- ba98 ---- 7654 ---- 3210
    v = (v ^ (v >> 4)) & 0x00ff00ff; // v = ---- ---- fedc ba98 ---- ---- 7654 3210
    v = (v ^ (v >> 8)) & 0x0000ffff; // v = ---- ---- ---- ---- fedc ba98 7654 3210
    return v;
}

__always_inline__ Vector2ui morton_decode_2D(unsigned int v) {
    return Vector2ui(compact_by_1(v >> 1), compact_by_1(v));
}

// Inverse of part_by_2, i.e. delete all bits not at positions divisible by 3 and compacts the rest.
__always_inline__ unsigned int compact_by_2(unsigned int v) {
    v &= 0x09249249;                  // v = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
    v = (v ^ (v >> 2)) & 0x030c30c3;  // v = ---- --98 ---- 76-- --54 ---- 32-- --10
    v = (v ^ (v >> 4)) & 0x0300f00f;  // v = ---- --98 ---- ---- 7654 ---- ---- 3210
    v = (v ^ (v >> 8)) & 0xff0000ff;  // v = ---- --98 ---- ---- ---- ---- 7654 3210
    v = (v ^ (v >> 16)) & 0x000003ff; // v = ---- ---- ---- ---- ---- --98 7654 3210
    return v;
}

__always_inline__ Vector3ui morton_decode_3D(unsigned int v) {
    return Vector3ui(compact_by_2(v >> 2), compact_by_2(v >> 1), compact_by_2(v));
}

} // NS Math
} // NS Bifrost

#endif // _BIFROST_MATH_MORTON_ENCODE_H_
