// Bifrost morton encoding functions.
// ------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _BIFROST_MATH_MORTON_ENCODE_H_
#define _BIFROST_MATH_MORTON_ENCODE_H_

namespace Bifrost {
namespace Math {

// Insert a 0 bit in between each of the 16 low bits of v.
inline unsigned int part_by_1(unsigned int v) {
    v &= 0x0000ffff;                 // v = ---- ---- ---- ---- fedc ba98 7654 3210
    v = (v ^ (v << 8)) & 0x00ff00ff; // v = ---- ---- fedc ba98 ---- ---- 7654 3210
    v = (v ^ (v << 4)) & 0x0f0f0f0f; // v = ---- fedc ---- ba98 ---- 7654 ---- 3210
    v = (v ^ (v << 2)) & 0x33333333; // v = --fe --dc --ba --98 --76 --54 --32 --10
    v = (v ^ (v << 1)) & 0x55555555; // v = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
    return v;
}

inline unsigned int morton_encode(unsigned int x, unsigned int y) {
    return part_by_1(y) | (part_by_1(x) << 1);
}

} // NS Math
} // NS Bifrost

#endif // _BIFROST_MATH_MORTON_ENCODE_H_
