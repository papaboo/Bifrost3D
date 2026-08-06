// Bifrost mathematical constants.
// ----------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ----------------------------------------------------------------------------

#ifndef _BIFROST_MATH_CONSTANTS_H_
#define _BIFROST_MATH_CONSTANTS_H_

#include <Bifrost/Core/Defines.h>

namespace Bifrost::Math {

template<typename T>
constexpr _inline_all_archs_ T PI() { return T(3.1415926535897932385); }
const float infinity = (float)(1e300 * 1e300); // Multiply two large enough values such that the result is out of range.

// The floating point number just below one.
const float nearly_one = 0xffffff / float(1 << 24);

} // NS Bifrost::Math

#endif // _BIFROST_MATH_CONSTANTS_H_
