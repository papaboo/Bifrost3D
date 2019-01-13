// Bifrost mathematical constants.
// ----------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ----------------------------------------------------------------------------

#ifndef _BIFROST_MATH_CONSTANTS_H_
#define _BIFROST_MATH_CONSTANTS_H_

namespace Bifrost {
namespace Math {

template<typename T>
inline T PI() { return T(3.1415926535897932385); }

// The floating point number just below one.
const float nearly_one = 0xffffff / float(1 << 24);

} // NS Math
} // NS Bifrost

#endif // _BIFROST_MATH_CONSTANTS_H_
