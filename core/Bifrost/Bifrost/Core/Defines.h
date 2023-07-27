// Bifrost defines.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _BIFROST_CORE_DEFINES_H_
#define _BIFROST_CORE_DEFINES_H_

#ifdef _MSC_VER
#define __always_inline__ __forceinline
#else
#define __always_inline__ inline
#endif

namespace Bifrost {

typedef unsigned char byte; // We don't use std::byte, as byte is mainly treated as an int8 and std::byte doesn't support math operations.

} // NS Bifrost

#endif // _BIFROST_CORE_DEFINES_H_
