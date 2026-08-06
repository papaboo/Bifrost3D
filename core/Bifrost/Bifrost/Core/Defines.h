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

#if (defined(__CUDACC__) || defined(__CUDABE__))
#define GPU_COMPILATION 1
#endif

// Patterns for __host__ __device__ programming in CUDA, Mejstrik and Woblistin, 2024
// Pattern: Host device everything (relevant)
#ifdef __CUDACC__
#define _all_archs_ __host__ __device__
#else
#define _all_archs_
#endif

// For backwards compatibility
#define GPU_ENABLED _all_archs_

#if GPU_COMPILATION
#    define _inline_all_archs_ __always_inline__ _all_archs_
#else
#    define _inline_all_archs_ __always_inline__
#endif

#if GPU_COMPILATION
#    define _constant_all_archs_ __constant__
#else
#    define _constant_all_archs_ static const
#endif

namespace Bifrost {

typedef unsigned char byte; // We don't use std::byte, as byte is mainly treated as an int8 and std::byte doesn't support math operations.

} // NS Bifrost

#endif // _BIFROST_CORE_DEFINES_H_
