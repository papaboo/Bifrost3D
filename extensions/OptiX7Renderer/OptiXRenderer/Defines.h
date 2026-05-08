// OptiX shading defines.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _OPTIXRENDERER_SHADING_DEFINES_H_
#define _OPTIXRENDERER_SHADING_DEFINES_H_

#include <Bifrost/Core/Defines.h>

#define __inline_all__ __always_inline__ GPU_ENABLED

#if GPU_COMPILATION
#    define __inline_dev__ __always_inline__ __device__
#else
#    define __inline_dev__ __always_inline__
#endif

#if GPU_COMPILATION
#    define __constant_all__ __constant__
#else
#    define __constant_all__ static const
#endif

// Constants.
#define PIf 3.14159265358979323846f
#define TWO_PIf 6.283185307f
#define RECIP_PIf 0.31830988618379067153776752674503f
#define FLT_MAX 3.402823466e+38F

#if GPU_COMPILATION
#define THROW(e) rtThrow(e)
#else
#define THROW(e) throw e
#endif

#define RT_ASSERT(condition, exception_ID) do { if (!condition) THROW(exception_ID); } while(false)

// OptiX exceptions.
#define OPTIX_NOT_IMPLEMENTED (RT_EXCEPTION_USER + 0)
#define OPTIX_SHADING_WRONG_HEMISPHERE_EXCEPTION (RT_EXCEPTION_USER + 1)
#define OPTIX_LIGHT_EVALUATED_OFF_SURFACE_EXCEPTION (RT_EXCEPTION_USER + 2)
#define OPTIX_NEGATIVE_PDF_SCALE_EXCEPTION (RT_EXCEPTION_USER + 3)
#define OPTIX_DELTA_DIRAC_PDF_ADDITION_EXCEPTION (RT_EXCEPTION_USER + 4)

#endif // _OPTIXRENDERER_SHADING_DEFINES_H_