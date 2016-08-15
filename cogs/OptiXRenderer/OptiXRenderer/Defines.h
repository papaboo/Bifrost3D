// OptiX shading defines.
// ---------------------------------------------------------------------------
// Copyright (C) 2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _OPTIXRENDERER_SHADING_DEFINES_H_
#define _OPTIXRENDERER_SHADING_DEFINES_H_

#define DOUBLE_PRECISION_ACCUMULATION_BUFFER 1
#define PATH_PDF_FIREFLY_FILTER 1

// Validate macro. Will validate the optix object in debug mode.
#ifdef _DEBUG
#define OPTIX_VALIDATE(o) o->validate()
#else
#define OPTIX_VALIDATE(o)
#endif

#if (defined(__CUDACC__) || defined(__CUDABE__))
#define GPU_DEVICE 1
#endif

#ifndef __inline_all__
#    if GPU_DEVICE
#        define __inline_all__ __forceinline__ __host__ __device__
#    else
#        define __inline_all__ inline
#    endif
#endif

#ifndef __inline_dev__
#    if GPU_DEVICE
#        define __inline_dev__ __forceinline__ __device__
#    else
#        define __inline_dev__ inline
#    endif
#endif

// Constants.
#define PIf 3.14159265358979323846f
#define RECIP_PIf 0.31830988618379067153776752674503f

// Error messages.
#define OPTIX_GGX_WRONG_HEMISPHERE_EXCEPTION (RT_EXCEPTION_USER + 0)

#endif // _OPTIXRENDERER_SHADING_DEFINES_H_