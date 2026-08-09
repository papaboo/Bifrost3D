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

#define DOUBLE_PRECISION_ACCUMULATION_BUFFER 1
#define PRESAMPLE_ENVIRONMENT_MAP 1

template <typename T>
void validate_optix_resource(T resource, char* file, int line) {
    try {
        resource->validate();
    } catch (optix::Exception e) {
        printf("Invalid resource in file %s, line %u:\n%s\n", file, line, e.getErrorString().c_str());
        throw e;
    }
}

// Validate macro. Will validate the optix object in debug mode.
#ifdef _DEBUG
#define OPTIX_VALIDATE(o) validate_optix_resource(o, __FILE__,__LINE__)
#else
#define OPTIX_VALIDATE(o)
#endif

#ifdef GPU_COMPILATION
#define GPU_DEVICE 1
#endif

#define __inline_all__ _inline_all_archs_

#ifndef __inline_dev__
#    if GPU_COMPILATION
#        define __inline_dev__ __always_inline__ __device__
#    else
#        define __inline_dev__ __always_inline__
#    endif
#endif

#define __constant_all__ _constant_all_archs_

#if GPU_COMPILATION
#define THROW(e) rtThrow(e)
#else
#define THROW(e) throw e
#endif

#define RT_ASSERT(condition, exception_ID) do { if (!condition) THROW(exception_ID); } while(false)

// Constants.
constexpr float PIf = 3.14159265358979323846f;
constexpr float TWO_PIf = 6.283185307f;
constexpr float RECIP_PIf = 0.31830988618379067153776752674503f;

// OptiX exceptions.
#define OPTIX_NOT_IMPLEMENTED (RT_EXCEPTION_USER + 0)
#define OPTIX_SHADING_WRONG_HEMISPHERE_EXCEPTION (RT_EXCEPTION_USER + 1)
#define OPTIX_LIGHT_EVALUATED_OFF_SURFACE_EXCEPTION (RT_EXCEPTION_USER + 2)
#define OPTIX_NEGATIVE_PDF_SCALE_EXCEPTION (RT_EXCEPTION_USER + 3)
#define OPTIX_DELTA_DIRAC_PDF_ADDITION_EXCEPTION (RT_EXCEPTION_USER + 4)

#endif // _OPTIXRENDERER_SHADING_DEFINES_H_