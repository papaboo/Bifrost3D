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

// Pick an RNG by defining which is used.
// Define LCG_RNG to 1 to use a simple LCG for random number generation.
#define LCG_RNG 0
// Define PRACTICAL_SOBOL_RNG to 1 to use the QMC RNG from Practical Hash-Based Owen Scrambling.
#define PRACTICAL_SOBOL_RNG 1
static_assert((LCG_RNG + PRACTICAL_SOBOL_RNG) == 1, "Only one RNG can be selected.");

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

#if (defined(__CUDACC__) || defined(__CUDABE__))
#define GPU_DEVICE 1
#endif

#ifndef __inline_all__
#    if GPU_DEVICE
#        define __inline_all__ __always_inline__ __host__ __device__
#    else
#        define __inline_all__ __always_inline__
#    endif
#endif

#ifndef __inline_dev__
#    if GPU_DEVICE
#        define __inline_dev__ __always_inline__ __device__
#    else
#        define __inline_dev__ __always_inline__
#    endif
#endif

#if GPU_DEVICE
#    define __constant_all__ __constant__
#else
#    define __constant_all__ static const
#endif

#if GPU_DEVICE
#define THROW(e) rtThrow(e)
#else
#define THROW(e) throw e
#endif

#define RT_ASSERT(condition, exception_ID) do { if (!condition) THROW(exception_ID); } while(false)

// Constants.
#define PIf 3.14159265358979323846f
#define TWO_PIf 6.283185307f
#define RECIP_PIf 0.31830988618379067153776752674503f
#define FLT_MAX 3.402823466e+38F

// OptiX exceptions.
#define OPTIX_NOT_IMPLEMENTED (RT_EXCEPTION_USER + 0)
#define OPTIX_SHADING_WRONG_HEMISPHERE_EXCEPTION (RT_EXCEPTION_USER + 1)
#define OPTIX_LIGHT_EVALUATED_OFF_SURFACE_EXCEPTION (RT_EXCEPTION_USER + 2)
#define OPTIX_NEGATIVE_PDF_SCALE_EXCEPTION (RT_EXCEPTION_USER + 3)
#define OPTIX_DELTA_DIRAC_PDF_ADDITION_EXCEPTION (RT_EXCEPTION_USER + 4)

#endif // _OPTIXRENDERER_SHADING_DEFINES_H_