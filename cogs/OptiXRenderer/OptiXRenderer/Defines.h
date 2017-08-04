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
// #define PATH_PDF_FIREFLY_FILTER 1
#define PRESAMPLE_ENVIRONMENT_MAP 1
// When disabling next event estimation to compare images, remember to increase the bounce count by one. 
// Otherwise the longest paths will be missing from the image.
#define ENABLE_NEXT_EVENT_ESTIMATION 1

template <typename T>
void validate_resource(T resource, char* file, int line) {
    try {
        resource->validate();
    } catch (optix::Exception e) {
        printf("Invalid resource in file %s, line %u:\n %s\n", file, line, e.getErrorString().c_str());
        throw e;
    }
}

// Validate macro. Will validate the optix object in debug mode.
#ifdef _DEBUG
#define OPTIX_VALIDATE(o) validate_resource(o, __FILE__,__LINE__)
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

#if GPU_DEVICE
#define THROW(e) rtThrow(e)
#else
#define THROW(e) throw e
#endif


// Constants.
#define PIf 3.14159265358979323846f
#define RECIP_PIf 0.31830988618379067153776752674503f

// Error messages.
#define OPTIX_GGX_WRONG_HEMISPHERE_EXCEPTION (RT_EXCEPTION_USER + 0)

#endif // _OPTIXRENDERER_SHADING_DEFINES_H_