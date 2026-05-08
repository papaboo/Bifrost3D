// OptiX renderer utilities for working with OptiX host code.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _OPTIXRENDERER_OPTIX_HOST_UTILS_H_
#define _OPTIXRENDERER_OPTIX_HOST_UTILS_H_

#include <OptiXRenderer/Defines.h>

#include <optix_types.h>

namespace OptiXRenderer {

void optix_check_error(OptixResult error, const char* const file, int line);
#define OPTIX_CHECK(error) optix_check_error(error, __FILE__,__LINE__)

} // NS OptiXRenderer

#endif // _OPTIXRENDERER_OPTIX_HOST_UTILS_H_