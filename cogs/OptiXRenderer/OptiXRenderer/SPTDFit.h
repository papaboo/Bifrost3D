// Fits for spherical pivot transformed distributions.
// ------------------------------------------------------------------------------------------------
// Copyright (C) 2017, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _OPTIXRENDERER_SPTD_FIT_H_
#define _OPTIXRENDERER_SPTD_FIT_H_

#include <optixu/optixpp_namespace.h>
#undef RGB

namespace OptiXRenderer {

optix::float4 ggx_sptd_fit(float cos_theta, float ggx_alpha);

optix::TextureSampler ggx_sptd_fit_texture(optix::Context& context);

} // NS OptiXRenderer

#endif // _OPTIXRENDERER_SPTD_FIT_H_