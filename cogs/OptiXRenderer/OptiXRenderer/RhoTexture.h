// Rho texture.
// ---------------------------------------------------------------------------
// Copyright (C) 2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _OPTIXRENDERER_RHO_TEXTURE_H_
#define _OPTIXRENDERER_RHO_TEXTURE_H_

#include <optixu/optixpp_namespace.h>
#undef RGB

namespace OptiXRenderer {

optix::TextureSampler default_shading_rho_texture(optix::Context& context);

} // NS OptiXRenderer

#endif // _OPTIXRENDERER_RHO_TEXTURE_H_