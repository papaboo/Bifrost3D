// OptiX renderer functions for the Lambert BSDF.
// ---------------------------------------------------------------------------
// Copyright (C) 2015-2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _OPTIXRENDERER_BSDFS_LAMBERT_H_
#define _OPTIXRENDERER_BSDFS_LAMBERT_H_

#include <OptiXRenderer/Distributions.h>
#include <OptiXRenderer/Types.h>

namespace OptiXRenderer {
namespace Shading {
namespace BSDFs {
namespace Lambert {

__inline_all__ float evaluate() {
    return 1.0f / PIf;
}

__inline_all__ optix::float3 evaluate(const optix::float3& tint) {
    return tint * evaluate();
}

__inline_all__ optix::float3 evaluate(const optix::float3& tint, const optix::float3& wo, const optix::float3& wi) {
    return tint * evaluate();
}

__inline_all__ BSDFSample sample(const optix::float3& tint, optix::float2 random_sample) {
    Distributions::DirectionalSample cosine_sample = Distributions::Cosine::sample(random_sample);
    BSDFSample bsdf_sample;
    bsdf_sample.direction = cosine_sample.direction;
    bsdf_sample.PDF = cosine_sample.PDF;
    bsdf_sample.weight = tint / PIf;
    return bsdf_sample;
}

__inline_all__ float PDF(const optix::float3& wo, const optix::float3& wi) {
    return wi.z / PIf;
}

} // NS Lambert
} // NS BSDFs
} // NS Shading
} // NS OptiXRenderer

#endif // _OPTIXRENDERER_BSDFS_LAMBERT_H_