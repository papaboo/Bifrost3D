// OptiX renderer functions for the Lambert BSDF.
// ---------------------------------------------------------------------------
// Copyright (C) 2015-2016, Bifrost. See AUTHORS.txt for authors
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

using namespace optix;

__inline_all__ float evaluate() {
    return RECIP_PIf;
}

__inline_all__ float3 evaluate(const float3& tint) {
    return tint * RECIP_PIf;
}

__inline_all__ float3 evaluate(const float3& tint, const float3& wo, const float3& wi) {
    return tint * RECIP_PIf;
}

__inline_all__ float PDF(const float3& wo, const float3& wi) {
    return Distributions::Cosine::PDF(wi.z);
}

__inline_all__ BSDFResponse evaluate_with_PDF(const float3& tint, const float3& wo, const float3& wi) {
    BSDFResponse response;
    response.weight = evaluate(tint);
    response.PDF = PDF(wo, wi);
    return response;
}

__inline_all__ BSDFSample sample(const float3& tint, float2 random_sample) {
    Distributions::DirectionalSample cosine_sample = Distributions::Cosine::sample(random_sample);
    BSDFSample bsdf_sample;
    bsdf_sample.direction = cosine_sample.direction;
    bsdf_sample.PDF = cosine_sample.PDF;
    bsdf_sample.weight = evaluate(tint);
    return bsdf_sample;
}

} // NS Lambert
} // NS BSDFs
} // NS Shading
} // NS OptiXRenderer

#endif // _OPTIXRENDERER_BSDFS_LAMBERT_H_