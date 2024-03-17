// OptiX renderer shading model interface.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _OPTIXRENDERER_SHADING_MODEL_SHADING_MODEL_INTERFACE_H_
#define _OPTIXRENDERER_SHADING_MODEL_SHADING_MODEL_INTERFACE_H_

#include <OptiXRenderer/Shading/BSDFs/Lambert.h>

namespace OptiXRenderer {
namespace Shading {
namespace ShadingModels {

// ---------------------------------------------------------------------------
// The shading model interface.
// As we don't use virtual calls, the interface is meant to be used for function that are templated with a shading model,
// to allow the compiler and developer to easily see the shading model interface contract.
// ---------------------------------------------------------------------------
template <class ShadingModel>
class IShadingModel : public ShadingModel {
private:
    ShadingModel m_shading_model;

public:

    __inline_all__ IShadingModel(ShadingModel shading_model) : m_shading_model(shading_model) { }

    __inline_all__ optix::float3 evaluate(optix::float3 wo, optix::float3 wi) const {
        return ShadingModel::evaluate(m_tint, wo, wi);
    }

    __inline_all__ float PDF(optix::float3 wo, optix::float3 wi) const {
        return ShadingModel::PDF(wo, wi);
    }

    __inline_all__ BSDFResponse evaluate_with_PDF(optix::float3 wo, optix::float3 wi) const {
        return ShadingModel::evaluate_with_PDF(wo, wi);
    }

    __inline_all__ BSDFSample sample(optix::float3 wo, optix::float3 random_sample) const {
        return ShadingModel::sample(wo, random_sample);
    }
};

} // NS ShadingModels
} // NS Shading
} // NS OptiXRenderer

#endif // _OPTIXRENDERER_SHADING_MODEL_SHADING_MODEL_INTERFACE_H_