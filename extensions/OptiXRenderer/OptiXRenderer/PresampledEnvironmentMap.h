// OptiXRenderer presampled environment map.
// ------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License. 
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _OPTIXRENDERER_PRESAMPLED_ENVIRONMENT_MAP_H_
#define _OPTIXRENDERER_PRESAMPLED_ENVIRONMENT_MAP_H_

#include <OptiXRenderer/Types.h>

#include <Bifrost/Assets/Texture.h>

#include <optixu/optixpp_namespace.h>

//-------------------------------------------------------------------------------------------------
// Forward declarations.
//-------------------------------------------------------------------------------------------------
namespace Bifrost::Assets { class InfiniteAreaLight; }

namespace OptiXRenderer {

//-------------------------------------------------------------------------------------------------
// Presampled environment map samples.
// The samples stored in the buffer are of type LightSample.
//-------------------------------------------------------------------------------------------------
class PresampledEnvironmentMap final {
public:
    //---------------------------------------------------------------------------------------------
    // Constructors and destructor.
    //---------------------------------------------------------------------------------------------
    PresampledEnvironmentMap() = default;
    PresampledEnvironmentMap(optix::float3 tint)
        : m_environment_map(Bifrost::Assets::Texture::invalid())
        , m_per_pixel_PDF(nullptr), m_samples(nullptr), m_light(PresampledEnvironmentLight::empty(tint)) {}
    PresampledEnvironmentMap(optix::Context& context, const Bifrost::Assets::InfiniteAreaLight& light, optix::float3 tint,
                             optix::TextureSampler environment_sampler, unsigned int sample_count = 8192);

    PresampledEnvironmentMap& operator=(PresampledEnvironmentMap&& rhs) {
        m_environment_map = rhs.m_environment_map;
        m_per_pixel_PDF = rhs.m_per_pixel_PDF; rhs.m_per_pixel_PDF = nullptr;
        m_samples = rhs.m_samples; rhs.m_samples = nullptr;
        m_light = rhs.m_light;
        return *this;
    }

    ~PresampledEnvironmentMap();

    inline void set_tint(optix::float3 tint) { m_light.tint = tint; }

    //---------------------------------------------------------------------------------------------
    // Getters.
    //---------------------------------------------------------------------------------------------
    Light get_light() const {
        Light light;
        light.flags = Light::PresampledEnvironment;
        light.presampled_environment = m_light;
        return light;
    }

    // Next event estimation is disabled if sample count is 1. The only sample in the buffer is going to be an invalid one.
    inline bool next_event_estimation_possible() const { return m_light.sample_count > 1; }

    Bifrost::Assets::Texture get_environment_map() const { return m_environment_map; }

private:
    PresampledEnvironmentMap(PresampledEnvironmentMap& other) = delete;
    PresampledEnvironmentMap(PresampledEnvironmentMap&& other) = delete;
    PresampledEnvironmentMap& operator=(PresampledEnvironmentMap& rhs) = delete;

    Bifrost::Assets::Texture m_environment_map;

    optix::TextureSampler m_per_pixel_PDF;
    optix::Buffer m_samples;

    PresampledEnvironmentLight m_light;
};

} // NS OptiXRenderer

#endif // _OPTIXRENDERER_PRESAMPLED_ENVIRONMENT_MAP_H_