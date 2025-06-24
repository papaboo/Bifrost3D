// OptiXRenderer environment map.
// ------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _OPTIXRENDERER_ENVIRONMENT_MAP_H_
#define _OPTIXRENDERER_ENVIRONMENT_MAP_H_

#include <OptiXRenderer/Types.h>

#include <Bifrost/Assets/Texture.h>

#include <optixu/optixpp_namespace.h>

//-------------------------------------------------------------------------------------------------
// Forward declarations.
//-------------------------------------------------------------------------------------------------
namespace Bifrost {
namespace Assets {
class InfiniteAreaLight;
}
}

namespace OptiXRenderer {

//-------------------------------------------------------------------------------------------------
// Environment mapping representation.
// Contains the environment texture and the corresponding CDFs and PDF buffers.
// In case the CDF's cannot be constructed the environment returned will 
// contain invalid values, e.g. invalud UID and nullptrs.
// For environment monte carlo sampling see PBRT v2 chapter 14.6.5.
// Future work:
// * Fast Product Importance Sampling of Environment Maps, Conty and Lecocq, 2018, for better material based importance sampling.
//   Perhaps combined with a fixed number of samples pr lower level in the hierarchy to avoid sampling a 2D distribution.
//-------------------------------------------------------------------------------------------------
class EnvironmentMap final {
public:
    //---------------------------------------------------------------------------------------------
    // Constructors and destructor.
    //---------------------------------------------------------------------------------------------
    EnvironmentMap() = default;
    EnvironmentMap(optix::float3 tint)
        : m_environment_map(Bifrost::Assets::Texture::invalid())
        , m_color_texture(nullptr), m_marginal_CDF(nullptr), m_conditional_CDF(nullptr), m_per_pixel_PDF(nullptr) {
        // Initialize the GPU environment light representation.
        m_environment_light = {};
        m_environment_light.set_tint(tint);
    }

    EnvironmentMap(optix::Context& context, const Bifrost::Assets::InfiniteAreaLight& light, optix::float3 tint, optix::TextureSampler environment_sampler);

    EnvironmentMap& operator=(EnvironmentMap&& rhs) {
        m_environment_light = rhs.m_environment_light;
        m_environment_map = rhs.m_environment_map;
        m_color_texture = rhs.m_color_texture; rhs.m_color_texture = nullptr;
        m_marginal_CDF = rhs.m_marginal_CDF; rhs.m_marginal_CDF = nullptr;
        m_conditional_CDF = rhs.m_conditional_CDF; rhs.m_conditional_CDF = nullptr;
        m_per_pixel_PDF = rhs.m_per_pixel_PDF; rhs.m_per_pixel_PDF = nullptr;
        return *this;
    }

    ~EnvironmentMap();

    inline void set_tint(optix::float3 tint) { m_environment_light.set_tint(tint); }

    //---------------------------------------------------------------------------------------------
    // Getters.
    //---------------------------------------------------------------------------------------------
    bool next_event_estimation_possible() const { return m_per_pixel_PDF != optix::TextureSampler(); }

    Bifrost::Assets::Texture get_environment_map() const { return m_environment_map; }

    Light get_light() const {
        Light light;
        light.flags = Light::Environment;
        light.environment = m_environment_light;
        return light;
    }

private:
    EnvironmentMap(EnvironmentMap& other) = delete;
    EnvironmentMap(EnvironmentMap&& other) = delete;
    EnvironmentMap& operator=(EnvironmentMap& rhs) = delete;

    EnvironmentLight m_environment_light;

    Bifrost::Assets::Texture m_environment_map;
    optix::TextureSampler m_color_texture;
    optix::TextureSampler m_marginal_CDF;
    optix::TextureSampler m_conditional_CDF;
    optix::TextureSampler m_per_pixel_PDF;
};

} // NS OptiXRenderer

#endif // _OPTIXRENDERER_ENVIRONMENT_MAP_H_