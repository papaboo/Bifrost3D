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
        : m_environment_map_ID(Bifrost::Assets::Textures::UID::invalid_UID())
        , m_color_texture(nullptr), m_marginal_CDF(nullptr), m_conditional_CDF(nullptr), m_per_pixel_PDF(nullptr)
        , m_tint(tint) { }

    EnvironmentMap(optix::Context& context, const Bifrost::Assets::InfiniteAreaLight& light, optix::float3 tint, optix::TextureSampler* texture_cache);

    EnvironmentMap& operator=(EnvironmentMap&& rhs) {
        m_environment_map_ID = rhs.m_environment_map_ID;
        m_color_texture = rhs.m_color_texture; rhs.m_color_texture = nullptr;
        m_marginal_CDF = rhs.m_marginal_CDF; rhs.m_marginal_CDF = nullptr;
        m_conditional_CDF = rhs.m_conditional_CDF; rhs.m_conditional_CDF = nullptr;
        m_per_pixel_PDF = rhs.m_per_pixel_PDF; rhs.m_per_pixel_PDF = nullptr;
        return *this;
    }

    ~EnvironmentMap();

    inline void set_tint(optix::float3 tint) { m_tint = tint; }

    //---------------------------------------------------------------------------------------------
    // Getters.
    //---------------------------------------------------------------------------------------------
    bool next_event_estimation_possible() const { return m_per_pixel_PDF != optix::TextureSampler(); }

    Bifrost::Assets::Textures::UID get_environment_map_ID() const { return m_environment_map_ID; }

    Light get_light() const {
        EnvironmentLight env_light;
        Bifrost::Assets::Image image = Bifrost::Assets::Textures::get_image_ID(m_environment_map_ID);
        env_light.width = image.get_width();
        env_light.height = image.get_height();
        env_light.set_tint(m_tint);
        env_light.environment_map_ID = m_color_texture->getId();
        if (next_event_estimation_possible()) {
            env_light.marginal_CDF_ID = m_marginal_CDF->getId();
            env_light.conditional_CDF_ID = m_conditional_CDF->getId();
            env_light.per_pixel_PDF_ID = m_per_pixel_PDF->getId();
        } else
            env_light.marginal_CDF_ID = env_light.conditional_CDF_ID = env_light.per_pixel_PDF_ID = RT_TEXTURE_ID_NULL;

        Light light;
        light.flags = Light::Environment;
        light.environment = env_light;
        return light;
    }

private:
    EnvironmentMap(EnvironmentMap& other) = delete;
    EnvironmentMap(EnvironmentMap&& other) = delete;
    EnvironmentMap& operator=(EnvironmentMap& rhs) = delete;

    Bifrost::Assets::Textures::UID m_environment_map_ID;
    optix::TextureSampler m_color_texture;
    optix::TextureSampler m_marginal_CDF;
    optix::TextureSampler m_conditional_CDF;
    optix::TextureSampler m_per_pixel_PDF;

    optix::float3 m_tint;
};

} // NS OptiXRenderer

#endif // _OPTIXRENDERER_ENVIRONMENT_MAP_H_