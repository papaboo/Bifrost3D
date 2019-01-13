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
// * Sample the environment based on cos_theta between the environment sample direction and 
//   the normal to sample more optimally compared to the total contribution of the environment. 
//-------------------------------------------------------------------------------------------------
class EnvironmentMap final {
public:
    //---------------------------------------------------------------------------------------------
    // Constructors and destructor.
    //---------------------------------------------------------------------------------------------
    EnvironmentMap()
        : m_environment_map_ID(Bifrost::Assets::Textures::UID::invalid_UID())
        , color_texture(nullptr), marginal_CDF(nullptr), conditional_CDF(nullptr), per_pixel_PDF(nullptr) { }

    EnvironmentMap(optix::Context& context, const Bifrost::Assets::InfiniteAreaLight& light, optix::TextureSampler* texture_cache);

    EnvironmentMap& operator=(EnvironmentMap&& rhs) {
        m_environment_map_ID = rhs.m_environment_map_ID;
        color_texture = rhs.color_texture; rhs.color_texture = nullptr;
        marginal_CDF = rhs.marginal_CDF; rhs.marginal_CDF = nullptr;
        conditional_CDF = rhs.conditional_CDF; rhs.conditional_CDF = nullptr;
        per_pixel_PDF = rhs.per_pixel_PDF; rhs.per_pixel_PDF = nullptr;
        return *this;
    }

    ~EnvironmentMap();

    //---------------------------------------------------------------------------------------------
    // Getters.
    //---------------------------------------------------------------------------------------------
    bool next_event_estimation_possible() const { return per_pixel_PDF != optix::TextureSampler(); }

    Bifrost::Assets::Textures::UID get_environment_map_ID() const { return m_environment_map_ID; }

    Light get_light() {
        EnvironmentLight env_light;
        Bifrost::Assets::Image image = Bifrost::Assets::Textures::get_image_ID(m_environment_map_ID);
        env_light.width = image.get_width();
        env_light.height = image.get_height();
        env_light.environment_map_ID = color_texture->getId();
        if (next_event_estimation_possible()) {
            env_light.marginal_CDF_ID = marginal_CDF->getId();
            env_light.conditional_CDF_ID = conditional_CDF->getId();
            env_light.per_pixel_PDF_ID = per_pixel_PDF->getId();
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
    optix::TextureSampler color_texture;
    optix::TextureSampler marginal_CDF;
    optix::TextureSampler conditional_CDF;
    optix::TextureSampler per_pixel_PDF;
};

} // NS OptiXRenderer

#endif // _OPTIXRENDERER_ENVIRONMENT_MAP_H_