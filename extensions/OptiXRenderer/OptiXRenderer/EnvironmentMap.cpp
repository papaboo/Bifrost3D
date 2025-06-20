// OptiXRenderer environment map.
// ------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#include <OptiXRenderer/EnvironmentMap.h>

#include <Bifrost/Assets/InfiniteAreaLight.h>
#include <Bifrost/Math/Color.h>

using namespace Bifrost;
using namespace Bifrost::Math;
using namespace optix;

namespace OptiXRenderer {

EnvironmentMap::EnvironmentMap(Context& context, const Assets::InfiniteAreaLight& light, float3 tint, TextureSampler environment_sampler)
    : m_environment_map_ID(light.get_texture().get_ID()), m_color_texture(environment_sampler), m_tint(tint) {

    int width = light.get_width(), height = light.get_height();

    bool is_tiny_image = (width * height) < (64 * 32);
    bool is_dark_image = light.image_integral() < 0.00001f;
    if (is_tiny_image || is_dark_image) {
        m_marginal_CDF = m_conditional_CDF = m_per_pixel_PDF = nullptr;
        return;
    }

    { // Marginal CDF sampler.
        Buffer marginal_CDF_buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT, height + 1);
        float* marginal_CDF_data = static_cast<float*>(marginal_CDF_buffer->map());
        memcpy(marginal_CDF_data, light.get_image_marginal_CDF(), sizeof(float) * (height + 1));
        marginal_CDF_buffer->unmap();

        TextureSampler& texture = m_marginal_CDF = context->createTextureSampler();
        texture->setWrapMode(0, RT_WRAP_CLAMP_TO_EDGE);
        texture->setIndexingMode(RT_TEXTURE_INDEX_ARRAY_INDEX);
        texture->setReadMode(RT_TEXTURE_READ_ELEMENT_TYPE); // Data is already in floating point format, so no need to normalize it.
        texture->setMaxAnisotropy(0.0f);
        texture->setMipLevelCount(1u);
        texture->setFilteringModes(RT_FILTER_NEAREST, RT_FILTER_NEAREST, RT_FILTER_NONE);
        texture->setArraySize(1u);
        texture->setBuffer(0u, 0u, marginal_CDF_buffer);
        OPTIX_VALIDATE(texture);
    }

    { // Conditional CDF sampler.
        Buffer conditional_CDF_buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT, width + 1, height);
        float* conditional_CDF_data = static_cast<float*>(conditional_CDF_buffer->map());
        memcpy(conditional_CDF_data, light.get_image_conditional_CDF(), sizeof(float) * (width + 1) * height);
        conditional_CDF_buffer->unmap();

        TextureSampler& texture = m_conditional_CDF = context->createTextureSampler();
        texture->setWrapMode(0, RT_WRAP_CLAMP_TO_EDGE);
        texture->setWrapMode(1, RT_WRAP_CLAMP_TO_EDGE);
        texture->setIndexingMode(RT_TEXTURE_INDEX_ARRAY_INDEX);
        texture->setReadMode(RT_TEXTURE_READ_ELEMENT_TYPE); // Data is already in floating point format, so no need to normalize it.
        texture->setMaxAnisotropy(0.0f);
        texture->setMipLevelCount(1u);
        texture->setFilteringModes(RT_FILTER_NEAREST, RT_FILTER_NEAREST, RT_FILTER_NONE);
        texture->setArraySize(1u);
        texture->setBuffer(0u, 0u, conditional_CDF_buffer);
        OPTIX_VALIDATE(texture);
    }

    { // Per pixel PDF sampler.
        Buffer per_pixel_PDF_buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT, width, height);
        float* per_pixel_PDF_data = static_cast<float*>(per_pixel_PDF_buffer->map());
        Assets::InfiniteAreaLightUtils::reconstruct_solid_angle_PDF_sans_sin_theta(light, per_pixel_PDF_data);
        per_pixel_PDF_buffer->unmap();

        TextureSampler& texture = m_per_pixel_PDF = context->createTextureSampler();
        texture->setWrapMode(0, RT_WRAP_CLAMP_TO_EDGE);
        texture->setWrapMode(1, RT_WRAP_CLAMP_TO_EDGE);
        texture->setIndexingMode(RT_TEXTURE_INDEX_NORMALIZED_COORDINATES);
        texture->setReadMode(RT_TEXTURE_READ_ELEMENT_TYPE); // Data is already in floating point format, so no need to normalize it.
        texture->setMaxAnisotropy(0.0f);
        texture->setMipLevelCount(1u);
        texture->setFilteringModes(RT_FILTER_NEAREST, RT_FILTER_NEAREST, RT_FILTER_NONE);
        texture->setArraySize(1u);
        texture->setBuffer(0u, 0u, per_pixel_PDF_buffer);
        OPTIX_VALIDATE(texture);
    }
}

EnvironmentMap::~EnvironmentMap() {
    if (m_marginal_CDF) m_marginal_CDF->destroy();
    if (m_conditional_CDF) m_conditional_CDF->destroy();
    if (m_per_pixel_PDF) m_per_pixel_PDF->destroy();
}

} // NS OptiXRenderer
