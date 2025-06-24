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
    : m_environment_map(light.get_texture()), m_color_texture(environment_sampler) {

    bool is_dark_image = light.image_integral() < 0.00001f;
    if (is_dark_image) {
        m_marginal_CDF = m_conditional_CDF = m_per_pixel_PDF = nullptr;
        return;
    }

    int PDF_width = light.get_PDF_width(), PDF_height = light.get_PDF_height();

    { // Marginal CDF sampler.
        Buffer marginal_CDF_buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT, PDF_height + 1);
        float* marginal_CDF_data = static_cast<float*>(marginal_CDF_buffer->map());
        memcpy(marginal_CDF_data, light.get_image_marginal_CDF(), sizeof(float) * (PDF_height + 1));
        marginal_CDF_buffer->unmap();

        m_marginal_CDF = context->createTextureSampler();
        m_marginal_CDF->setWrapMode(0, RT_WRAP_CLAMP_TO_EDGE);
        m_marginal_CDF->setIndexingMode(RT_TEXTURE_INDEX_ARRAY_INDEX);
        m_marginal_CDF->setReadMode(RT_TEXTURE_READ_ELEMENT_TYPE); // Data is already in floating point format, so no need to normalize it.
        m_marginal_CDF->setMaxAnisotropy(0.0f);
        m_marginal_CDF->setMipLevelCount(1u);
        m_marginal_CDF->setFilteringModes(RT_FILTER_NEAREST, RT_FILTER_NEAREST, RT_FILTER_NONE);
        m_marginal_CDF->setArraySize(1u);
        m_marginal_CDF->setBuffer(marginal_CDF_buffer);
        OPTIX_VALIDATE(m_marginal_CDF);
    }

    { // Conditional CDF sampler.
        Buffer conditional_CDF_buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT, PDF_width + 1, PDF_height);
        float* conditional_CDF_data = static_cast<float*>(conditional_CDF_buffer->map());
        memcpy(conditional_CDF_data, light.get_image_conditional_CDF(), sizeof(float) * (PDF_width + 1) * PDF_height);
        conditional_CDF_buffer->unmap();

        m_conditional_CDF = context->createTextureSampler();
        m_conditional_CDF->setWrapMode(0, RT_WRAP_CLAMP_TO_EDGE);
        m_conditional_CDF->setWrapMode(1, RT_WRAP_CLAMP_TO_EDGE);
        m_conditional_CDF->setIndexingMode(RT_TEXTURE_INDEX_ARRAY_INDEX);
        m_conditional_CDF->setReadMode(RT_TEXTURE_READ_ELEMENT_TYPE); // Data is already in floating point format, so no need to normalize it.
        m_conditional_CDF->setMaxAnisotropy(0.0f);
        m_conditional_CDF->setMipLevelCount(1u);
        m_conditional_CDF->setFilteringModes(RT_FILTER_NEAREST, RT_FILTER_NEAREST, RT_FILTER_NONE);
        m_conditional_CDF->setArraySize(1u);
        m_conditional_CDF->setBuffer(conditional_CDF_buffer);
        OPTIX_VALIDATE(m_conditional_CDF);
    }

    { // Per pixel PDF sampler.
        Buffer per_pixel_PDF_buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT, PDF_width, PDF_height);
        float* per_pixel_PDF_data = static_cast<float*>(per_pixel_PDF_buffer->map());
        Assets::InfiniteAreaLightUtils::reconstruct_solid_angle_PDF_sans_sin_theta(light, per_pixel_PDF_data);
        per_pixel_PDF_buffer->unmap();

        m_per_pixel_PDF = context->createTextureSampler();
        m_per_pixel_PDF->setWrapMode(0, RT_WRAP_CLAMP_TO_EDGE);
        m_per_pixel_PDF->setWrapMode(1, RT_WRAP_CLAMP_TO_EDGE);
        m_per_pixel_PDF->setIndexingMode(RT_TEXTURE_INDEX_NORMALIZED_COORDINATES);
        m_per_pixel_PDF->setReadMode(RT_TEXTURE_READ_ELEMENT_TYPE); // Data is already in floating point format, so no need to normalize it.
        m_per_pixel_PDF->setMaxAnisotropy(0.0f);
        m_per_pixel_PDF->setMipLevelCount(1u);
        m_per_pixel_PDF->setFilteringModes(RT_FILTER_NEAREST, RT_FILTER_NEAREST, RT_FILTER_NONE);
        m_per_pixel_PDF->setArraySize(1u);
        m_per_pixel_PDF->setBuffer(per_pixel_PDF_buffer);
        OPTIX_VALIDATE(m_per_pixel_PDF);
    }

    // Initialize the GPU environment light representation.
    m_environment_light.PDF_width = PDF_width;
    m_environment_light.PDF_height = PDF_height;
    m_environment_light.set_tint(tint);
    m_environment_light.environment_map_ID = m_color_texture->getId();
    if (next_event_estimation_possible()) {
        m_environment_light.marginal_CDF_ID = m_marginal_CDF->getId();
        m_environment_light.conditional_CDF_ID = m_conditional_CDF->getId();
        m_environment_light.per_pixel_PDF_ID = m_per_pixel_PDF->getId();
    } else
        m_environment_light.marginal_CDF_ID = m_environment_light.conditional_CDF_ID = m_environment_light.per_pixel_PDF_ID = RT_TEXTURE_ID_NULL;
}

EnvironmentMap::~EnvironmentMap() {
    if (m_marginal_CDF) m_marginal_CDF->destroy();
    if (m_conditional_CDF) m_conditional_CDF->destroy();
    if (m_per_pixel_PDF) m_per_pixel_PDF->destroy();
}

} // NS OptiXRenderer
