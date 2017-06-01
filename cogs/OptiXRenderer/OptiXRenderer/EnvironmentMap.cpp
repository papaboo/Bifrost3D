// OptiXRenderer environment map.
// ------------------------------------------------------------------------------------------------
// Copyright (C) 2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#include <OptiXRenderer/EnvironmentMap.h>

#include <Cogwheel/Assets/InfiniteAreaLight.h>
#include <Cogwheel/Math/Color.h>

using namespace Cogwheel;
using namespace Cogwheel::Math;

namespace OptiXRenderer {

EnvironmentMap::EnvironmentMap(optix::Context& context, const Assets::InfiniteAreaLight& light, optix::TextureSampler* texture_cache)
    : m_environment_map_ID(light.get_texture_ID()), color_texture(texture_cache[light.get_texture_ID()]) {

    int width = light.get_width(), height = light.get_height();

    bool is_tiny_image = (width * height) < (64 * 32);
    bool is_dark_image = light.image_integral() < 0.00001f;
    if (is_tiny_image || is_dark_image) {
        marginal_CDF = conditional_CDF = per_pixel_PDF = nullptr;
        return;
    }

    { // Marginal CDF sampler.
        optix::Buffer marginal_CDF_buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT, height + 1);
        float* marginal_CDF_data = static_cast<float*>(marginal_CDF_buffer->map());
        memcpy(marginal_CDF_data, light.get_image_marginal_CDF(), sizeof(float) * (height + 1));
        marginal_CDF_buffer->unmap();

        optix::TextureSampler& texture = marginal_CDF = context->createTextureSampler();
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
        optix::Buffer conditional_CDF_buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT, width + 1, height);
        float* conditional_CDF_data = static_cast<float*>(conditional_CDF_buffer->map());
        memcpy(conditional_CDF_data, light.get_image_conditional_CDF(), sizeof(float) * (width + 1) * height);
        conditional_CDF_buffer->unmap();

        optix::TextureSampler& texture = conditional_CDF = context->createTextureSampler();
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
        optix::Buffer per_pixel_PDF_buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT, width, height);
        float* per_pixel_PDF_data = static_cast<float*>(per_pixel_PDF_buffer->map());

        // Compute PDF from CDF and apply most of the infinite area light normalization.
        // The remaining normalization by sin_theta is done when the specific sampling direction is known.
        float PDF_image_scaling = width * height * light.image_integral();
        float PDF_normalization_term = 1.0f / (float(light.image_integral()) * 2.0f * PIf * PIf);
        float PDF_scale = PDF_image_scaling * PDF_normalization_term;
        #pragma omp parallel for schedule(dynamic, 16)
        for (int y = 0; y < height; ++y) {
            float marginal_PDF = light.get_image_marginal_CDF()[y + 1] - light.get_image_marginal_CDF()[y];

            for (int x = 0; x < width; ++x) {
                const float* const conditional_CDF_offset = light.get_image_conditional_CDF() + x + y * (width + 1);
                float conditional_PDF = conditional_CDF_offset[1] - conditional_CDF_offset[0];

                per_pixel_PDF_data[x + y * width] = marginal_PDF * conditional_PDF * PDF_scale;
            }
        }
        per_pixel_PDF_buffer->unmap();

        optix::TextureSampler& texture = per_pixel_PDF = context->createTextureSampler();
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
    if (marginal_CDF) marginal_CDF->destroy();
    if (conditional_CDF) conditional_CDF->destroy();
    if (per_pixel_PDF) per_pixel_PDF->destroy();
}

} // NS OptiXRenderer
