// OptiXRenderer presampled environment map.
// ------------------------------------------------------------------------------------------------
// Copyright (C) 2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. 
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#include <OptiXRenderer/PresampledEnvironmentMap.h>

#include <Cogwheel/Assets/InfiniteAreaLight.h>
#include <Cogwheel/Math/RNG.h>

using namespace Cogwheel;
using namespace optix;

namespace OptiXRenderer {

PresampledEnvironmentMap::PresampledEnvironmentMap(Context& context, const Assets::InfiniteAreaLight& light, 
                                                   TextureSampler* texture_cache, int sample_count) {

    int width = light.get_width(), height = light.get_height();

    // Check if we should disable importance sampling.
    // To avoid too much branching in the shaders, a presampled environment with 
    // importance sampling disabled only contains a single invalid sampling.
    bool is_tiny_image = (width * height) < (64 * 32);
    bool is_dark_image = light.image_integral() < 0.00001f;
    bool disable_importance_sampling = is_tiny_image || is_dark_image;

    { // Per pixel PDF sampler.
        optix::Buffer per_pixel_PDF_buffer;
        if (disable_importance_sampling) {
            per_pixel_PDF_buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT, 1, 1);
            float* per_pixel_PDF_data = static_cast<float*>(per_pixel_PDF_buffer->map());
            per_pixel_PDF_data[0] = 0.0f;
            per_pixel_PDF_buffer->unmap();
        } else {
            per_pixel_PDF_buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT, width, height);
            float* per_pixel_PDF_data = static_cast<float*>(per_pixel_PDF_buffer->map());
            Assets::InfiniteAreaLightUtils::reconstruct_solid_angle_PDF_sans_sin_theta(light, per_pixel_PDF_data);
            per_pixel_PDF_buffer->unmap();
        }

        m_per_pixel_PDF = context->createTextureSampler();
        m_per_pixel_PDF->setWrapMode(0, RT_WRAP_CLAMP_TO_EDGE);
        m_per_pixel_PDF->setWrapMode(1, RT_WRAP_CLAMP_TO_EDGE);
        m_per_pixel_PDF->setIndexingMode(RT_TEXTURE_INDEX_NORMALIZED_COORDINATES);
        m_per_pixel_PDF->setReadMode(RT_TEXTURE_READ_ELEMENT_TYPE); // Data is already in floating point format, so no need to normalize it.
        m_per_pixel_PDF->setMaxAnisotropy(0.0f);
        m_per_pixel_PDF->setMipLevelCount(1u);
        m_per_pixel_PDF->setFilteringModes(RT_FILTER_NEAREST, RT_FILTER_NEAREST, RT_FILTER_NONE);
        m_per_pixel_PDF->setArraySize(1u);
        m_per_pixel_PDF->setBuffer(0u, 0u, per_pixel_PDF_buffer);
        OPTIX_VALIDATE(m_per_pixel_PDF);
    }

    { // Draw light samples.
        sample_count = sample_count <= 0 ? 8192 : Math::next_power_of_two(sample_count);
        int exponent = (int)log2(sample_count);
        if (disable_importance_sampling)
            sample_count = 1;

        m_samples = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER, sample_count);
        m_samples->setElementSize(sizeof(LightSample));
        LightSample* samples_data = static_cast<LightSample*>(m_samples->map());
        if (disable_importance_sampling) {
            samples_data[0] = LightSample::none();
        } else {
            #pragma omp parallel for schedule(dynamic, 16)
            for (int i = 0; i < sample_count; ++i) {
                // RNG::sample02 is correlated with the seeding strategy in the path tracer (hash ^ brev(accumulation).
                // To avoid this, and keep the drawing the nicely distributed samples that sample02 gives us,
                // we change the order in which the samples are drawn by reversing the bit pattern.
                int s = Math::reverse_bits(i) >> (32 - exponent);
                Assets::LightSample sample = light.sample(Math::RNG::sample02(s));
                samples_data[i].radiance = { sample.radiance.r, sample.radiance.g, sample.radiance.b };
                samples_data[i].PDF = sample.PDF;
                samples_data[i].direction_to_light = { sample.direction_to_light.x, sample.direction_to_light.y, sample.direction_to_light.z };
                samples_data[i].distance = sample.distance;
            }
        }
        m_samples->unmap();
        OPTIX_VALIDATE(m_samples);
    }

    // Fill the presampled light.
    m_light.samples_ID = m_samples->getId();
    m_light.per_pixel_PDF_ID = m_per_pixel_PDF->getId();
    m_light.sample_count = sample_count;
    m_light.environment_map_ID = texture_cache[light.get_texture_ID()]->getId();
}

PresampledEnvironmentMap::~PresampledEnvironmentMap() {
    if (m_samples) m_samples->destroy();
    if (m_per_pixel_PDF) m_per_pixel_PDF->destroy();
}

} // NS OptiXRenderer
