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

//-------------------------------------------------------------------------------------------------
// Environment map CDF calculation. Returns true if the CDF is successfully computed, otherwise false.
// The CDF is ill-defined if fx the input image is completely black or contains negative values.
//-------------------------------------------------------------------------------------------------
inline bool compute_environment_CDFs(Assets::TextureND environment_map, optix::Context& context,
    optix::Buffer& marginal_CDF, optix::Buffer& conditional_CDF, optix::Buffer& per_pixel_PDF) {

    Assets::Image environment = environment_map.get_image();
    unsigned int width = environment.get_width();
    unsigned int height = environment.get_height();

    // Don't importance sample small environment maps. The overhead is simply not worth it.
    // NOTE We might want to delay this decision a bit and take the frequency of the environment map into account.
    //      Even low resolution images can be high frequent you know.
    if ((width * height) < (64 * 32))
        return false;

    per_pixel_PDF = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT, width, height);
    float* per_pixel_PDF_data = static_cast<float*>(per_pixel_PDF->map());
    Assets::InfiniteAreaLight::compute_PDF(environment_map.get_ID(), per_pixel_PDF_data);

    // Perform computations in double precision to maintain some precision during summation.
    double* marginal_CDFd = new double[height + 1];
    double* conditional_CDFd = new double[(width + 1) * height];
    float environment_integral = float(Distribution2D<double>::compute_CDFs(per_pixel_PDF_data, width, height,
        marginal_CDFd, conditional_CDFd));
    if (environment_integral < 0.00001f)
        return false;

    { // Upload data to OptiX buffers.
        // Marginal CDF.
        marginal_CDF = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT, height + 1);
        float* marginal_CDF_data = static_cast<float*>(marginal_CDF->map());
        for (unsigned int y = 0; y < height + 1; ++y)
            marginal_CDF_data[y] = float(marginal_CDFd[y]);
        marginal_CDF->unmap();

        // Conditional CDF.
        conditional_CDF = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT, width + 1, height);
        float* conditional_CDF_data = static_cast<float*>(conditional_CDF->map());
        #pragma omp parallel for schedule(dynamic, 16)
        for (int i = 0; i < int((width + 1) * height); ++i)
            conditional_CDF_data[i] = float(conditional_CDFd[i]);
        conditional_CDF->unmap();

        // Precompute the PDF of the subtended solid angle of each pixel. The PDF must be scaled by 1 / sin_theta before use. See PBRT v2 page 728.
        float PDF_normalization_term = 1.0f / (environment_integral * 2.0f * PIf * PIf);
        #pragma omp parallel for schedule(dynamic, 16)
        for (int i = 0; i < int(width * height); ++i)
            per_pixel_PDF_data[i] *= PDF_normalization_term;
        per_pixel_PDF->unmap();
    }

    delete[] marginal_CDFd;
    delete[] conditional_CDFd;

    return true;
}


EnvironmentMap::EnvironmentMap(optix::Context& context, Assets::Textures::UID environment_map_ID, optix::TextureSampler* texture_cache)
    : m_environment_map_ID(environment_map_ID), color_texture(texture_cache[environment_map_ID]) {

    optix::Buffer marginal_CDF_buffer, conditional_CDF_buffer, per_pixel_PDF_buffer;

    bool success = compute_environment_CDFs(environment_map_ID, context, marginal_CDF_buffer, conditional_CDF_buffer, per_pixel_PDF_buffer);
    if (!success) {
        marginal_CDF = conditional_CDF = per_pixel_PDF = nullptr;
        return;
    }

    { // Marginal CDF sampler.
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
