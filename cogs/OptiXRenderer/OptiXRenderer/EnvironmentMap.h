// OptiXRenderer environment map.
// ---------------------------------------------------------------------------
// Copyright (C) 2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _OPTIXRENDERER_ENVIRONMENT_MAP_H_
#define _OPTIXRENDERER_ENVIRONMENT_MAP_H_

#include <OptiXRenderer/Types.h>

#include <Cogwheel/Assets/InfiniteAreaLight.h>
#include <Cogwheel/Math/Color.h>

#include <optixu/optixpp_namespace.h>

namespace OptiXRenderer {

//-----------------------------------------------------------------------------
// Environment mapping representation.
// Contains the environment texture and the corrosponding CDFs and PDF buffers.
// Future work:
// * Structuring the CDF as 'breath first' should improve the cache hit rate 
//   of the first couple of lookups when we do binary search or? Profile!
// * Create a kd-tree'ish structure instead of the current CDFs.
//   Then instead of using a single float to update lower or higher, 
//   store four children pr node, so we can load a uint4 during traversal.
//   This should make better use of the bandwidth.
// * Sample the environment based on cos_theta between the environment sample 
//   direction and the normal to sample more optimally compared to the total 
//   contribution of the environment. This can be achieved easily when sampling 
//   by reestimating the CDF values by cos_theta between their direction and 
//   the normal. Would it also be possible to estimate the per pixel PDF 
//   without traversing the whole acceleration structure ? And how expensive is it?
//   Or can I do this by simply changing the distribution of the two 
//   random numbers before searching the CDF?
//-----------------------------------------------------------------------------
struct EnvironmentMap {
    Cogwheel::Assets::TextureND map;
    optix::TextureSampler marginal_CDF;
    optix::TextureSampler conditional_CDF;
    optix::TextureSampler per_pixel_PDF;

    bool next_event_estimation_possible() { return per_pixel_PDF != optix::TextureSampler(); }

    EnvironmentLight to_light_source(optix::TextureSampler* texture_cache) {
        EnvironmentLight light;
        light.width = map.get_image().get_width();
        light.height = map.get_image().get_height();
        light.environment_map_ID = texture_cache[map.get_ID()]->getId();
        if (next_event_estimation_possible()) {
            light.marginal_CDF_ID = marginal_CDF->getId();
            light.conditional_CDF_ID = conditional_CDF->getId();
            light.per_pixel_PDF_ID = per_pixel_PDF->getId();
        } else
            light.marginal_CDF_ID = light.conditional_CDF_ID = light.per_pixel_PDF_ID = RT_TEXTURE_ID_NULL;
        return light;
    }
};

//----------------------------------------------------------------------------
// Environment map CDF calculation. Returns true if the CDF is successfully 
// computed, otherwise false.
// The CDF is ill-defined if fx the input image is completely black 
// or contains negative values.
//----------------------------------------------------------------------------
bool compute_environment_CDFs(Cogwheel::Assets::TextureND environment_map, optix::Context& context,
                              optix::Buffer& marginal_CDF, optix::Buffer& conditional_CDF, optix::Buffer& per_pixel_PDF) {

    using namespace Cogwheel;
    using namespace Cogwheel::Math;

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
    Cogwheel::Assets::InfiniteAreaLight::compute_PDF(environment_map.get_ID(), per_pixel_PDF_data);

    // Perform computations in double precision to maintain some precision during summation.
    double* marginal_CDFd = new double[height + 1];
    double* conditional_CDFd = new double[(width + 1) * height];
    float environment_integral = float(Cogwheel::Math::Distribution2D<double>::compute_CDFs(per_pixel_PDF_data, width, height, 
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

//----------------------------------------------------------------------------
// Creates an environment representation from an environment map.
// This includes constructing the environment map CDFs and per pixel PDF.
// In case the CDF's cannot be constructed the environment returned will 
// contain invalid values, e.g. invalud UID and nullptrs.
// For environment monte carlo sampling see PBRT v2 chapter 14.6.5.
//----------------------------------------------------------------------------
EnvironmentMap create_environment(Cogwheel::Assets::TextureND environment_map, optix::Context& context) {

    optix::Buffer marginal_CDF, conditional_CDF, per_pixel_PDF;
    bool success = compute_environment_CDFs(environment_map, context, marginal_CDF, conditional_CDF, per_pixel_PDF);
    if (!success) {
        EnvironmentMap env = { environment_map.get_ID(), nullptr, nullptr, nullptr };
        return env;
    }

    EnvironmentMap environment;
    environment.map = environment_map;

    { // Marginal CDF sampler.
        optix::TextureSampler& texture = environment.marginal_CDF = context->createTextureSampler();
        texture->setWrapMode(0, RT_WRAP_CLAMP_TO_EDGE);
        texture->setIndexingMode(RT_TEXTURE_INDEX_ARRAY_INDEX);
        texture->setReadMode(RT_TEXTURE_READ_ELEMENT_TYPE); // Data is already in floating point format, so no need to normalize it.
        texture->setMaxAnisotropy(0.0f);
        texture->setMipLevelCount(1u);
        texture->setFilteringModes(RT_FILTER_NEAREST, RT_FILTER_NEAREST, RT_FILTER_NONE);
        texture->setArraySize(1u);
        texture->setBuffer(0u, 0u, marginal_CDF);
        OPTIX_VALIDATE(texture);
    }

    { // Conditional CDF sampler.
        optix::TextureSampler& texture = environment.conditional_CDF = context->createTextureSampler();
        texture->setWrapMode(0, RT_WRAP_CLAMP_TO_EDGE);
        texture->setWrapMode(1, RT_WRAP_CLAMP_TO_EDGE);
        texture->setIndexingMode(RT_TEXTURE_INDEX_ARRAY_INDEX);
        texture->setReadMode(RT_TEXTURE_READ_ELEMENT_TYPE); // Data is already in floating point format, so no need to normalize it.
        texture->setMaxAnisotropy(0.0f);
        texture->setMipLevelCount(1u);
        texture->setFilteringModes(RT_FILTER_NEAREST, RT_FILTER_NEAREST, RT_FILTER_NONE);
        texture->setArraySize(1u);
        texture->setBuffer(0u, 0u, conditional_CDF);
        OPTIX_VALIDATE(texture);
    }

    { // Per pixel PDF sampler.
        optix::TextureSampler& texture = environment.per_pixel_PDF = context->createTextureSampler();
        texture->setWrapMode(0, RT_WRAP_CLAMP_TO_EDGE);
        texture->setWrapMode(1, RT_WRAP_CLAMP_TO_EDGE);
        texture->setIndexingMode(RT_TEXTURE_INDEX_NORMALIZED_COORDINATES);
        texture->setReadMode(RT_TEXTURE_READ_ELEMENT_TYPE); // Data is already in floating point format, so no need to normalize it.
        texture->setMaxAnisotropy(0.0f);
        texture->setMipLevelCount(1u);
        texture->setFilteringModes(RT_FILTER_NEAREST, RT_FILTER_NEAREST, RT_FILTER_NONE);
        texture->setArraySize(1u);
        texture->setBuffer(0u, 0u, per_pixel_PDF);
        OPTIX_VALIDATE(texture);
    }

    return environment;
}

} // NS OptiXRenderer

#endif // _OPTIXRENDERER_ENVIRONMENT_MAP_H_