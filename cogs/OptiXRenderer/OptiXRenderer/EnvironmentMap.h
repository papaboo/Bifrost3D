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

#include <Cogwheel/Assets/Image.h>
#include <Cogwheel/Assets/Texture.h>
#include <Cogwheel/Math/Color.h>

#include <optixu/optixpp_namespace.h>

namespace OptiXRenderer {

//-----------------------------------------------------------------------------
// Environment mapping representation.
// Contains the environment texture and the corrosponding CDFs and PDF buffers.
// Future work:
// * Specialize pixel importance computation to images with float or 
//   unsigned char components and move it to it's own translation unit.
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

    // Perform computations in double precision to maintain some precision during summation.
    double* marginal_CDFd = new double[height + 1];
    double* conditional_CDFd = new double[(width + 1) * height];
    per_pixel_PDF = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT, width, height);
    float* per_pixel_PDF_data = static_cast<float*>(per_pixel_PDF->map());

    { // Compute pixel importance scaled by projected area of the pixel.
        double* pixel_importance = conditional_CDFd; // Alias to increase readability.

        #pragma omp parallel for schedule(dynamic, 16)
        for (int y = 0; y < int(height); ++y) {
            // PBRT p. 728. Account for the non-uniform surface area of the pixels, e.g. the higher density near the poles.
            float sin_theta = sinf(PIf * (y + 0.5f) / float(height));

            double* per_pixel_PDF_row = pixel_importance + y * width;
            for (unsigned int x = 0; x < width; ++x) {
                // TODO This can be specialized to floating point and unsigned char textures.
                RGB pixel = environment.get_pixel(Vector2ui(x, y)).rgb();
                per_pixel_PDF_row[x] = (pixel.r + pixel.g + pixel.b) * sin_theta; // TODO Use luminance instead? Perhaps define a global importance(RGB / float3) function and use it here and for BRDF sampling.
            }
        }

        // If the texture is unfiltered, then the per pixel importance corresponds to the PDF.
        // If filtering is enabled, then we need to filter the PDF as well.
        // Generally this doesn't change much in terms of convergence, but it helps us to 
        // avoid artefacts in cases where a black pixel would have a PDF of 0,
        // but due to filtering the pixel wouldn't actually be black.
        if (environment_map.get_magnification_filter() == Assets::MagnificationFilter::None) {
            #pragma omp parallel for schedule(dynamic, 16)
            for (int p = 0; p < int(width * height); ++p)
                per_pixel_PDF_data[p] = float(pixel_importance[p]);
        } else {
            #pragma omp parallel for schedule(dynamic, 16)
            for (int y = 0; y < int(height); ++y) {
                // Blur per pixel importance to account for linear interpolation.
                // The pixel's own contribution is 20 / 32.
                // Neighbours on the side contribute by 2 / 32.
                // Neighbours in the corners contribute by 1 / 32.
                // Weights have been estimated based on linear interpolation.
                for (int x = 0; x < int(width); ++x) {
                    float& pixel_PDF = per_pixel_PDF_data[x + y * width];
                    pixel_PDF = float(pixel_importance[x + y * width]) * (20.0f / 32.0f);

                    { // Add contribution from left column.
                        int left_x = x - 1 < 0 ? (width - 1) : (x - 1); // Repeat mode.

                        int lower_left_index = left_x + max(0, y - 1) * width;
                        pixel_PDF += float(pixel_importance[lower_left_index]) * (1.0f / 32.0f);

                        int middle_left_index = left_x + y * width;
                        pixel_PDF += float(pixel_importance[middle_left_index]) * (2.0f / 32.0f);

                        int upper_left_index = left_x + min(int(height) - 1, y + 1) * width;
                        pixel_PDF += float(pixel_importance[upper_left_index]) * (1.0f / 32.0f);
                    }

                    { // Add contribution from right column.
                        int right_x = x + 1 == width ? 0 : (x + 1); // Repeat mode.

                        int lower_right_index = right_x + max(0, y - 1) * width;
                        pixel_PDF += float(pixel_importance[lower_right_index]) * (1.0f / 32.0f);

                        int middle_right_index = right_x + y * width;
                        pixel_PDF += float(pixel_importance[middle_right_index]) * (2.0f / 32.0f);

                        int upper_right_index = right_x + min(int(height) - 1, y + 1) * width;
                        pixel_PDF += float(pixel_importance[upper_right_index]) * (1.0f / 32.0f);
                    }

                    { // Add contribution from middle column. Center was added above.
                        int lower_middle_index = x + max(0, y - 1) * width;
                        pixel_PDF += float(pixel_importance[lower_middle_index]) * (2.0f / 32.0f);

                        int upper_middle_index = x + min(int(height) - 1, y + 1) * width;
                        pixel_PDF += float(pixel_importance[upper_middle_index]) * (2.0f / 32.0f);
                    }
                }
            }
        }
    }

    // Compute conditional CDF.
    #pragma omp parallel for schedule(dynamic, 16)
    for (int y = 0; y < int(height); ++y) {
        float* per_pixel_PDF_row = per_pixel_PDF_data + y * width;
        double* conditional_CDF_row = conditional_CDFd + y * (width + 1);
        conditional_CDF_row[0] = 0.0;
        for (unsigned int x = 0; x < width; ++x)
            conditional_CDF_row[x + 1] = conditional_CDF_row[x] + per_pixel_PDF_row[x];
    }

    // Compute marginal CDF.
    marginal_CDFd[0] = 0.0;
    for (unsigned int y = 0; y < height; ++y)
        marginal_CDFd[y + 1] = marginal_CDFd[y] + conditional_CDFd[(y + 1) * (width + 1) - 1];

    // Integral of the environment map.
    float environment_integral = float(marginal_CDFd[height] / (width * height));

    if (environment_integral < 0.00001f)
        return false;

    // Normalize marginal CDF.
    for (unsigned int y = 1; y < height; ++y)
        marginal_CDFd[y] /= marginal_CDFd[height];
    marginal_CDFd[height] = 1.0;

    // Normalize conditional CDF.
    #pragma omp parallel for schedule(dynamic, 16)
    for (int y = 0; y < int(height); ++y) {
        double* conditional_CDF_row = conditional_CDFd + y * (width + 1);
        if (conditional_CDF_row[width] > 0.0f)
            for (unsigned int x = 1; x < width; ++x)
                conditional_CDF_row[x] /= conditional_CDF_row[width];
        // Last value should always be one. Even in rows with no contribution.
        // This ensures that the binary search is well-defined and will never select the last element.
        conditional_CDF_row[width] = 1.0f;
    }

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