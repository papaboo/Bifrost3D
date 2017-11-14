// Fits for spherical pivot transformed distributions.
// ------------------------------------------------------------------------------------------------
// Copyright (C) 2017, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#include <OptiXRenderer/SPTDFit.h>
#include <OptiXRenderer/Defines.h>

#include <optixu/optixu_math_namespace.h>

#include <Cogwheel/Assets/Shading/GGXSPTDFit.h>

using namespace optix;

namespace OptiXRenderer {

float4 GGX_SPTD_fit_lookup(float cos_theta, float ggx_alpha) {
    auto params = Cogwheel::Assets::Shading::GGX_SPTD_fit_lookup(cos_theta, ggx_alpha);
    return make_float4(params.x, params.y, params.z, params.z);
}

TextureSampler GGX_SPTD_fit_texture(Context& context) {
    using namespace Cogwheel::Assets::Shading;

    // Create buffer.
    unsigned int width = GGX_SPTD_fit_angular_sample_count;
    unsigned int height = GGX_SPTD_fit_roughness_sample_count;
    Buffer buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT4, width, height);

    float4* data = static_cast<float4*>(buffer->map());
    for (unsigned int i = 0; i < width * height; ++i)
        data[i] = make_float4(GGX_SPTD_fit[i].x, GGX_SPTD_fit[i].y, GGX_SPTD_fit[i].z, GGX_SPTD_fit[i].z);
    buffer->unmap();
    OPTIX_VALIDATE(buffer);

    // ... and wrap it in a texture sampler.
    TextureSampler& texture = context->createTextureSampler();
    texture->setWrapMode(0, RT_WRAP_CLAMP_TO_EDGE);
    texture->setWrapMode(1, RT_WRAP_CLAMP_TO_EDGE);
    texture->setWrapMode(2, RT_WRAP_CLAMP_TO_EDGE);
    texture->setIndexingMode(RT_TEXTURE_INDEX_NORMALIZED_COORDINATES);
    texture->setReadMode(RT_TEXTURE_READ_ELEMENT_TYPE);
    texture->setMaxAnisotropy(1.0f);
    texture->setMipLevelCount(1u);
    texture->setFilteringModes(RT_FILTER_LINEAR, RT_FILTER_LINEAR, RT_FILTER_NONE);
    texture->setArraySize(1u);
    texture->setBuffer(0u, 0u, buffer);
    OPTIX_VALIDATE(texture);

    return texture;
}

} // NS OptiXRenderer
