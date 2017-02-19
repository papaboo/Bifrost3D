// Rho texture.
// ---------------------------------------------------------------------------
// Copyright (C) 2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#include <OptiXRenderer/RhoTexture.h>

#include <Cogwheel/Assets/Shading/GGXWithFresnelRho.h>

using namespace Cogwheel::Assets::Shading;
using namespace optix;

namespace OptiXRenderer {

TextureSampler ggx_with_fresnel_rho_texture(Context& context) {
    // Create buffer.
    unsigned int width = GGX_with_fresnel_angle_sample_count;
    unsigned int height = GGX_with_fresnel_roughness_sample_count;
    Buffer rho_buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT, width, height);

    float* rho_data = static_cast<float*>(rho_buffer->map());
    memcpy(rho_data, GGX_with_fresnel_rho, width * height * sizeof(float));
    rho_buffer->unmap();
    // ggx_with_fresnel_rho_buffer->validate();

    // ... and wrap it in a texture sampler.
    TextureSampler& rho_texture = context->createTextureSampler();
    rho_texture->setWrapMode(0, RT_WRAP_CLAMP_TO_EDGE);
    rho_texture->setWrapMode(1, RT_WRAP_CLAMP_TO_EDGE);
    rho_texture->setWrapMode(2, RT_WRAP_CLAMP_TO_EDGE);
    rho_texture->setIndexingMode(RT_TEXTURE_INDEX_NORMALIZED_COORDINATES);
    rho_texture->setReadMode(RT_TEXTURE_READ_ELEMENT_TYPE);
    rho_texture->setMaxAnisotropy(1.0f);
    rho_texture->setMipLevelCount(1u);
    rho_texture->setFilteringModes(RT_FILTER_NEAREST, RT_FILTER_LINEAR, RT_FILTER_NONE);
    rho_texture->setArraySize(1u);
    rho_texture->setBuffer(0u, 0u, rho_buffer);
    // rho_texture->validate()

    return rho_texture;
}

} // NS OptiXRenderer
