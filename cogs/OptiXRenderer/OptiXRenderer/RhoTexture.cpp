// Rho texture.
// ---------------------------------------------------------------------------
// Copyright (C) 2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#include <OptiXRenderer/RhoTexture.h>
#include <OptiXRenderer/Defines.h>

#include <Cogwheel/Assets/Shading/Fittings.h>

using namespace Cogwheel::Assets::Shading;
using namespace optix;

namespace OptiXRenderer {

TextureSampler ggx_with_fresnel_rho_texture(Context& context) {
    // Create buffer.
    unsigned int width = Rho::GGX_with_fresnel_angle_sample_count;
    unsigned int height = Rho::GGX_with_fresnel_roughness_sample_count;
    Buffer rho_buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_SHORT2, width, height);

    unsigned short* rho_data = static_cast<unsigned short*>(rho_buffer->map());
    for (unsigned int i = 0; i < width * height; ++i) {
        rho_data[2 * i] = unsigned short(Rho::GGX_with_fresnel[i] * 65535 + 0.5f); // No specularity
        rho_data[2 * i + 1] = unsigned short(Rho::GGX[i] * 65535 + 0.5f); // Full specularity
    }
    rho_buffer->unmap();
    OPTIX_VALIDATE(rho_buffer);

    // ... and wrap it in a texture sampler.
    TextureSampler& rho_texture = context->createTextureSampler();
    rho_texture->setWrapMode(0, RT_WRAP_CLAMP_TO_EDGE);
    rho_texture->setWrapMode(1, RT_WRAP_CLAMP_TO_EDGE);
    rho_texture->setWrapMode(2, RT_WRAP_CLAMP_TO_EDGE);
    rho_texture->setIndexingMode(RT_TEXTURE_INDEX_NORMALIZED_COORDINATES);
    rho_texture->setReadMode(RT_TEXTURE_READ_NORMALIZED_FLOAT);
    rho_texture->setMaxAnisotropy(1.0f);
    rho_texture->setMipLevelCount(1u);
    rho_texture->setFilteringModes(RT_FILTER_LINEAR, RT_FILTER_LINEAR, RT_FILTER_NONE);
    rho_texture->setArraySize(1u);
    rho_texture->setBuffer(0u, 0u, rho_buffer);
    OPTIX_VALIDATE(rho_texture);

    return rho_texture;
}

} // NS OptiXRenderer
