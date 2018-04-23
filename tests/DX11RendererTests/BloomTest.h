// DX11Renderer bloom test.
// ------------------------------------------------------------------------------------------------
// Copyright (C) 2018, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _DX11RENDERER_BLOOM_TEST_H_
#define _DX11RENDERER_BLOOM_TEST_H_

#include <gtest/gtest.h>
#include <Utils.h>

#include <Cogwheel/Math/Color.h>
#include <Cogwheel/Math/Half.h>

#include <DX11Renderer/Compositor.h>
#include <DX11Renderer/ToneMapper.h>
#include <DX11Renderer/Utils.h>

namespace DX11Renderer {

// ------------------------------------------------------------------------------------------------
// DX11Renderer bloom test.
// ------------------------------------------------------------------------------------------------
class Bloom : public ::testing::Test {
protected:
    inline OID3D11Buffer create_constants(OID3D11Device1& device, float bloom_threshold) {
        Tonemapper::Constants constants;
        constants.bloom_threshold = bloom_threshold;
        OID3D11Buffer constant_buffer;
        THROW_ON_FAILURE(create_constant_buffer(device, constants, &constant_buffer));
        return constant_buffer;
    }
};

TEST_F(Bloom, dual_kawase_energy_conservation) {
    using namespace Cogwheel::Math;

    auto device = create_performant_device1();
    OID3D11DeviceContext1 context;
    device->GetImmediateContext1(&context);

    // Create image.
    const int width = 64, height = 64;
    const int pixel_count = width * height;
    half4 pixels[pixel_count];
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x) {
            int i = x + y * width;
            RGB color = RGB::black();
            if (x < width / 2)
                color = (y < height / 2) ? RGB::red() : RGB::green();
            else if (y < height / 2)
                color = RGB::blue();
        pixels[i] = { half(color.r), half(color.g), half(color.b), half(1.0f) };
    }
    OID3D11ShaderResourceView pixel_SRV;
    create_texture_2D(*device, DXGI_FORMAT_R16G16B16A16_FLOAT, pixels, width, height, &pixel_SRV);

    OID3D11Buffer constants = create_constants(device, 0.0f);
    context->CSSetConstantBuffers(0, 1, &constants);
    OID3D11SamplerState bilinear_sampler = create_bilinear_sampler(device);
    context->CSSetSamplers(0, 1, &bilinear_sampler);

    // Blur
    DualKawaseBloom bloom = DualKawaseBloom(*device, DX11_SHADER_ROOT);
    auto& filtered_SRV = bloom.filter(context, constants, bilinear_sampler, pixel_SRV, width, height, 1);

    ID3D11Resource* filtered_texture_2D;
    filtered_SRV->GetResource(&filtered_texture_2D);
    half4 filtered_pixels[pixel_count];
    Readback::texture2D(device, context, (ID3D11Texture2D*)filtered_texture_2D, filtered_pixels, filtered_pixels + pixel_count);
    filtered_texture_2D->Release();

    Vector3d summed_pixels = Vector3d::zero();
    Vector3d summed_filtered_pixels = Vector3d::zero();
    for (int p = 0; p < pixel_count; ++p) {
        summed_pixels += Vector3d(pixels[p].x, pixels[p].y, pixels[p].z);
        summed_filtered_pixels += Vector3d(filtered_pixels[p].x, filtered_pixels[p].y, filtered_pixels[p].z);
    }

    EXPECT_VECTOR3F_EQ_PCT((Vector3f)summed_pixels, (Vector3f)summed_filtered_pixels, Vector3f(0.0001f));
}

TEST_F(Bloom, dual_kawase_mirroring) {
    using namespace Cogwheel::Math;

    auto device = create_performant_device1();
    OID3D11DeviceContext1 context;
    device->GetImmediateContext1(&context);

    const int width = 64, height = 64;
    const int pixel_count = width * height;
    
    // Create image.
    half4 pixels[pixel_count];
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x) {
            int i = x + y * width;
            RGB color = RGB::black();
            if (x < width / 2)
                color = (y < height / 2) ? RGB::red() : RGB::green();
            else if (y < height / 2)
                color = RGB::blue();
            pixels[i] = { half(color.r), half(color.g), half(color.b), half(1.0f) };
        }
    OID3D11ShaderResourceView pixel_SRV;
    create_texture_2D(*device, DXGI_FORMAT_R16G16B16A16_FLOAT, pixels, width, height, &pixel_SRV);

    // Create mirrored image.
    half4 mirrored_pixels[pixel_count];
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x) {
            int i = x + y * width;
            int mirrored_i = (width - 1 - x) + (height - 1 - y) * width;
            mirrored_pixels[mirrored_i] = pixels[i];
        }
    OID3D11ShaderResourceView mirrored_pixel_SRV;
    create_texture_2D(*device, DXGI_FORMAT_R16G16B16A16_FLOAT, mirrored_pixels, width, height, &mirrored_pixel_SRV);

    OID3D11Buffer constants = create_constants(device, 0.0f);
    context->CSSetConstantBuffers(0, 1, &constants);
    OID3D11SamplerState bilinear_sampler = create_bilinear_sampler(device);
    context->CSSetSamplers(0, 1, &bilinear_sampler);

    // Blur
    DualKawaseBloom bloom = DualKawaseBloom(*device, DX11_SHADER_ROOT);
    auto& filtered_SRV = bloom.filter(context, constants, bilinear_sampler, pixel_SRV, width, height, 4);
    auto& filtered_mirrored_SRV = bloom.filter(context, constants, bilinear_sampler, mirrored_pixel_SRV, width, height, 4);

    // Readback textures.
    ID3D11Resource* filtered_tex2D;
    filtered_SRV->GetResource(&filtered_tex2D);
    half4 filtered_pixels[pixel_count];
    Readback::texture2D(device, context, (ID3D11Texture2D*)filtered_tex2D, filtered_pixels, filtered_pixels + pixel_count);
    filtered_tex2D->Release();

    ID3D11Resource* filtered_mirrored_tex2D;
    filtered_SRV->GetResource(&filtered_mirrored_tex2D);
    half4 filtered_mirrored_pixels[pixel_count];
    Readback::texture2D(device, context, (ID3D11Texture2D*)filtered_mirrored_tex2D, filtered_mirrored_pixels, filtered_mirrored_pixels + pixel_count);
    filtered_mirrored_tex2D->Release();

    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x) {
            int i = x + y * width;
            half4 pixel = pixels[i];
            int mirrored_i = (width - 1 - x) + (height - 1 - y) * width;
            half4 mirrored_pixel = mirrored_pixels[mirrored_i];
            EXPECT_VECTOR3F_EQ(Vector3f(pixel.x, pixel.y, pixel.z), Vector3f(mirrored_pixel.x, mirrored_pixel.y, mirrored_pixel.z));
        }
}

} // NS DX11Renderer

#endif // _DX11RENDERER_BLOOM_TEST_H_