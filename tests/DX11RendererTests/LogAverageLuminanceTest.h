// DX11Renderer linear exposure from geometric mean of log-average luminance test.
// ------------------------------------------------------------------------------------------------
// Copyright (C) 2018, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _DX11RENDERER_LOG_AVERAGE_LUMINANCE_TEST_H_
#define _DX11RENDERER_LOG_AVERAGE_LUMINANCE_TEST_H_

#include <gtest/gtest.h>
#include <Utils.h>

#include <Cogwheel/Math/Half.h>
#include <Cogwheel/Math/Vector.h>
#include <Cogwheel/Math/Utils.h>

#include <DX11Renderer/Compositor.h>
#include <DX11Renderer/ToneMapper.h>
#include <DX11Renderer/Utils.h>

namespace DX11Renderer {

// ------------------------------------------------------------------------------------------------
// DX11Renderer linear exposure from geometric mean of log-average luminance test fixture.
// ------------------------------------------------------------------------------------------------
class LogAverageLuminanceFixture : public ::testing::Test {
protected:

    inline OID3D11ShaderResourceView create_texture_SRV(OID3D11Device1& device, unsigned int width, unsigned int height, half4* pixels) {
        D3D11_TEXTURE2D_DESC tex_desc;
        tex_desc.Width = width;
        tex_desc.Height = height;
        tex_desc.MipLevels = 1;
        tex_desc.ArraySize = 1;
        tex_desc.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
        tex_desc.SampleDesc.Count = 1;
        tex_desc.SampleDesc.Quality = 0;
        tex_desc.Usage = D3D11_USAGE_DEFAULT;
        tex_desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        tex_desc.CPUAccessFlags = 0;
        tex_desc.MiscFlags = 0;

        D3D11_SUBRESOURCE_DATA resource_data = {};
        resource_data.SysMemPitch = sizeof_dx_format(tex_desc.Format) * width;
        resource_data.SysMemSlicePitch = resource_data.SysMemPitch * height;
        resource_data.pSysMem = pixels;

        OID3D11Texture2D texture;
        THROW_ON_FAILURE(device->CreateTexture2D(&tex_desc, &resource_data, &texture));
        OID3D11ShaderResourceView texture_SRV;
        THROW_ON_FAILURE(device->CreateShaderResourceView(texture, nullptr, &texture_SRV));
        return texture_SRV;
    }
};

TEST_F(LogAverageLuminanceFixture, tiny_image) {
    using namespace Cogwheel::Math;

    auto device = create_performant_device1();
    OID3D11DeviceContext1 context;
    device->GetImmediateContext1(&context);

    LogAverageLuminance& log_average_exposure = LogAverageLuminance(*device, DX11_SHADER_ROOT);

    float min_log_luminance = -8;
    float max_log_luminance = 4;
    OID3D11Buffer constant_buffer = create_tonemapping_constants(device, min_log_luminance, max_log_luminance);

    // Image with one element in each bucket.
    const unsigned int pixel_count = 64;
    half4 pixels[pixel_count];
    for (int i = 0; i < pixel_count; ++i) {
        half g = half(float(i));
        pixels[i] = { g, g, g, half(1.0f) };
    }
    OID3D11ShaderResourceView pixel_SRV = create_texture_SRV(device, pixel_count, 1, pixels);

    OID3D11UnorderedAccessView log_average_UAV;
    OID3D11Buffer log_average_buffer = create_default_buffer(device, DXGI_FORMAT_R32_FLOAT, 1, nullptr, &log_average_UAV);
    log_average_exposure.compute_log_average(*context, constant_buffer, pixel_SRV, pixel_count, log_average_UAV);

    float log_average_GPU;
    Readback::buffer(device, context, log_average_buffer, &log_average_GPU, &log_average_GPU + 1);

    double log_average = 0.0;
    for (int i = 0; i < pixel_count; ++i)
        log_average += log2(fmaxf(pixels[i].x, 0.0001f));
    log_average /= pixel_count;

    EXPECT_FLOAT_EQ((float)log_average, log_average_GPU);
}

TEST_F(LogAverageLuminanceFixture, large_image) {
    using namespace Cogwheel::Math;

    auto device = create_performant_device1();
    OID3D11DeviceContext1 context;
    device->GetImmediateContext1(&context);

    LogAverageLuminance& log_average_exposure = LogAverageLuminance(*device, DX11_SHADER_ROOT);

    float min_log_luminance = -8;
    float max_log_luminance = 4;
    OID3D11Buffer constant_buffer = create_tonemapping_constants(device, min_log_luminance, max_log_luminance);

    // Image with one element in each bucket.
    const unsigned int width = LogAverageLuminance::max_groups_dispatched * LogAverageLuminance::group_width + 17;
    const unsigned int height = 13;
    const unsigned int pixel_count = width * height;
    half4* pixels = new half4[pixel_count];
    for (int i = 0; i < pixel_count; ++i) {
        half g = half(float(i));
        pixels[i] = { g, g, g, half(1.0f) };
    }
    std::random_shuffle(pixels, pixels + pixel_count);

    OID3D11ShaderResourceView pixel_SRV = create_texture_SRV(device, width, height, pixels);

    OID3D11UnorderedAccessView log_average_UAV;
    OID3D11Buffer log_average_buffer = create_default_buffer(device, DXGI_FORMAT_R32_FLOAT, 1, nullptr, &log_average_UAV);
    log_average_exposure.compute_log_average(*context, constant_buffer, pixel_SRV, width, log_average_UAV);

    float log_average_GPU;
    Readback::buffer(device, context, log_average_buffer, &log_average_GPU, &log_average_GPU + 1);

    double log_average = 0.0;
    for (int i = 0; i < pixel_count; ++i)
        log_average += log2(fmaxf(pixels[i].x, 0.0001f));
    log_average /= pixel_count;

    EXPECT_FLOAT_EQ((float)log_average, log_average_GPU);
}

} // NS DX11Renderer

#endif // _DX11RENDERER_LOG_AVERAGE_LUMINANCE_TEST_H_