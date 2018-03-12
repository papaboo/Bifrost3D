// DX11Renderer exposure histogram test.
// ------------------------------------------------------------------------------------------------
// Copyright (C) 2018, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _DX11RENDERER_EXPOSURE_HISTOGRAM_TEST_H_
#define _DX11RENDERER_EXPOSURE_HISTOGRAM_TEST_H_

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
// DX11Renderer exposure histogram test.
// ------------------------------------------------------------------------------------------------
class ExposureHistogramFixture : public ::testing::Test {
protected:

    using half4 = Cogwheel::Math::Vector4<half>;

    struct ExposureHistogramConstants {
        float min_log_luminance;
        float max_log_luminance;
        float2 __padding2;
    };

    inline OID3D11ShaderResourceView create_texture(OID3D11Device1& device, unsigned int width, unsigned int height, half4* pixels) {
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

TEST_F(ExposureHistogramFixture, tiny_image) {
    using namespace Cogwheel::Math;

    auto device = create_performant_device1();
    ID3D11DeviceContext1* context;
    device->GetImmediateContext1(&context);

    ExposureHistogram& histogram = ExposureHistogram(*device, DX11_SHADER_ROOT);
    const unsigned int bin_count = ExposureHistogram::bin_count;

    OID3D11Buffer constant_buffer;
    float min_log_luminance = -8;
    float max_log_luminance = 4;
    ExposureHistogramConstants constants = { min_log_luminance, max_log_luminance, 0, 0 };
    THROW_ON_FAILURE(create_constant_buffer(*device, constants, &constant_buffer));

    // Image with one element in each bucket.
    half4 pixels[bin_count];
    for (int i = 0; i < bin_count; ++i) {
        half g = half(exp2(lerp(min_log_luminance, max_log_luminance, (i + 0.5f) / bin_count)));
        pixels[i] = { g, g, g, half(1.0f) };
    }
    OID3D11ShaderResourceView pixel_SRV = create_texture(device, bin_count, 1, pixels);

    ID3D11ShaderResourceView* histogram_SRV = histogram.reduce_histogram(*context, constant_buffer, pixel_SRV, bin_count);

    ID3D11Resource* histogram_resource;
    histogram_SRV->GetResource(&histogram_resource);
    std::vector<unsigned int> cpu_histogram; cpu_histogram.resize(bin_count);
    readback_buffer(device, context, (ID3D11Buffer*)histogram_resource, cpu_histogram.begin(), cpu_histogram.end());

    for (int bin = 0; bin < cpu_histogram.size(); ++bin)
        EXPECT_EQ(cpu_histogram[bin], 1);
}

TEST_F(ExposureHistogramFixture, small_image) {
    using namespace Cogwheel::Math;

    auto device = create_performant_device1();
    ID3D11DeviceContext1* context;
    device->GetImmediateContext1(&context);

    ExposureHistogram& histogram = ExposureHistogram(*device, DX11_SHADER_ROOT);
    const unsigned int bin_count = ExposureHistogram::bin_count;

    OID3D11Buffer constant_buffer;
    float min_log_luminance = -8;
    float max_log_luminance = 4;
    ExposureHistogramConstants constants = { min_log_luminance, max_log_luminance, 0, 0 };
    THROW_ON_FAILURE(create_constant_buffer(*device, constants, &constant_buffer));

    // Image with 4 elements in each bucket and (4 + width) elements in the first and last bucket
    const int width = bin_count;
    const int height = 6;
    const int element_count = width * height;
    half4 pixels[element_count];
    for (int x = 0; x < width; ++x) {
        half g1 = half(exp2(lerp(min_log_luminance, max_log_luminance, (x + 0.5f) / bin_count)));
        half g2 = half(exp2(lerp(min_log_luminance, max_log_luminance, 1.0f - (x + 0.5f) / bin_count)));
        float min_luminance = exp2(min_log_luminance) * 0.5f;
        pixels[x + 0 * width] = half4(Vector4f(min_luminance, min_luminance, min_luminance, 1.0f));
        pixels[x + 1 * width] = half4(g1, g1, g1, half(1.0f));
        pixels[x + 2 * width] = half4(g2, g2, g2, half(1.0f));
        pixels[x + 3 * width] = half4(g1, g1, g1, half(1.0f));
        pixels[x + 4 * width] = half4(g2, g2, g2, half(1.0f));
        float max_luminance = exp2(max_log_luminance) * 2.0f;
        pixels[x + 5 * width] = half4(Vector4f(max_luminance, max_luminance, max_luminance, 1.0f));
    }
    OID3D11ShaderResourceView pixel_SRV = create_texture(device, width, height, pixels);

    ID3D11ShaderResourceView* histogram_SRV = histogram.reduce_histogram(*context, constant_buffer, pixel_SRV, width);

    ID3D11Resource* histogram_resource;
    histogram_SRV->GetResource(&histogram_resource);
    std::vector<unsigned int> cpu_histogram; cpu_histogram.resize(bin_count);
    readback_buffer(device, context, (ID3D11Buffer*)histogram_resource, cpu_histogram.begin(), cpu_histogram.end());

    EXPECT_EQ(cpu_histogram[0], 4 + width);
    for (int bin = 1; bin < cpu_histogram.size() - 1; ++bin)
        EXPECT_EQ(cpu_histogram[bin], 4);
    EXPECT_EQ(cpu_histogram[cpu_histogram.size()-1], 4 + width);
}

} // NS DX11Renderer

#endif // _DX11RENDERER_EXPOSURE_HISTOGRAM_TEST_H_