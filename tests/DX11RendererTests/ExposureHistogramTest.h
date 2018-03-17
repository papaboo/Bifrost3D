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

    inline OID3D11Buffer create_constant_buffer(OID3D11Device1& device, float min_log_luminance, float max_log_luminance, float min_percentage = 0.8f, float max_percentage = 0.95f) {
        OID3D11Buffer constant_buffer;
        Tonemapper::Constants constants = { min_log_luminance, max_log_luminance, min_percentage, max_percentage };
        THROW_ON_FAILURE(DX11Renderer::create_constant_buffer(device, constants, &constant_buffer));
        return constant_buffer;
    }

    inline float compute_linear_exposure(float normalized_index, float min_log_luminance, float max_log_luminance) {
        float recip_exposure = exp2(Cogwheel::Math::lerp(min_log_luminance, max_log_luminance, normalized_index));
        float scene_key = 1.0f;
        return scene_key / recip_exposure;
    }

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

TEST_F(ExposureHistogramFixture, tiny_image) {
    using namespace Cogwheel::Math;

    auto device = create_performant_device1();
    OID3D11DeviceContext1 context;
    device->GetImmediateContext1(&context);

    ExposureHistogram& histogram = ExposureHistogram(*device, DX11_SHADER_ROOT);
    const unsigned int bin_count = ExposureHistogram::bin_count;

    float min_log_luminance = -8;
    float max_log_luminance = 4;
    OID3D11Buffer constant_buffer = create_constant_buffer(device, min_log_luminance, max_log_luminance);

    // Image with one element in each bucket.
    half4 pixels[bin_count];
    for (int i = 0; i < bin_count; ++i) {
        half g = half(exp2(lerp(min_log_luminance, max_log_luminance, (i + 0.5f) / bin_count)));
        pixels[i] = { g, g, g, half(1.0f) };
    }
    OID3D11ShaderResourceView pixel_SRV = create_texture_SRV(device, bin_count, 1, pixels);

    OID3D11ShaderResourceView& histogram_SRV = histogram.reduce_histogram(*context, constant_buffer, pixel_SRV, bin_count);

    OID3D11Resource histogram_resource;
    histogram_SRV->GetResource(&histogram_resource);
    std::vector<unsigned int> cpu_histogram; cpu_histogram.resize(bin_count);
    Readback::buffer(device, context, (ID3D11Buffer*)histogram_resource.get(), cpu_histogram.begin(), cpu_histogram.end());

    for (int bin = 0; bin < cpu_histogram.size(); ++bin)
        EXPECT_EQ(cpu_histogram[bin], 1);
}

TEST_F(ExposureHistogramFixture, small_image) {
    using namespace Cogwheel::Math;

    auto device = create_performant_device1();
    OID3D11DeviceContext1 context;
    device->GetImmediateContext1(&context);

    ExposureHistogram& histogram = ExposureHistogram(*device, DX11_SHADER_ROOT);
    const unsigned int bin_count = ExposureHistogram::bin_count;

    float min_log_luminance = -8;
    float max_log_luminance = 4;
    OID3D11Buffer constant_buffer = create_constant_buffer(device, min_log_luminance, max_log_luminance);

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
    OID3D11ShaderResourceView pixel_SRV = create_texture_SRV(device, width, height, pixels);

    OID3D11ShaderResourceView& histogram_SRV = histogram.reduce_histogram(*context, constant_buffer, pixel_SRV, width);

    OID3D11Resource histogram_resource;
    histogram_SRV->GetResource(&histogram_resource);
    std::vector<unsigned int> cpu_histogram; cpu_histogram.resize(bin_count);
    Readback::buffer(device, context, (ID3D11Buffer*)histogram_resource.get(), cpu_histogram.begin(), cpu_histogram.end());

    EXPECT_EQ(cpu_histogram[0], 4 + width);
    for (int bin = 1; bin < cpu_histogram.size() - 1; ++bin)
        EXPECT_EQ(cpu_histogram[bin], 4);
    EXPECT_EQ(cpu_histogram[cpu_histogram.size()-1], 4 + width);
}

TEST_F(ExposureHistogramFixture, exposure_from_constant_histogram) {
    using namespace Cogwheel::Math;

    auto device = create_performant_device1();
    OID3D11DeviceContext1 context;
    device->GetImmediateContext1(&context);

    const unsigned int bin_count = ExposureHistogram::bin_count;

    unsigned int histogram[bin_count];
    for (int i = 0; i < bin_count; ++i)
        histogram[i] = 1;

    OID3D11UnorderedAccessView histogram_UAV;
    create_default_buffer(device, DXGI_FORMAT_R32_UINT, bin_count, histogram, nullptr, &histogram_UAV);

    OID3D11UnorderedAccessView linear_exposure_UAV;
    OID3D11Buffer linear_exposure_buffer = create_default_buffer(device, DXGI_FORMAT_R32_FLOAT, 1, nullptr, &linear_exposure_UAV);

    OID3DBlob compute_exposure_blob = compile_shader(DX11_SHADER_ROOT + std::wstring(L"Compute\\ReduceExposureHistogram.hlsl"), "cs_5_0", "compute_exposure");
    OID3D11ComputeShader compute_exposure_shader;
    THROW_ON_FAILURE(device->CreateComputeShader(UNPACK_BLOB_ARGS(compute_exposure_blob), nullptr, &compute_exposure_shader));

    float min_log_luminance = -8;
    float max_log_luminance = 4;
    float min_percentage = 0.8f;
    float max_percentage = 0.95f;
    OID3D11Buffer constant_buffer = create_constant_buffer(device, min_log_luminance, max_log_luminance, min_percentage, max_percentage);

    context->CSSetShader(compute_exposure_shader, nullptr, 0u);
    context->CSSetConstantBuffers(0, 1, &constant_buffer);
    ID3D11UnorderedAccessView* UAVs[2] = { histogram_UAV.get(), linear_exposure_UAV.get() };
    context->CSSetUnorderedAccessViews(0, 2, UAVs, 0u);
    context->Dispatch(1, 1, 1);

    std::vector<float> cpu_linear_exposure; cpu_linear_exposure.resize(1);
    Readback::buffer(device, context, linear_exposure_buffer, cpu_linear_exposure.begin(), cpu_linear_exposure.end());
    float linear_exposure = cpu_linear_exposure[0];

    float normalized_index = (min_percentage + max_percentage) * 0.5f;
    float reference_linear_exposure = compute_linear_exposure(normalized_index, min_log_luminance, max_log_luminance);

    EXPECT_FLOAT_EQ(linear_exposure, reference_linear_exposure);
}

TEST_F(ExposureHistogramFixture, exposure_from_constant_image) {
    using namespace Cogwheel::Math;

    auto device = create_performant_device1();
    OID3D11DeviceContext1 context;
    device->GetImmediateContext1(&context);

    ExposureHistogram& histogram = ExposureHistogram(*device, DX11_SHADER_ROOT);

    float min_log_luminance = -8;
    float max_log_luminance = 4;
    OID3D11Buffer constant_buffer = create_constant_buffer(device, min_log_luminance, max_log_luminance);

    // Constant luminance image
    const int width = 640;
    const int height = 480;
    const int pixel_count = width * height;
    half4* pixels = new half4[pixel_count];
    for (int i = 0; i < pixel_count; ++i) {
        half g = half(0.5f);
        pixels[i] = { g, g, g, half(1.0f) };
    }
    OID3D11ShaderResourceView pixel_SRV = create_texture_SRV(device, width, height, pixels);

    OID3D11ShaderResourceView& linear_exposure_SRV = histogram.compute_linear_exposure(*context, constant_buffer, pixel_SRV, width);

    OID3D11Resource linear_exposure_resource;
    linear_exposure_SRV->GetResource(&linear_exposure_resource);
    float cpu_linear_exposure;
    Readback::buffer(device, context, (ID3D11Buffer*)linear_exposure_resource.get(), &cpu_linear_exposure, &cpu_linear_exposure + 1);

    printf("cpu_linear_exposure: %f\n", cpu_linear_exposure);

    delete pixels;
}

} // NS DX11Renderer

#endif // _DX11RENDERER_EXPOSURE_HISTOGRAM_TEST_H_