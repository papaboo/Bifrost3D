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

#include <DX11Renderer/CameraEffects.h>
#include <DX11Renderer/Compositor.h>
#include <DX11Renderer/Utils.h>

namespace DX11Renderer {

// ------------------------------------------------------------------------------------------------
// DX11Renderer exposure histogram test.
// ------------------------------------------------------------------------------------------------
class ExposureHistogramFixture : public ::testing::Test {
protected:

    // Unreal 4 PostProcessHistogramCommon.ush::ComputeAverageLuminaneWithoutOutlier
    inline float compute_average_luminance_without_outlier(unsigned int* histogram_begin, unsigned int* histogram_end, 
        float min_percentage, float max_percentage, float min_log_luminance, float max_log_luminance) {

        auto compute_luminance_from_histogram_position = [=](float normalized_index) -> float {
            return exp2(Cogwheel::Math::lerp(min_log_luminance, max_log_luminance, normalized_index));
        };

        int histogram_size = int(histogram_end - histogram_begin);

        int pixel_count = 0;
        for (int i = 0; i < histogram_size; ++i)
            pixel_count += histogram_begin[i];

        float min_pixel_count = pixel_count * min_percentage;
        float max_pixel_count = pixel_count * max_percentage;

        Cogwheel::Math::Vector2f sum = Cogwheel::Math::Vector2f::zero();
        for (int i = 0; i < histogram_size; ++i) {
            float bucket_count = float(histogram_begin[i]);

            // remove outlier at lower end
            float sub = fminf(bucket_count, min_pixel_count);
            bucket_count -= sub;
            min_pixel_count -= sub;
            max_pixel_count -= sub;

            // remove outlier at upper end
            bucket_count = fminf(bucket_count, max_pixel_count);
            max_pixel_count -= bucket_count;

            float luminance_at_bucket = compute_luminance_from_histogram_position((i + 0.5f) / float(histogram_size));

            sum += Cogwheel::Math::Vector2f(luminance_at_bucket * bucket_count, bucket_count);
        }

        return sum.x / fmaxf(0.0001f, sum.y);
    }
};

TEST_F(ExposureHistogramFixture, tiny_image) {
    using namespace Cogwheel::Math;

    auto device = create_performant_device1();
    ODeviceContext1 context;
    device->GetImmediateContext1(&context);

    ExposureHistogram& histogram = ExposureHistogram(*device, DX11_SHADER_ROOT);
    const unsigned int bin_count = ExposureHistogram::bin_count;

    float min_log_luminance = -8;
    float max_log_luminance = 4;
    OBuffer constant_buffer = create_camera_effects_constants(device, min_log_luminance, max_log_luminance);
    context->CSSetConstantBuffers(0, 1, &constant_buffer);

    // Image with one element in each bucket.
    half4 pixels[bin_count];
    for (int i = 0; i < bin_count; ++i) {
        half g = half(exp2(lerp(min_log_luminance, max_log_luminance, (i + 0.5f) / bin_count)));
        pixels[i] = { g, g, g, half(1.0f) };
    }
    OShaderResourceView pixel_SRV;
    create_texture_2D(*device, DXGI_FORMAT_R16G16B16A16_FLOAT, pixels, bin_count, 1, &pixel_SRV);

    OShaderResourceView& histogram_SRV = histogram.reduce_histogram(*context, constant_buffer, pixel_SRV, bin_count);

    OResource histogram_resource;
    histogram_SRV->GetResource(&histogram_resource);
    std::vector<unsigned int> cpu_histogram; cpu_histogram.resize(bin_count);
    Readback::buffer(device, context, (ID3D11Buffer*)histogram_resource.get(), cpu_histogram.begin(), cpu_histogram.end());

    for (int bin = 0; bin < cpu_histogram.size(); ++bin)
        EXPECT_EQ(1, cpu_histogram[bin]);
}

TEST_F(ExposureHistogramFixture, small_image) {
    using namespace Cogwheel::Math;

    auto device = create_performant_device1();
    ODeviceContext1 context;
    device->GetImmediateContext1(&context);

    ExposureHistogram& histogram = ExposureHistogram(*device, DX11_SHADER_ROOT);
    const unsigned int bin_count = ExposureHistogram::bin_count;

    float min_log_luminance = -8;
    float max_log_luminance = 4;
    OBuffer constant_buffer = create_camera_effects_constants(device, min_log_luminance, max_log_luminance);
    context->CSSetConstantBuffers(0, 1, &constant_buffer);

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
    OShaderResourceView pixel_SRV;
    create_texture_2D(*device, DXGI_FORMAT_R16G16B16A16_FLOAT, pixels, width, height, &pixel_SRV);

    OShaderResourceView& histogram_SRV = histogram.reduce_histogram(*context, constant_buffer, pixel_SRV, width);

    OResource histogram_resource;
    histogram_SRV->GetResource(&histogram_resource);
    std::vector<unsigned int> cpu_histogram; cpu_histogram.resize(bin_count);
    Readback::buffer(device, context, (ID3D11Buffer*)histogram_resource.get(), cpu_histogram.begin(), cpu_histogram.end());

    EXPECT_EQ(4 + width, cpu_histogram[0]);
    for (int bin = 1; bin < cpu_histogram.size() - 1; ++bin)
        EXPECT_EQ(4, cpu_histogram[bin]);
    EXPECT_EQ(4 + width, cpu_histogram[cpu_histogram.size()-1]);
}

TEST_F(ExposureHistogramFixture, exposure_from_constant_histogram) {
    using namespace Cogwheel::Math;

    auto device = create_performant_device1();
    ODeviceContext1 context;
    device->GetImmediateContext1(&context);

    const unsigned int bin_count = ExposureHistogram::bin_count;

    unsigned int histogram[bin_count];
    for (int i = 0; i < bin_count; ++i)
        histogram[i] = 1;

    OShaderResourceView histogram_SRV;
    create_default_buffer(device, DXGI_FORMAT_R32_UINT, histogram, bin_count, &histogram_SRV, nullptr);

    OUnorderedAccessView linear_exposure_UAV;
    OBuffer linear_exposure_buffer = create_default_buffer(device, DXGI_FORMAT_R32_FLOAT, 1, nullptr, &linear_exposure_UAV);

    OBlob compute_exposure_blob = compile_shader(DX11_SHADER_ROOT + std::wstring(L"CameraEffects\\ReduceExposureHistogram.hlsl"), "cs_5_0", "CameraEffects::compute_linear_exposure");
    OComputeShader compute_exposure_shader;
    THROW_ON_FAILURE(device->CreateComputeShader(UNPACK_BLOB_ARGS(compute_exposure_blob), nullptr, &compute_exposure_shader));

    float min_log_luminance = -8;
    float max_log_luminance = 4;
    float min_percentage = 0.8f;
    float max_percentage = 0.95f;
    OBuffer constant_buffer = create_camera_effects_constants(device, min_log_luminance, max_log_luminance, min_percentage, max_percentage);

    context->CSSetShader(compute_exposure_shader, nullptr, 0u);
    context->CSSetConstantBuffers(0, 1, &constant_buffer);
    context->CSSetShaderResources(0, 1, &histogram_SRV);
    context->CSSetUnorderedAccessViews(0, 1, &linear_exposure_UAV, 0u);
    context->Dispatch(1, 1, 1);

    float gpu_linear_exposure;
    Readback::buffer(device, context, linear_exposure_buffer, &gpu_linear_exposure, &gpu_linear_exposure + 1);

    float average_luminance = compute_average_luminance_without_outlier(histogram, histogram + bin_count, 
        min_percentage, max_percentage, min_log_luminance, max_log_luminance);
    float reference_linear_exposure = 1.0f / average_luminance;

    EXPECT_FLOAT_EQ(reference_linear_exposure, gpu_linear_exposure);
}

TEST_F(ExposureHistogramFixture, exposure_from_histogram) {
    using namespace Cogwheel::Math;

    auto device = create_performant_device1();
    ODeviceContext1 context;
    device->GetImmediateContext1(&context);

    const unsigned int bin_count = ExposureHistogram::bin_count;

    unsigned int histogram[bin_count];
    for (int i = 0; i < bin_count; ++i)
        histogram[i] = i;
    std::random_shuffle(histogram, histogram + bin_count);

    OShaderResourceView histogram_SRV;
    create_default_buffer(device, DXGI_FORMAT_R32_UINT, histogram, bin_count, &histogram_SRV, nullptr);

    OUnorderedAccessView linear_exposure_UAV;
    OBuffer linear_exposure_buffer = create_default_buffer(device, DXGI_FORMAT_R32_FLOAT, 1, nullptr, &linear_exposure_UAV);

    OBlob compute_exposure_blob = compile_shader(DX11_SHADER_ROOT + std::wstring(L"CameraEffects\\ReduceExposureHistogram.hlsl"), "cs_5_0", "CameraEffects::compute_linear_exposure");
    OComputeShader compute_exposure_shader;
    THROW_ON_FAILURE(device->CreateComputeShader(UNPACK_BLOB_ARGS(compute_exposure_blob), nullptr, &compute_exposure_shader));

    float min_log_luminance = -8;
    float max_log_luminance = 4;
    float min_percentage = 0.8f;
    float max_percentage = 0.95f;
    OBuffer constant_buffer = create_camera_effects_constants(device, min_log_luminance, max_log_luminance, min_percentage, max_percentage);

    context->CSSetShader(compute_exposure_shader, nullptr, 0u);
    context->CSSetConstantBuffers(0, 1, &constant_buffer);
    context->CSSetShaderResources(0, 1, &histogram_SRV);
    context->CSSetUnorderedAccessViews(0, 1, &linear_exposure_UAV, 0u);
    context->Dispatch(1, 1, 1);

    float gpu_linear_exposure;
    Readback::buffer(device, context, linear_exposure_buffer, &gpu_linear_exposure, &gpu_linear_exposure + 1);

    float average_luminance = compute_average_luminance_without_outlier(histogram, histogram + bin_count,
        min_percentage, max_percentage, min_log_luminance, max_log_luminance);
    float reference_linear_exposure = 1.0f / average_luminance;

    EXPECT_FLOAT_EQ(reference_linear_exposure, gpu_linear_exposure);
}

} // NS DX11Renderer

#endif // _DX11RENDERER_EXPOSURE_HISTOGRAM_TEST_H_