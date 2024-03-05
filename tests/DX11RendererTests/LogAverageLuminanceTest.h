// DX11Renderer linear exposure from geometric mean of log-average luminance test.
// ------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _DX11RENDERER_LOG_AVERAGE_LUMINANCE_TEST_H_
#define _DX11RENDERER_LOG_AVERAGE_LUMINANCE_TEST_H_

#include <gtest/gtest.h>
#include <Utils.h>

#include <Bifrost/Math/Color.h>
#include <Bifrost/Math/Half.h>
#include <Bifrost/Math/Vector.h>
#include <Bifrost/Math/Utils.h>

#include <DX11Renderer/CameraEffects.h>
#include <DX11Renderer/ShaderManager.h>
#include <DX11Renderer/Utils.h>

#include <random>

namespace DX11Renderer {

// ------------------------------------------------------------------------------------------------
// DX11Renderer linear exposure from geometric mean of log-average luminance test fixture.
// ------------------------------------------------------------------------------------------------
class LogAverageLuminanceFixture : public ::testing::Test {
protected:

    // Compute linear exposure from the geometric mean. See MJP's tonemapping sample.
    // https://mynameismjp.wordpress.com/2010/04/30/a-closer-look-at-tone-mapping/
    inline float geometric_mean_linear_exposure(float log_average_luminance) {
        float key_value = 1.03f - (2.0f / (2 + log10(log_average_luminance + 1)));
        return key_value / log_average_luminance;
    }

    inline void test_image(int width, int height, half4* pixels) {
        auto device = create_test_device();
        auto context = get_immidiate_context1(device);

        auto& log_average_exposure = LogAverageLuminance(*device, ShaderManager());

        float min_log_luminance = -24;
        float max_log_luminance = 24;
        OBuffer constant_buffer = create_camera_effects_constants(device, { width, height }, min_log_luminance, max_log_luminance);
        context->CSSetConstantBuffers(0, 1, &constant_buffer);

        OShaderResourceView pixel_SRV;
        create_texture_2D(*device, DXGI_FORMAT_R16G16B16A16_FLOAT, pixels, width, height, &pixel_SRV);

        OUnorderedAccessView output_UAV;
        OBuffer output_buffer = create_default_buffer(device, DXGI_FORMAT_R32_FLOAT, 1, nullptr, &output_UAV);

        float log_average_GPU = 0.0f;
        { // Test log-average computation 
            log_average_exposure.compute_log_average(*context, constant_buffer, pixel_SRV, width, output_UAV);

            Readback::buffer(device, context, output_buffer, &log_average_GPU, &log_average_GPU + 1);

            double log_average = 0.0;
            for (int i = 0; i < width * height; ++i) {
                auto pixel = Bifrost::Math::RGB(pixels[i].x, pixels[i].y, pixels[i].z);
                log_average += log2(fmaxf(Bifrost::Math::luminance(pixel), 0.0001f));
            }
            log_average = exp2(log_average / (width * height));

            EXPECT_FLOAT_EQ_PCT((float)log_average, log_average_GPU, 0.00001f);
        }

        { // Test linear exposure computation from log-average.
            float linear_exposure = geometric_mean_linear_exposure(log_average_GPU);

            // Upload CPU computed log average to avoid issues with eye adaptation and numerical precision
            context->UpdateSubresource(output_buffer, 0, nullptr, &linear_exposure, 0u, 0u);

            log_average_exposure.compute_linear_exposure(*context, constant_buffer, pixel_SRV, width, output_UAV);

            float linear_exposure_GPU;
            Readback::buffer(device, context, output_buffer, &linear_exposure_GPU, &linear_exposure_GPU + 1);

            EXPECT_FLOAT_EQ_PCT(linear_exposure, linear_exposure_GPU, 0.00001f);
        }
    }
};

TEST_F(LogAverageLuminanceFixture, tiny_image) {
    using namespace Bifrost::Math;

    // Image with one element in each bucket.
    const unsigned int pixel_count = 64;
    half4 pixels[pixel_count];
    for (int i = 0; i < pixel_count; ++i) {
        half g = half(float(i));
        pixels[i] = { g, g, g, half(1.0f) };
    }

    test_image(pixel_count, 1, pixels);
}

TEST_F(LogAverageLuminanceFixture, large_image) {
    using namespace Bifrost::Math;

    const unsigned int width = LogAverageLuminance::max_groups_dispatched * LogAverageLuminance::group_width + 17;
    const unsigned int height = 21;
    const unsigned int pixel_count = width * height;
    half4* pixels = new half4[pixel_count];
    for (int i = 0; i < pixel_count; ++i) {
        half g = half(float(i));
        pixels[i] = { g, g, g, half(1.0f) };
    }
    std::shuffle(pixels, pixels + pixel_count, std::minstd_rand(1234567799));

    test_image(width, height, pixels);
}

TEST_F(LogAverageLuminanceFixture, black_image) {
    using namespace Bifrost::Math;

    const unsigned int width = LogAverageLuminance::max_groups_dispatched * LogAverageLuminance::group_width + 17;
    const unsigned int height = 21;
    const unsigned int pixel_count = width * height;
    const half zero = half(0.0f);
    half4* pixels = new half4[pixel_count];
    for (int i = 0; i < pixel_count; ++i)
        pixels[i] = { zero, zero, zero, half(1.0f) };
    std::shuffle(pixels, pixels + pixel_count, std::minstd_rand(1234567799));

    test_image(width, height, pixels);
}

} // NS DX11Renderer

#endif // _DX11RENDERER_LOG_AVERAGE_LUMINANCE_TEST_H_