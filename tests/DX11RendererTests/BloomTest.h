// DX11Renderer bloom test.
// ------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _DX11RENDERER_BLOOM_TEST_H_
#define _DX11RENDERER_BLOOM_TEST_H_

#include <gtest/gtest.h>
#include <Utils.h>

#include <Bifrost/Math/Color.h>
#include <Bifrost/Math/Half.h>

#include <DX11Renderer/CameraEffects.h>
#include <DX11Renderer/Compositor.h>
#include <DX11Renderer/Utils.h>

namespace DX11Renderer {

// ------------------------------------------------------------------------------------------------
// DX11Renderer bloom test.
// ------------------------------------------------------------------------------------------------
class Bloom : public ::testing::Test {
protected:
    // --------------------------------------------------------------------------------------------
    // Fixture setup
    // --------------------------------------------------------------------------------------------
    virtual void SetUp() {
        m_device = create_performant_device1();
        m_device->GetImmediateContext1(&m_context);
    }

    virtual void TearDown() {
        m_device = nullptr;
        m_context = nullptr;
    }

    // --------------------------------------------------------------------------------------------
    // Helpers
    // --------------------------------------------------------------------------------------------
    inline OBuffer create_and_bind_constants(ODevice1& device, ODeviceContext1& context, int width, int height, float bloom_threshold, int support = 11) {
        CameraEffects::Constants constants;
        constants.input_viewport = Bifrost::Math::Rect<float>(0, 0, float(width), float(height));
        constants.bloom_threshold = bloom_threshold;
        m_support = constants.bloom_support = support;
        float std_dev = constants.bloom_support * 0.25f;
        OBuffer constant_buffer;
        THROW_DX11_ERROR(create_constant_buffer(device, constants, &constant_buffer));

        context->CSSetConstantBuffers(0, 1, &constant_buffer);

        return constant_buffer;
    }

    // --------------------------------------------------------------------------------------------
    // Tests
    // --------------------------------------------------------------------------------------------
    using BloomFilter = std::function<OShaderResourceView&(ID3D11Buffer& constants, ID3D11ShaderResourceView* pixels, unsigned int image_width, unsigned int image_height)>;

    inline void test_energy_conservation(BloomFilter bloom_filter, Bifrost::Math::Vector3f error_pct) {
        using namespace Bifrost::Math;

        // Create image.
        const int width = 64, height = 64;
        const int pixel_count = width * height;
        half4 pixels[pixel_count];
        for (int y = 0; y < height; ++y)
            for (int x = 0; x < width; ++x) {
                int i = x + y * width;
                RGB color = RGB::white();
                pixels[i] = { half(color.r), half(color.g), half(color.b), half(1.0f) };
            }
        OShaderResourceView pixel_SRV;
        create_texture_2D(*m_device, DXGI_FORMAT_R16G16B16A16_FLOAT, pixels, width, height, &pixel_SRV);

        OBuffer constants = create_and_bind_constants(m_device, m_context, width, height, 0.0f);

        // Blur
        auto& filtered_SRV = bloom_filter(constants, pixel_SRV, width, height);

        ID3D11Resource* filtered_texture_2D;
        filtered_SRV->GetResource(&filtered_texture_2D);
        half4 filtered_pixels[pixel_count];
        Readback::texture2D(m_device, m_context, (ID3D11Texture2D*)filtered_texture_2D, filtered_pixels, filtered_pixels + pixel_count);
        filtered_texture_2D->Release();

        Vector3d summed_pixels = Vector3d::zero();
        Vector3d summed_filtered_pixels = Vector3d::zero();
        for (int p = 0; p < pixel_count; ++p) {
            summed_pixels += Vector3d(pixels[p].x, pixels[p].y, pixels[p].z);
            summed_filtered_pixels += Vector3d(filtered_pixels[p].x, filtered_pixels[p].y, filtered_pixels[p].z);
        }

        EXPECT_VECTOR3F_EQ_PCT((Vector3f)summed_pixels, (Vector3f)summed_filtered_pixels, error_pct);
    }

    inline void test_mirroring(BloomFilter bloom_filter) {
        using namespace Bifrost::Math;

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
        OShaderResourceView pixel_SRV;
        create_texture_2D(*m_device, DXGI_FORMAT_R16G16B16A16_FLOAT, pixels, width, height, &pixel_SRV);

        // Create mirrored image.
        half4 mirrored_pixels[pixel_count];
        for (int y = 0; y < height; ++y)
            for (int x = 0; x < width; ++x) {
                int i = x + y * width;
                int mirrored_i = (width - 1 - x) + (height - 1 - y) * width;
                mirrored_pixels[mirrored_i] = pixels[i];
            }
        OShaderResourceView mirrored_pixel_SRV;
        create_texture_2D(*m_device, DXGI_FORMAT_R16G16B16A16_FLOAT, mirrored_pixels, width, height, &mirrored_pixel_SRV);

        OBuffer constants = create_and_bind_constants(m_device, m_context, width, height, 0.0f);

        // Blur
        DualKawaseBloom bloom = DualKawaseBloom(*m_device);
        auto& filtered_SRV = bloom_filter(constants, pixel_SRV, width, height);
        auto& filtered_mirrored_SRV = bloom_filter(constants, mirrored_pixel_SRV, width, height);

        // Readback textures.
        ID3D11Resource* filtered_tex2D;
        filtered_SRV->GetResource(&filtered_tex2D);
        half4 filtered_pixels[pixel_count];
        Readback::texture2D(m_device, m_context, (ID3D11Texture2D*)filtered_tex2D, filtered_pixels, filtered_pixels + pixel_count);
        filtered_tex2D->Release();

        ID3D11Resource* filtered_mirrored_tex2D;
        filtered_SRV->GetResource(&filtered_mirrored_tex2D);
        half4 filtered_mirrored_pixels[pixel_count];
        Readback::texture2D(m_device, m_context, (ID3D11Texture2D*)filtered_mirrored_tex2D, filtered_mirrored_pixels, filtered_mirrored_pixels + pixel_count);
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

    inline void test_thresholding(BloomFilter bloom_filter, double error_pct) {
        using namespace Bifrost::Math;

        // Create image.
        const int width = 64, height = 64;
        const int pixel_count = width * height;
        half4 pixels[pixel_count];
        for (int y = 0; y < height; ++y)
            for (int x = 0; x < width; ++x) {
                int i = x + y * width;
                pixels[i] = { half(float(i)), half(float(x)), half(float(y * y)), half(1.0f) };
            }
        OShaderResourceView pixel_SRV;
        create_texture_2D(*m_device, DXGI_FORMAT_R16G16B16A16_FLOAT, pixels, width, height, &pixel_SRV);

        float bloom_threshold = 5.0f;
        OBuffer constants = create_and_bind_constants(m_device, m_context, width, height, bloom_threshold);

        auto& high_intensity_SRV = bloom_filter(constants, pixel_SRV, width, height);

        ID3D11Resource* high_intensity_texture_2D;
        high_intensity_SRV->GetResource(&high_intensity_texture_2D);
        half4 gpu_high_intensity_pixels[pixel_count];
        Readback::texture2D(m_device, m_context, (ID3D11Texture2D*)high_intensity_texture_2D, gpu_high_intensity_pixels, gpu_high_intensity_pixels + pixel_count);
        high_intensity_texture_2D->Release();

        Vector4d summed_cpu_high_intensity = Vector4d::zero();
        Vector4d summed_gpu_high_intensity = Vector4d::zero();
        for (int i = 0; i < width * height; ++i) {
            summed_cpu_high_intensity += { fmaxf(pixels[i][0] - bloom_threshold, 0.0f),
                fmaxf(pixels[i][1] - bloom_threshold, 0.0f),
                fmaxf(pixels[i][2] - bloom_threshold, 0.0f),
                pixels[i][3] };
            summed_gpu_high_intensity += { gpu_high_intensity_pixels[i][0], gpu_high_intensity_pixels[i][1], gpu_high_intensity_pixels[i][2], gpu_high_intensity_pixels[i][3] };
        }
        EXPECT_DOUBLE_EQ_PCT(summed_cpu_high_intensity[0], summed_gpu_high_intensity[0], error_pct);
        EXPECT_DOUBLE_EQ_PCT(summed_cpu_high_intensity[1], summed_gpu_high_intensity[1], error_pct);
        EXPECT_DOUBLE_EQ_PCT(summed_cpu_high_intensity[2], summed_gpu_high_intensity[2], error_pct);
        EXPECT_DOUBLE_EQ_PCT(summed_cpu_high_intensity[3], summed_gpu_high_intensity[3], error_pct);
    }

    // --------------------------------------------------------------------------------------------
    // Members
    // --------------------------------------------------------------------------------------------
    ODevice1 m_device;
    ODeviceContext1 m_context;

    int m_support;
};

// ------------------------------------------------------------------------------------------------
// Dual kawase tests
// ------------------------------------------------------------------------------------------------

TEST_F(Bloom, dual_kawase_energy_conservation) {
    DualKawaseBloom bloom = DualKawaseBloom(*m_device);
    auto bloom_filter = [&](ID3D11Buffer& constants, ID3D11ShaderResourceView* pixels, unsigned int image_width, unsigned int image_height) -> OShaderResourceView& {
        return bloom.filter(m_context, constants, pixels, image_width, image_height, 1);
    };

    test_energy_conservation(bloom_filter, Bifrost::Math::Vector3f(0.0001f));
}

TEST_F(Bloom, dual_kawase_mirroring) {
    DualKawaseBloom bloom = DualKawaseBloom(*m_device);
    auto bloom_filter = [&](ID3D11Buffer& constants, ID3D11ShaderResourceView* pixels, unsigned int image_width, unsigned int image_height) -> OShaderResourceView& {
        return bloom.filter(m_context, constants, pixels, image_width, image_height, 4);
    };

    test_mirroring(bloom_filter);
}

TEST_F(Bloom, dual_kawase_threshold) {
    DualKawaseBloom bloom = DualKawaseBloom(*m_device);
    auto bloom_filter = [&](ID3D11Buffer& constants, ID3D11ShaderResourceView* pixels, unsigned int image_width, unsigned int image_height) -> OShaderResourceView& {
        return bloom.filter(m_context, constants, pixels, image_width, image_height, 0);
    };

    test_thresholding(bloom_filter, 0.001);
}

// ------------------------------------------------------------------------------------------------
// Gaussian tests
// ------------------------------------------------------------------------------------------------

TEST_F(Bloom, gaussian_energy_conservation) {
    GaussianBloom bloom = GaussianBloom(*m_device);
    auto bloom_filter = [&](ID3D11Buffer& constants, ID3D11ShaderResourceView* pixels, unsigned int image_width, unsigned int image_height) -> OShaderResourceView& {
        return bloom.filter(m_context, constants, pixels, image_width, image_height, m_support);
    };

    test_energy_conservation(bloom_filter, Bifrost::Math::Vector3f(0.002f));
}

TEST_F(Bloom, guassian_mirroring) {
    GaussianBloom bloom = GaussianBloom(*m_device);
    auto bloom_filter = [&](ID3D11Buffer& constants, ID3D11ShaderResourceView* pixels, unsigned int image_width, unsigned int image_height) -> OShaderResourceView& {
        return bloom.filter(m_context, constants, pixels, image_width, image_height, m_support);
    };

    test_mirroring(bloom_filter);
}

TEST_F(Bloom, gaussian_threshold) {
    GaussianBloom bloom = GaussianBloom(*m_device);
    auto bloom_filter = [&](ID3D11Buffer& constants, ID3D11ShaderResourceView* pixels, unsigned int image_width, unsigned int image_height) -> OShaderResourceView& {
        return bloom.filter(m_context, constants, pixels, image_width, image_height, m_support);
    };

    test_thresholding(bloom_filter, 0.01);
}

} // NS DX11Renderer

#endif // _DX11RENDERER_BLOOM_TEST_H_