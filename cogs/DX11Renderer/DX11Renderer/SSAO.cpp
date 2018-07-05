// DirectX 11 renderer screen space ambient occlusion implementations.
// ------------------------------------------------------------------------------------------------
// Copyright (C) 2018, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#include <DX11Renderer/SSAO.h>
#include <DX11Renderer/Utils.h>

#include <Cogwheel/Math/RNG.h>

namespace DX11Renderer {
namespace SSAO {

// ------------------------------------------------------------------------------------------------
// Bilateral blur for SSAO.
// ------------------------------------------------------------------------------------------------
BilateralBlur::BilateralBlur(ID3D11Device1& device, const std::wstring& shader_folder_path)
    : m_width(0), m_height(0), m_intermediate_RTV(nullptr), m_intermediate_SRV(nullptr) {

    OBlob vertex_shader_blob = compile_shader(shader_folder_path + L"SSAO.hlsl", "vs_5_0", "main_vs");
    THROW_ON_FAILURE(device.CreateVertexShader(UNPACK_BLOB_ARGS(vertex_shader_blob), nullptr, &m_vertex_shader));

    OBlob filter_blob = compile_shader(shader_folder_path + L"SSAO.hlsl", "ps_5_0", "BilateralBoxBlur::filter_ps");
    THROW_ON_FAILURE(device.CreatePixelShader(UNPACK_BLOB_ARGS(filter_blob), nullptr, &m_filter_shader));

    for (int i = 0; i < max_passes; ++i) {
        Constants constants = { i * 2 + 1.0f };
        create_constant_buffer(device, constants, &m_constants[i]);
    }
}

OShaderResourceView& BilateralBlur::apply(ID3D11DeviceContext1& context, ORenderTargetView& ao_RTV, OShaderResourceView& ao_SRV, int width, int height) {
    if (m_width != width || m_height != height) {
        m_intermediate_SRV.release();
        m_intermediate_RTV.release();

        // Resize backbuffer
        ODevice1 device = get_device1(context);
        create_texture_2D(device, DXGI_FORMAT_R16G16B16A16_FLOAT, width, height, &m_intermediate_SRV, nullptr, &m_intermediate_RTV);

        m_width = width;
        m_height = height;
    }

    // Need to grab the normal and depth buffers before OMSetRenderTarget clears them.
    ID3D11ShaderResourceView* srvs[2];
    context.PSGetShaderResources(0, 2, srvs);

    context.VSSetShader(m_vertex_shader, 0, 0);
    context.PSSetShader(m_filter_shader, 0, 0);
    int i;
    for (i = 0; i < max_passes; ++i) {
        auto& rtv = (i % 2) == 0 ? m_intermediate_RTV : ao_RTV;
        context.OMSetRenderTargets(1, &rtv, nullptr);
        context.PSSetShaderResources(0, 2, srvs);
        auto& srv = (i % 2) == 0 ? ao_SRV : m_intermediate_SRV;
        context.PSSetShaderResources(2, 1, &srv);
        context.PSSetConstantBuffers(2, 1, &m_constants[max_passes - i - 1]);
        context.Draw(3, 0);
    }

    srvs[0]->Release();
    srvs[1]->Release();

    return (i % 2) == 0 ? ao_SRV : m_intermediate_SRV;
}

// ------------------------------------------------------------------------------------------------
// The Alchemy screen-space ambient obscurance algorithm.
// http://casual-effects.com/research/McGuire2011AlchemyAO/index.html
// ------------------------------------------------------------------------------------------------
AlchemyAO::AlchemyAO(ID3D11Device1& device, const std::wstring& shader_folder_path) 
    : m_width(0), m_height(0), m_SSAO_RTV(nullptr), m_SSAO_SRV(nullptr) {

    using namespace Cogwheel::Math;

    create_constant_buffer(device, sizeof(SsaoSettings) + sizeof(float4), &m_constants);

    { // Allocate samples.
        static auto cosine_disk_sampling = [](Vector2f sample_uv) -> Vector2f {
            float r = sample_uv.x;
            float theta = 2.0f * PI<float>() * sample_uv.y;
            return r * Vector2f(cos(theta), sin(theta));
        };

        auto* samples = new Vector2f[max_sample_count];
        for (int i = 0; i < max_sample_count; ++i)
            samples[i] = cosine_disk_sampling(RNG::sample02(i+1)); // Drop the first sample as it is (0, 0)

        D3D11_BUFFER_DESC desc = {};
        desc.Usage = D3D11_USAGE_IMMUTABLE;
        desc.ByteWidth = sizeof(float2) * max_sample_count;
        desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        desc.CPUAccessFlags = 0;
        desc.MiscFlags = 0;

        D3D11_SUBRESOURCE_DATA resource_data = {};
        resource_data.pSysMem = samples;
        device.CreateBuffer(&desc, &resource_data, &m_samples);

        delete[] samples;
    }

    OBlob vertex_shader_blob = compile_shader(shader_folder_path + L"SSAO.hlsl", "vs_5_0", "main_vs");
    THROW_ON_FAILURE(device.CreateVertexShader(UNPACK_BLOB_ARGS(vertex_shader_blob), nullptr, &m_vertex_shader));

    OBlob pixel_shader_blob = compile_shader(shader_folder_path + L"SSAO.hlsl", "ps_5_0", "alchemy_ps");
    THROW_ON_FAILURE(device.CreatePixelShader(UNPACK_BLOB_ARGS(pixel_shader_blob), nullptr, &m_pixel_shader));

    m_filter = BilateralBlur(device, shader_folder_path);
}

OShaderResourceView& AlchemyAO::apply(ID3D11DeviceContext1& context, OShaderResourceView& normals, OShaderResourceView& depth, int width, int height, SsaoSettings settings) {
    if (m_width != width || m_height != height) {
        m_SSAO_SRV.release();
        m_SSAO_RTV.release();

        // Resize backbuffer
        ODevice1 device = get_device1(context);
        create_texture_2D(device, DXGI_FORMAT_R16G16B16A16_FLOAT, width, height, &m_SSAO_SRV, nullptr, &m_SSAO_RTV);

        m_width = width;
        m_height = height;
    }

    struct SsaoConstants {
        SsaoSettings settings;
        int2 texture_height;
        float2 recip_texture_height;
    };
    SsaoConstants constants = { settings, m_width, m_height, 1.0f / m_width, 1.0f / m_height };
    constants.settings.normal_std_dev = 0.5f / (constants.settings.normal_std_dev * constants.settings.normal_std_dev);
    constants.settings.plane_std_dev = 0.5f / (constants.settings.plane_std_dev * constants.settings.plane_std_dev);
    context.UpdateSubresource(m_constants, 0u, nullptr, &constants, 0u, 0u);

    ID3D11Buffer* constant_buffers[] = { m_constants, m_samples };
    context.PSSetConstantBuffers(1, 2, constant_buffers);

    // Setup state.
    context.OMSetRenderTargets(1, &m_SSAO_RTV, nullptr);
    ID3D11ShaderResourceView* SRVs[2] = { normals, depth };
    context.PSSetShaderResources(0, 2, SRVs);

    // Compute SSAO.
    context.VSSetShader(m_vertex_shader, 0, 0);
    context.PSSetShader(m_pixel_shader, 0, 0);
    context.Draw(3, 0);

    // Filter
    OShaderResourceView& ao_SRV = settings.filtering_enabled ?
        m_filter.apply(context, m_SSAO_RTV, m_SSAO_SRV, width, height) :
        m_SSAO_SRV;

    // Unbind SSAO_RTV
    ID3D11RenderTargetView* null_RTV = nullptr;
    context.OMSetRenderTargets(1, &null_RTV, nullptr);

    return ao_SRV;
}

OShaderResourceView& AlchemyAO::apply_none(ID3D11DeviceContext1& context, int width, int height) {
    if (m_width != width || m_height != height) {
        m_SSAO_SRV.release();
        m_SSAO_RTV.release();

        // Resize backbuffer
        ODevice1 device = get_device1(context);
        create_texture_2D(device, DXGI_FORMAT_R16G16B16A16_FLOAT, width, height, &m_SSAO_SRV, nullptr, &m_SSAO_RTV);

        m_width = width;
        m_height = height;
    }

    float cleared_ssao[4] = { 1, 0, 0, 0 };
    context.ClearView(m_SSAO_RTV, cleared_ssao, nullptr, 0);

    return m_SSAO_SRV;
}

} // NS SSAO
} // NS DX11Renderer