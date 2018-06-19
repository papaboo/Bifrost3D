// DirectX 11 renderer screen space ambient occlusion implementations.
// ------------------------------------------------------------------------------------------------
// Copyright (C) 2018, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#include <DX11Renderer/SSAO.h>
#include <DX11Renderer/Utils.h>

namespace DX11Renderer {
namespace SSAO {

// ------------------------------------------------------------------------------------------------
// The Alchemy screen-space ambient obscurance algorithm.
// http://casual-effects.com/research/McGuire2011AlchemyAO/index.html
// ------------------------------------------------------------------------------------------------
AlchemyAO::AlchemyAO(ID3D11Device1& device, const std::wstring& shader_folder_path) 
    : m_width(0), m_height(0), m_SSAO_RTV(nullptr), m_SSAO_SRV(nullptr) {

    OBlob vertex_shader_blob = compile_shader(shader_folder_path + L"SSAO.hlsl", "vs_5_0", "main_vs");
    THROW_ON_FAILURE(device.CreateVertexShader(UNPACK_BLOB_ARGS(vertex_shader_blob), nullptr, &m_vertex_shader));

    OBlob pixel_shader_blob = compile_shader(shader_folder_path + L"SSAO.hlsl", "ps_5_0", "alchemy_ps");
    THROW_ON_FAILURE(device.CreatePixelShader(UNPACK_BLOB_ARGS(pixel_shader_blob), nullptr, &m_pixel_shader));

    D3D11_SAMPLER_DESC sampler_desc = {};
    sampler_desc.Filter = D3D11_FILTER_MIN_MAG_MIP_POINT;
    sampler_desc.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
    sampler_desc.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
    sampler_desc.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
    sampler_desc.ComparisonFunc = D3D11_COMPARISON_NEVER;
    sampler_desc.MinLOD = 0;
    sampler_desc.MaxLOD = D3D11_FLOAT32_MAX;

    THROW_ON_FAILURE(device.CreateSamplerState(&sampler_desc, &m_point_sampler));
}

OShaderResourceView& AlchemyAO::apply(ID3D11DeviceContext1& context, OShaderResourceView& normals, OShaderResourceView& depth, int width, int height) {
    if (m_width != width || m_height != height) {
        // Resize backbuffer
        ODevice1 device = get_device1(context);
        create_texture_2D(device, DXGI_FORMAT_R16G16B16A16_FLOAT, width, height, &m_SSAO_SRV, nullptr, &m_SSAO_RTV);

        m_width = width;
        m_height = height;
    }

    // Setup state.
    context.OMSetRenderTargets(1, &m_SSAO_RTV, nullptr);
    ID3D11ShaderResourceView* SRVs[2] = { normals, depth };
    context.PSSetShaderResources(0, 2, SRVs);
    context.PSSetSamplers(14, 1, &m_point_sampler);

    // Compute SSAO.
    context.VSSetShader(m_vertex_shader, 0, 0);
    context.PSSetShader(m_pixel_shader, 0, 0);
    context.Draw(3, 0);
    
    // Unbind SSAO_RTV
    ID3D11RenderTargetView* null_RTV = nullptr;
    context.OMSetRenderTargets(1, &null_RTV, nullptr);

    return m_SSAO_SRV;
}

OShaderResourceView& AlchemyAO::apply_none(ID3D11DeviceContext1& context, int width, int height) {
    if (m_width != width || m_height != height) {
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